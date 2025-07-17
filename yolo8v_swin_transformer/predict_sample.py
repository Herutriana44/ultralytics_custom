import torch
import cv2
import numpy as np
import os
from glob import glob
from models.backbone.yolo_backbone import yolo_swin_medium
from train_custom_yolo_swin import YOLOSwinDetector, DetectionHead

class YOLOPredictor:
    def __init__(self, model_path='best_yoloswin_detector.pt', img_size=640, conf_threshold=0.5, iou_threshold=0.45):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model with same architecture as training
        model = YOLOSwinDetector(num_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        # Store original dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Resize and normalize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, img, (orig_h, orig_w)
    
    def postprocess_predictions(self, predictions, orig_shape):
        """Convert model output to bounding boxes"""
        orig_h, orig_w = orig_shape
        
        # predictions: [B, A, 5+num_classes, H, W]
        pred = predictions[0]  # [A, 5+num_classes, H, W]
        
        boxes = []
        confidences = []
        
        # Extract predictions from each grid cell
        for anchor in range(pred.shape[0]):
            for grid_y in range(pred.shape[2]):
                for grid_x in range(pred.shape[3]):
                    # Get predictions for this cell
                    x_center = pred[anchor, 0, grid_y, grid_x].item()
                    y_center = pred[anchor, 1, grid_y, grid_x].item()
                    width = pred[anchor, 2, grid_y, grid_x].item()
                    height = pred[anchor, 3, grid_y, grid_x].item()
                    confidence = torch.sigmoid(pred[anchor, 4, grid_y, grid_x]).item()
                    
                    # Filter by confidence
                    if confidence > self.conf_threshold:
                        # Convert grid coordinates to image coordinates
                        x_center = (x_center + grid_x) * self.img_size / pred.shape[3]
                        y_center = (y_center + grid_y) * self.img_size / pred.shape[2]
                        width = width * self.img_size
                        height = height * self.img_size
                        
                        # Convert to corner coordinates
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        # Scale to original image size
                        x1 = x1 * orig_w / self.img_size
                        y1 = y1 * orig_h / self.img_size
                        x2 = x2 * orig_w / self.img_size
                        y2 = y2 * orig_h / self.img_size
                        
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(confidence)
        
        # Apply non-maximum suppression
        if boxes:
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            
            # Simple NMS implementation
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                confidences.tolist(), 
                self.conf_threshold, 
                self.iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = boxes[indices]
                confidences = confidences[indices]
            else:
                boxes = np.array([])
                confidences = np.array([])
        else:
            boxes = np.array([])
            confidences = np.array([])
        
        return boxes, confidences
    
    def draw_predictions(self, img, boxes, confidences):
        """Draw bounding boxes on image"""
        img_draw = img.copy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"medis: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img_draw
    
    def predict_single_image(self, img_path, save_result=True):
        """Predict on single image"""
        print(f"Predicting: {img_path}")
        
        # Preprocess
        img_tensor, img, orig_shape = self.preprocess_image(img_path)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Postprocess
        boxes, confidences = self.postprocess_predictions(predictions, orig_shape)
        
        # Draw results
        result_img = self.draw_predictions(img, boxes, confidences)
        
        # Save result
        if save_result:
            output_path = img_path.replace('.', '_predicted.')
            cv2.imwrite(output_path, result_img)
            print(f"Result saved to: {output_path}")
        
        # Print results
        print(f"Found {len(boxes)} objects")
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            print(f"  Object {i+1}: Box {box.astype(int)}, Confidence: {conf:.3f}")
        
        return boxes, confidences, result_img
    
    def predict_folder(self, folder_path, save_results=True):
        """Predict on all images in folder"""
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob(os.path.join(folder_path, ext.upper())))
        
        if not image_paths:
            print(f"No images found in folder: {folder_path}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        results = []
        for img_path in image_paths:
            try:
                boxes, confidences, result_img = self.predict_single_image(img_path, save_results)
                results.append({
                    'image_path': img_path,
                    'boxes': boxes,
                    'confidences': confidences,
                    'num_detections': len(boxes)
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Summary
        total_detections = sum(r['num_detections'] for r in results)
        print(f"\nSummary: {len(results)} images processed, {total_detections} total detections")
        
        return results

def main():
    # Example usage
    predictor = YOLOPredictor(
        model_path='best_yoloswin_detector.pt',
        conf_threshold=0.5,
        iou_threshold=0.45
    )
    
    # Option 1: Predict single image
    # predictor.predict_single_image('sample_image.jpg')
    
    # Option 2: Predict folder of images
    sample_folder = '../dataset_medis/images/val'  # Use validation images as samples
    if os.path.exists(sample_folder):
        predictor.predict_folder(sample_folder)
    else:
        print(f"Sample folder not found: {sample_folder}")
        print("Please provide a valid image path or folder path")

if __name__ == '__main__':
    main() 