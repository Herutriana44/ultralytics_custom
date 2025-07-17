import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
from glob import glob
from models.backbone.yolo_backbone import yolo_swin_medium

# --- Dataset Loader ---
class YOLODetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, augment=False):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*')))
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)
        # Load label
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        # Convert to absolute coords (for loss)
                        x, y, w, h = x * self.img_size, y * self.img_size, w * self.img_size, h * self.img_size
                        boxes.append([cls, x, y, w, h])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        return img, boxes

# --- Detection Head ---
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, num_anchors=1):
        super().__init__()
        # Output: [B, num_anchors*(5+num_classes), H/32, W/32]
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W]
        out = self.conv(x)
        B, _, H, W = out.shape
        out = out.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        return out

# --- Model ---
class YOLOSwinDetector(nn.Module):
    def __init__(self, backbone=None, num_classes=1):
        super().__init__()
        self.backbone = backbone if backbone is not None else yolo_swin_medium()
        # Dapatkan channel output terakhir backbone secara dinamis
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            feats = self.backbone(dummy)
        out_channels = feats[-1].shape[1]
        self.head = DetectionHead(in_channels=out_channels, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats[-1])
        return out

# --- Loss Function (YOLO-style, simplified) ---
def bbox_iou(box1, box2, eps=1e-7):
    # box: [x, y, w, h] (center, width, height)
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    area1 = (b1_x2 - b1_x1).clamp(0) * (b1_y2 - b1_y1).clamp(0)
    area2 = (b2_x2 - b2_x1).clamp(0) * (b2_y2 - b2_y1).clamp(0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    return iou

def yolo_loss(pred, targets, img_size=640, num_classes=1, device='cpu'):
    # pred: [B, A, 5+num_classes, H, W]
    # targets: [B, N, 5] (cls, x, y, w, h)
    obj_loss = 0
    box_loss = 0
    cls_loss = 0
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for b in range(pred.shape[0]):
        p = pred[b]  # [A, 5+num_classes, H, W]
        t = targets[b]  # [N, 5]
        if t.shape[0] == 0:
            # No object
            obj_loss += bce(p[:, 4, :, :], torch.zeros_like(p[:, 4, :, :]))
            continue
        # For each target, find best anchor/grid
        for target in t:
            cls, x, y, w, h = target
            gx, gy = x / img_size * p.shape[2], y / img_size * p.shape[3]
            gi, gj = int(gx), int(gy)
            # Clamp to grid
            gi = min(max(gi, 0), p.shape[2] - 1)
            gj = min(max(gj, 0), p.shape[3] - 1)
            # Box prediction at grid
            pred_box = p[0, :4, gj, gi]
            pred_obj = p[0, 4, gj, gi]
            pred_cls = p[0, 5:, gj, gi]
            # Target box (normalized to grid)
            tx = gx - gi
            ty = gy - gj
            tw = w / img_size
            th = h / img_size
            target_box = torch.tensor([tx, ty, tw, th], device=device)
            # Box loss (MSE)
            box_loss += mse(pred_box, target_box)
            # Objectness loss
            obj_loss += bce(pred_obj, torch.ones_like(pred_obj))
            # Class loss
            cls_loss += bce(pred_cls, torch.zeros_like(pred_cls))  # 1 class, always 0
        # Background
        obj_loss += bce(p[:, 4, :, :], torch.zeros_like(p[:, 4, :, :])) * 0.1
    return box_loss, obj_loss, cls_loss

# --- Training Script ---
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Paths
    DATASET_DIR = '../dataset_medis'
    IMG_SIZE = 640
    BATCH_SIZE = 4
    EPOCHS = 50
    NUM_CLASSES = 1
    # Dataset
    train_dataset = YOLODetectionDataset(
        os.path.join(DATASET_DIR, 'images/train'),
        os.path.join(DATASET_DIR, 'labels/train'),
        img_size=IMG_SIZE
    )
    val_dataset = YOLODetectionDataset(
        os.path.join(DATASET_DIR, 'images/val'),
        os.path.join(DATASET_DIR, 'labels/val'),
        img_size=IMG_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # Model
    model = YOLOSwinDetector(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]
            preds = model(imgs)
            box_loss, obj_loss, cls_loss = yolo_loss(preds, targets, img_size=IMG_SIZE, num_classes=NUM_CLASSES, device=device)
            loss = box_loss + obj_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = [t.to(device) for t in targets]
                preds = model(imgs)
                box_loss, obj_loss, cls_loss = yolo_loss(preds, targets, img_size=IMG_SIZE, num_classes=NUM_CLASSES, device=device)
                loss = box_loss + obj_loss + cls_loss
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Val Loss: {avg_val_loss:.4f}")
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_yoloswin_detector.pt')
                print("  Model saved!")

if __name__ == '__main__':
    train() 