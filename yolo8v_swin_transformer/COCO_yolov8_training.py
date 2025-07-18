import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
from models.backbone.yolo_backbone import yolo_swin_medium

# --- COCO Dataset Loader ---
class CocoDetectionCustom(CocoDetection):
    def __init__(self, img_folder, ann_file, img_size=640, transform=None):
        super().__init__(img_folder, ann_file)
        self.img_size = img_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = self.transform(img)
        # target: list of dicts, each with bbox, category_id, etc.
        boxes = []
        for obj in target:
            x, y, w, h = obj['bbox']
            x_c = x + w / 2
            y_c = y + h / 2
            # Normalize to 0-1
            x_c /= self.img_size
            y_c /= self.img_size
            w /= self.img_size
            h /= self.img_size
            cls = obj['category_id'] - 1  # COCO class ids start at 1
            boxes.append([cls, x_c, y_c, w, h])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        return img, boxes

# --- Detection Head & Model (sama seperti sebelumnya) ---
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=80, num_anchors=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv(x)
        B, _, H, W = out.shape
        out = out.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        return out

class YOLOSwinDetector(nn.Module):
    def __init__(self, backbone=None, num_classes=80):
        super().__init__()
        self.backbone = backbone if backbone is not None else yolo_swin_medium()
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
def yolo_loss(pred, targets, img_size=640, num_classes=80, device='cpu'):
    obj_loss = 0
    box_loss = 0
    cls_loss = 0
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for b in range(pred.shape[0]):
        p = pred[b]  # [A, 5+num_classes, H, W]
        t = targets[b]  # [N, 5]
        if t.shape[0] == 0:
            obj_loss += bce(p[:, 4, :, :], torch.zeros_like(p[:, 4, :, :]))
            continue
        for target in t:
            cls, x, y, w, h = target
            gx, gy = x * p.shape[2], y * p.shape[3]
            gi, gj = int(gx), int(gy)
            gi = min(max(gi, 0), p.shape[2] - 1)
            gj = min(max(gj, 0), p.shape[3] - 1)
            pred_box = p[0, :4, gj, gi]
            pred_obj = p[0, 4, gj, gi]
            pred_cls = p[0, 5:, gj, gi]
            tx = gx - gi
            ty = gy - gj
            tw = w
            th = h
            target_box = torch.tensor([tx, ty, tw, th], device=device)
            box_loss += mse(pred_box, target_box)
            obj_loss += bce(pred_obj, torch.ones_like(pred_obj))
            cls_target = torch.zeros(num_classes, device=device)
            cls_target[int(cls)] = 1.0
            cls_loss += bce(pred_cls, cls_target)
        obj_loss += bce(p[:, 4, :, :], torch.zeros_like(p[:, 4, :, :])) * 0.1
    return box_loss, obj_loss, cls_loss

# --- Training Script ---
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Ganti path berikut sesuai dataset COCO eksternal kamu
    COCO_TRAIN_IMG = '/content/train2014'
    COCO_TRAIN_ANN = '/content/annotations/instances_train2014.json'
    COCO_VAL_IMG = '/content/val2014'
    COCO_VAL_ANN = '/content/annotations/instances_val2014.json'
    IMG_SIZE = 640
    BATCH_SIZE = 4
    EPOCHS = 50

    # Dapatkan jumlah kelas dari file COCO
    coco = COCO(COCO_TRAIN_ANN)
    num_classes = len(coco.getCatIds())

    train_dataset = CocoDetectionCustom(COCO_TRAIN_IMG, COCO_TRAIN_ANN, img_size=IMG_SIZE)
    val_dataset = CocoDetectionCustom(COCO_VAL_IMG, COCO_VAL_ANN, img_size=IMG_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = YOLOSwinDetector(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]
            preds = model(imgs)
            box_loss, obj_loss, cls_loss = yolo_loss(preds, targets, img_size=IMG_SIZE, num_classes=num_classes, device=device)
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
                box_loss, obj_loss, cls_loss = yolo_loss(preds, targets, img_size=IMG_SIZE, num_classes=num_classes, device=device)
                loss = box_loss + obj_loss + cls_loss
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_yoloswin_detector_coco.pt')
                print("  Model saved!")

if __name__ == '__main__':
    train()