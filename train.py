import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 25
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
MODEL_SAVE_PATH = "best_model.pth"

BASE_DIR = 'D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET'
RESOLUTION_FOLDER = '640x360'

TRAIN_IMG_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, RESOLUTION_FOLDER, 'train')
VAL_IMG_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, RESOLUTION_FOLDER, 'test')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, 'train_masks')
VAL_MASK_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, 'test_masks')

class RunwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask[mask > 0] = 1.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)

        return image, mask

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    model.train()
    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def check_iou(loader, model):
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice_score += iou
    final_iou = dice_score / len(loader)
    print(f"Validation IoU score: {final_iou:.4f}")
    model.train()
    return final_iou

def main():
    print(f"Using device: {DEVICE}")

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = RunwayDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_ds = RunwayDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    best_iou = -1.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        current_iou = check_iou(val_loader, model)
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ New best model saved with IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
