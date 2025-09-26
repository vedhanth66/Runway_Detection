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

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 8
num_epochs = 25
image_height = 360
image_width = 640
model_save_path = "best_model.pth"

base_dir = 'D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET'
resolution_folder = '640x360'

train_img_dir = os.path.join(base_dir, resolution_folder, resolution_folder, 'train')
val_img_dir = os.path.join(base_dir, resolution_folder, resolution_folder, 'test')
train_mask_dir = os.path.join(base_dir, resolution_folder, 'train_masks')
val_mask_dir = os.path.join(base_dir, resolution_folder, 'test_masks')

class runway_dataset(Dataset):
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
        data, targets = data.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
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
            x, y = x.to(device), y.to(device)
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
    print(f"Using device: {device}")

    train_transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=image_height, width=image_width),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = runway_dataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_ds = runway_dataset(val_img_dir, val_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    best_iou = -1.0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        current_iou = check_iou(val_loader, model)
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()