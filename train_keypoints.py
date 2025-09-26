import os
import cv2
import torch
import json
import numpy as np
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from tqdm import tqdm

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 40
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
MODEL_SAVE_PATH = "best_keypoint_model.pth"
NUM_OUTPUTS = 8 # 4 points * 2 coordinates (x, y)

BASE_DIR = 'D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET'
RESOLUTION_FOLDER = '640x360'

TRAIN_IMG_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, RESOLUTION_FOLDER, 'train')
VAL_IMG_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, RESOLUTION_FOLDER, 'test')
TRAIN_JSON_PATH = os.path.join(BASE_DIR, 'labels', 'labels', 'lines', 'train_labels_640x360.json')
VAL_JSON_PATH = os.path.join(BASE_DIR, 'labels', 'labels', 'lines', 'test_labels_640x360.json')

# --- 2. Custom Keypoint Dataset Class ---
class KeypointDataset(Dataset):
    def __init__(self, image_dir, json_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        
        self.image_files = []
        for filename, shapes in self.json_data.items():
            if os.path.exists(os.path.join(image_dir, filename)):
                has_ledg = any(shape['label'] == 'LEDG' for shape in shapes)
                has_redg = any(shape['label'] == 'REDG' for shape in shapes)
                if has_ledg and has_redg:
                    self.image_files.append(filename)
        
        print(f"Found {len(self.json_data.keys())} total labels, but filtered down to {len(self.image_files)} images with complete keypoint data.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        img_path = os.path.join(self.image_dir, image_filename)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        shapes = self.json_data[image_filename]
        ledg_points, redg_points = None, None
        for shape in shapes:
            if shape['label'] == 'LEDG':
                ledg_points = shape['points']
            elif shape['label'] == 'REDG':
                redg_points = shape['points']

        keypoints = np.array([
            ledg_points[0], redg_points[0],
            redg_points[1], ledg_points[1]
        ], dtype=np.float32)

        # --- CORRECTED LOGIC ---
        # 1. Apply augmentations to the image ONLY
        if self.transform:
            image = self.transform(image=image)['image']
        
        # 2. Normalize the ORIGINAL keypoints
        keypoints[:, 0] /= IMAGE_WIDTH
        keypoints[:, 1] /= IMAGE_HEIGHT
        
        # 3. Convert keypoints to a tensor and flatten
        keypoints_tensor = torch.from_numpy(keypoints.flatten())
        
        return image, keypoints_tensor

# --- 3. Training Loop ---
def train_loop(model, train_loader, val_loader):
    loss_fn = MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Keypoint Model: Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        loop = tqdm(train_loader, desc="Training Keypoints")
        for images, keypoints in loop:
            images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
                preds = model(images)
                loss = loss_fn(preds, keypoints)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, keypoints in tqdm(val_loader, desc="Validating Keypoints"):
                images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
                preds = model(images)
                val_loss += loss_fn(preds, keypoints).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ New best keypoint model saved with loss: {best_loss:.6f}")

# --- 4. Main Execution ---
def main():
    print(f"Using device: {DEVICE}")
    
    # --- CORRECTED: No keypoint_params needed as we handle it manually ---
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_ds = KeypointDataset(TRAIN_IMG_DIR, TRAIN_JSON_PATH, transform=transform)
    val_ds = KeypointDataset(VAL_IMG_DIR, VAL_JSON_PATH, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_OUTPUTS)
    model.to(DEVICE)

    train_loop(model, train_loader, val_loader)

if __name__ == "__main__":
    main()