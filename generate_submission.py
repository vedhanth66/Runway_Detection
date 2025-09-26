import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pipeline  

def calculate_true_iou(pred_mask, gt_mask):
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0
    
    intersection = np.sum(pred_mask_bool & gt_mask_bool)
    union = np.sum(pred_mask_bool | gt_mask_bool)
    
    iou_score = (intersection + 1e-6) / (union + 1e-6)
    return iou_score

def main():
    print("Starting Final Submission Generation")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    BASE_DIR = 'D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET'
    RESOLUTION_FOLDER = '640x360'
    TEST_IMG_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, RESOLUTION_FOLDER, 'test')
    GT_MASK_DIR = os.path.join(BASE_DIR, RESOLUTION_FOLDER, 'test_masks') 
    
    OUTPUT_CSV_PATH = "submission.csv"

    MODEL = pipeline.load_model(device=DEVICE)

    test_image_files = os.listdir(TEST_IMG_DIR)
    results_list = []

    for image_file in tqdm(test_image_files, desc="Processing Test Set"):
        image_path = os.path.join(TEST_IMG_DIR, image_file)
        gt_mask_path = os.path.join(GT_MASK_DIR, image_file)

        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask not found for {image_file}. Skipping.")
            continue

        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        gt_mask_np = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

        results = pipeline.run_full_pipeline(image_np, MODEL, device=DEVICE)

        transform = pipeline.A.Compose([
            pipeline.A.Resize(height=360, width=640),
            pipeline.A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            pipeline.A.pytorch.ToTensorV2(),
        ])
        input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_mask_tensor = torch.sigmoid(MODEL(input_tensor))
            pred_mask_np = (pred_mask_tensor > 0.5).float().cpu().numpy().squeeze()
            
        true_iou = calculate_true_iou(pred_mask_np, gt_mask_np)

        results_list.append({
            "Image Name": image_file,
            "IOU score": true_iou,
            "Anchor Score": results["anchor_score"],
            "Boolen_score": int(results["boolean_score"]) 
        })

    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- Submission Generation Complete! ---")
    print(f"Final results saved to: {OUTPUT_CSV_PATH}")
    print("\nFinal Mean Scores:")
    print(df[['IOU score', 'Anchor Score', 'Boolen_score']].mean())

if __name__ == "__main__":
    main()