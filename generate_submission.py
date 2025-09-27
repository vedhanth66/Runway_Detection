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
    return (intersection + 1e-6) / (union + 1e-6)


def main():
    print("Starting Final Submission Generation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = 'D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET'
    resolution_folder = '640x360'
    test_img_dir = os.path.join(base_dir, resolution_folder, resolution_folder, 'test')
    gt_mask_dir = os.path.join(base_dir, resolution_folder, 'test_masks')
    output_csv_path = "submission.csv"

    model = pipeline.load_model(device=device)
    test_image_files = os.listdir(test_img_dir)
    results_list = []

    for image_file in tqdm(test_image_files, desc="Processing Test Set"):
        image_path = os.path.join(test_img_dir, image_file)
        gt_mask_path = os.path.join(gt_mask_dir, image_file)

        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask not found for {image_file}. Skipping.")
            continue

        image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        gt_mask_np = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

        results = pipeline.run_full_pipeline(image_np, model, device=device)

        transform = pipeline.A.Compose([
            pipeline.A.Resize(height=360, width=640),
            pipeline.A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            pipeline.A.pytorch.ToTensorV2(),
        ])
        input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_tensor = torch.sigmoid(model(input_tensor))
            pred_mask_np = (pred_mask_tensor > 0.5).float().cpu().numpy().squeeze()

        true_iou = calculate_true_iou(pred_mask_np, gt_mask_np)

        results_list.append({
            "image_name": image_file,
            "iou_score": true_iou,
            "anchor_score": results["anchor_score"],
            "boolean_score": int(results["boolean_score"])
        })

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv_path, index=False)

    print("\n--- Submission Generation Complete! ---")
    print(f"Final results saved to: {output_csv_path}")
    print("\nFinal Mean Scores:")
    print(df[['iou_score', 'anchor_score', 'boolean_score']].mean())


if __name__ == "__main__":
    main()
 