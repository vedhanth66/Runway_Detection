import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def generate_masks(json_path, output_dir, resolution):

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded data for {len(data)} images from {json_path}")

    Height, width = resolution

    for image_filename, shapes in tqdm(data.items(), desc=f"Generating masks for {os.path.basename(json_path)}"):
        mask = np.zeros((Height, width), dtype=np.uint8)

        ledg_points = None
        redg_points = None

        for shape in shapes:
            if shape['label'] == 'LEDG':
                ledg_points = shape['points']
            elif shape['label'] == 'REDG':
                redg_points = shape['points']
        
        if ledg_points and redg_points:
            top_left = ledg_points[0]
            bottom_left = ledg_points[1]
            top_right = redg_points[0]
            bottom_right = redg_points[1]

            runway_polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            
            cv2.fillPoly(mask, [runway_polygon], color=(255))

        output_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(output_path, mask)


if __name__ == '__main__':
    BASE_DIR = 'RUNWAY_DATASET'
    RESOLUTION_FOLDER = '640x360'
    RESOLUTION_DIMENSIONS = (360, 640)

    train_json_path = os.path.join(BASE_DIR, 'labels', 'labels', 'lines', 'train_labels_640x360.json')
    train_masks_output_dir = os.path.join(BASE_DIR, RESOLUTION_FOLDER, 'train_masks')

    test_json_path = os.path.join(BASE_DIR, 'labels', 'labels', 'lines', 'test_labels_640x360.json')
    test_masks_output_dir = os.path.join(BASE_DIR, RESOLUTION_FOLDER, 'test_masks')

    print("--- Starting Data Preprocessing (Corrected) ---")
    generate_masks(train_json_path, train_masks_output_dir, RESOLUTION_DIMENSIONS)
    generate_masks(test_json_path, test_masks_output_dir, RESOLUTION_DIMENSIONS)
    
    print("\n--- Preprocessing Complete! ---")
    print(f"Training masks saved to: {train_masks_output_dir}")
    print(f"Testing masks saved to: {test_masks_output_dir}")