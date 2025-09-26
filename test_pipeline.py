import cv2
import pipeline
import os

print("Testing the Full Pipeline")

DEVICE = "cuda"
MODEL = pipeline.load_model(device=DEVICE)

IMAGE_PATH = "D:/Vedhanth/studies/Coding/Hackathon/Runway_Detection/RUNWAY_DATASET/640x360/640x360/test/4AK606_1_4LDImage3.png"
if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Test image not found at {IMAGE_PATH}")
else:
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pipeline.run_full_pipeline(image, MODEL, device=DEVICE)

    if results:
        print("\nAnalysis Complete")
        print(f"Anchor Score: {results['anchor_score']:.4f}")
        print(f"Boolean Score: {results['boolean_score']}")
        print(f"LEDG Coords: {results['ledg_coords']}")
        
        result_image_bgr = cv2.cvtColor(results['visual_result'], cv2.COLOR_RGB2BGR)
        cv2.imshow("Test Result", result_image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()