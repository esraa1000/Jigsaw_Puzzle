import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def run_preprocessing(image_path, save_steps=True, steps_dir="./output/steps"):
    steps = {}

    # 1. Read image
    img = cv2.imread(image_path)
    steps["Original"] = img.copy()

    # 2. Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["Gray"] = gray.copy()

    # 3. CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    steps["CLAHE"] = clahe_img.copy()

    # 4. Gaussian blur
    blur = cv2.GaussianBlur(clahe_img, (5,5), 0)
    steps["Blurred"] = blur.copy()

    # 5. Canny
    edges = cv2.Canny(blur, 75, 150)
    steps["Edges"] = edges.copy()

    # Save all steps as images (optional)
    if save_steps:
        os.makedirs(steps_dir, exist_ok=True)
        for name, img in steps.items():
            cv2.imwrite(f"{steps_dir}/{name}.png", img)

    return edges, steps
