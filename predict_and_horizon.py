import os
import cv2
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage
from google.colab import drive

#2. Config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = '/content/Deeplab_V3_dataset' 

MODEL_PATH_BEST = os.path.join(DATA_DIR, 'deeplabv3plus_horizon_best4.pth')
MODEL_PATH_LAST = os.path.join(DATA_DIR, 'deeplabv3plus_horizon_last_epoch4.pth')
IMAGE_DIR = os.path.join(DATA_DIR, 'sample_test_images')
OUTPUT_DIR = os.path.join(DATA_DIR, 'predictions_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. HELPER FUNCTIONS

def get_transforms():
    """Returns the Albumentations transform for inference."""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model(encoder_name, model_path_best, model_path_last, device):
    "Initializes and loads the trained model."
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path_best, map_location=device))
        print(f"Successfully loaded best model from {model_path_best}")
    except FileNotFoundError:
        print(f"Best model not found. Trying last epoch model...")
        try:
            model.load_state_dict(torch.load(model_path_last, map_location=device))
            print(f"Successfully loaded last model from {model_path_last}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    model.eval()
    return model

def get_image_files_recursive(root_folder):
    "Recursively finds all image files in a directory."
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(dirpath, f))
    return sorted(files)

def extract_horizon(mask):
    "Extracts a horizon line from a segmentation mask."
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, 8, cv2.CV_32S)
    if num_labels <= 1:
        return np.zeros(mask.shape[1], dtype=int) - 1
    largest_component_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_component_label = i
    largest_component_mask = np.zeros_like(mask_cleaned, dtype=np.uint8)
    largest_component_mask[labels == largest_component_label] = 1
    blurred_mask = ndimage.gaussian_filter(largest_component_mask.astype(float), sigma=2)
    horizon = np.zeros(mask.shape[1], dtype=int) - 1
    for col in range(mask.shape[1]):
        column = blurred_mask[:, col]
        gradient = np.abs(np.diff(column))
        if np.max(gradient) > 0.1:
            peak_idx = np.argmax(gradient)
            horizon[col] = peak_idx
    return horizon

def fit_horizon_line(horizon, img_shape):
    "Fits a line to the extracted horizon points."
    valid_x, valid_y = zip(*[(x, y) for x, y in enumerate(horizon) if y != -1]) if any(y != -1 for y in horizon) else (
    [], [])
    if len(valid_x) < 10:
        return None
    points = np.array(list(zip(valid_x, valid_y)), dtype=np.int32)
    mean_y = np.mean(valid_y)
    std_y = np.std(valid_y)
    if std_y < 10:
        return [(0, int(mean_y)), (img_shape[1], int(mean_y))]
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    img_width, img_height = img_shape[1], img_shape[0]
    if abs(vx[0]) < 1e-6:
        return None
    lefty = int((0 - x0[0]) * (vy[0] / vx[0]) + y0[0])
    righty = int((img_width - 1 - x0[0]) * (vy[0] / vx[0]) + y0[0])
    lefty = np.clip(lefty, 0, img_height - 1)
    righty = np.clip(righty, 0, img_height - 1)
    slope = (righty - lefty) / img_width
    if abs(slope) > 0.1:
        avg_y = int(np.mean(valid_y))
        return [(0, avg_y), (img_width, avg_y)]
    return [(0, lefty), (img_width - 1, righty)]

def run_inference(model, transform, image_dir, output_dir):
    "Main function to run the inference loop."
    print(f"Reading images from: {image_dir}")
    print(f"Saving predictions to: {output_dir}")
    image_files = get_image_files_recursive(image_dir)
    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        orig_img_np = np.array(img)
        augmented = transform(image=orig_img_np)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)

        pred_mask_resized = cv2.resize(pred_mask, (orig_img_np.shape[1], orig_img_np.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        pred_mask_path = os.path.join(output_dir, f"mask_pred_{img_name}")
        cv2.imwrite(pred_mask_path, pred_mask_resized * 255)

        horizon = extract_horizon(pred_mask_resized)
        line_points = fit_horizon_line(horizon, orig_img_np.shape)

        if line_points is not None:
            fitted_img = orig_img_np.copy()
            cv2.line(fitted_img, line_points[0], line_points[1], (0, 0, 255), 2)
            out_path = os.path.join(output_dir, f"horizon_fitted_straight_{img_name}")
            cv2.imwrite(out_path, cv2.cvtColor(fitted_img, cv2.COLOR_RGB2BGR))
            print(f"Saved horizon-fitted image: {out_path}")

            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.imshow(pred_mask_resized, cmap='gray')
            plt.title(f"Predicted Mask - {img_name}")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(fitted_img)
            plt.title(f"Fitted Horizon - {img_name}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"fitted_horizon_plot_{img_name}.png"))
            plt.close()
            print(f"Saved visualization plot: fitted_horizon_plot_{img_name}.png")
        else:
            print(f"Could not fit a horizon line for {img_name}")
            raw_img = orig_img_np.copy()
            for x, y in enumerate(horizon):
                if y != -1:
                    cv2.circle(raw_img, (x, y), 1, (0, 0, 255), -1)
            out_path = os.path.join(output_dir, f"horizon_detected_raw_{img_name}")
            cv2.imwrite(out_path, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
            print(f"Saved raw horizon overlay image: {out_path}")

            plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            plt.imshow(pred_mask_resized, cmap='gray')
            plt.title(f"Predicted Mask - {img_name}")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(raw_img)
            plt.title(f"Raw Horizon Points - {img_name}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"raw_horizon_plot_{img_name}.png"))
            plt.close()
            print(f"Saved visualization plot: raw_horizon_plot_{img_name}.png")

    print("\nPrediction process completed!")

# 4. MAIN EXECUTION

if __name__ == "__main__":
    inference_transform = get_transforms()
    loaded_model = load_model("resnet50", MODEL_PATH_BEST, MODEL_PATH_LAST, DEVICE)
    run_inference(loaded_model, inference_transform, IMAGE_DIR, OUTPUT_DIR)
