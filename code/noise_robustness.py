"""
Evaluate model robustness to additive Gaussian noise.

This script is the cleaned, non-plotting version of the noise-evaluation
workflow.  The model is selected with MODEL_NAME and the corresponding
model/scaler paths are defined in MODEL_REGISTRY.

The test dataset is not resampled: every SNR is evaluated on the same
test samples, with newly generated Gaussian noise.
"""

from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models import build_model


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("./data")

# Select the model to evaluate.
MODEL_NAME = "Custom CNN"

# Update these paths to the trained models used
MODEL_REGISTRY = {
    "Custom CNN": {
        "model_path": "./output/CustomCNN/saved_models/best_model_fold1.pt",
        "scaler_path": "./output/CustomCNN/StandardScaler.pkl",
    },
    # Example for adding the other architectures:
    #"VGG16": {
    #    "model_path": "./output/VGG16/saved_models/best_model_fold8.pt",
    #    "scaler_path": "./output/VGG16/StandardScaler.pkl",
    #},
    # "ResNet18": {...},
    # "EfficientNetB0": {...},
    # "MobileNetV3": {...},
}

SNR_LEVELS = [10, 15, 20, 25, 30, 35, 40, 45, 50]

BATCH_SIZE = 16
IMAGE_SIZE = 128
ORIGINAL_IMAGE_SIZE = 512

TARGET_COLUMNS = [
    "c1_m1tx", "c1_m1ty",
    "c2_m1tx", "c2_m1ty",
    "c3_m1tx", "c3_m1ty",
    "c4_m1tx", "c4_m1ty",
    "c1_m1p", "c2_m1p",
    "c3_m1p", "c4_m1p",
]

TILT_X_IDX = [0, 2, 4, 6]
TILT_Y_IDX = [1, 3, 5, 7]
PISTON_IDX = [8, 9, 10, 11]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_gaussian_noise_snr(image, snr_db):
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*image.shape) * np.sqrt(noise_power)
    return image + noise


class NoisyTestDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, snr_db, scaler=None):
        self.images = images
        self.targets = targets
        self.snr_db = snr_db
        self.scaler = scaler
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32).reshape(ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE)
        image = add_gaussian_noise_snr(image, self.snr_db)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, axis=0)
                    
        target = self.targets[idx]      
        if self.scaler is not None:
            target = self.scaler.transform(target.reshape(1,-1)).reshape(-1)

        return (torch.from_numpy(image).float(), torch.from_numpy(target).float())


def load_data(data_dir):
    files = sorted(Path(data_dir).glob("*.feather"))
    if not files:
        raise FileNotFoundError(f"No Feather files found in {data_dir}")

    data = pd.concat(
        [pd.read_feather(path) for path in files],
        ignore_index=True,
    )

    images = data["psf_flat"].values
    targets = data[TARGET_COLUMNS].values

    return images, targets


def evaluate_model(model, images, targets, scaler, snr_db):
    dataset = NoisyTestDataset(images, targets, snr_db, scaler)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []
    scaled_targets = []

    model.eval()

    with torch.no_grad():
        for inputs, batch_targets in loader:
            outputs = model(inputs.to(DEVICE))
            predictions.append(outputs.cpu().numpy())
            scaled_targets.append(batch_targets.numpy())

    predictions = np.vstack(predictions)
    scaled_targets = np.vstack(scaled_targets)

    predictions = scaler.inverse_transform(predictions)
    true_targets = scaler.inverse_transform(scaled_targets)

    error = predictions - true_targets

    # ==========================================================
    # Tilt X - RMSE for each mirror followed by the mean and std across the 4 mirrors
    # ==========================================================
    rmse_tiltX_mirror = np.sqrt(np.mean(error[:, TILT_X_IDX] ** 2, axis=0))
    rmse_tiltX = np.mean(rmse_tiltX_mirror)
    rmse_tiltX_std = np.std(rmse_tiltX_mirror, ddof=0)

    # ==========================================================
    # Tilt Y
    # ==========================================================
    rmse_tiltY_mirror = np.sqrt(np.mean(error[:, TILT_Y_IDX] ** 2, axis=0))
    rmse_tiltY = np.mean(rmse_tiltY_mirror)
    rmse_tiltY_std = np.std(rmse_tiltY_mirror, ddof=0)

    # ==========================================================
    # Piston
    # ==========================================================
    rmse_piston_mirror = np.sqrt(np.mean(error[:, PISTON_IDX] ** 2, axis=0))
    rmse_piston = np.mean(rmse_piston_mirror)
    rmse_piston_std = np.std(rmse_piston_mirror, ddof=0)

    return {
        "snr_db": snr_db,
        "rmse_tiltX": float(rmse_tiltX),
        "rmse_tiltX_std": float(rmse_tiltX_std),
        "rmse_tiltY": float(rmse_tiltY),
        "rmse_tiltY_std": float(rmse_tiltY_std),
        "rmse_piston": float(rmse_piston),
        "rmse_piston_std": float(rmse_piston_std),
    }


def main():
    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(
            f"{MODEL_NAME!r} is not present in MODEL_REGISTRY."
        )

    info = MODEL_REGISTRY[MODEL_NAME]

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    images, targets = load_data(DATA_DIR)
    scaler = joblib.load(info["scaler_path"])

    model = build_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(info["model_path"], map_location=DEVICE))
    model.eval()

    results = []

    for snr_db in SNR_LEVELS:
        print(f"Evaluating SNR = {snr_db} dB")
        result = evaluate_model(model, images, targets, scaler, snr_db)
        results.append(result)

    output_path = Path("./output/noise_robustness")
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{MODEL_NAME.replace(' ', '_')}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also save a compact CSV for plotting later
    pd.DataFrame(results).to_csv(
        output_path / f"{MODEL_NAME.replace(' ', '_')}.csv",
        index=False,
    )

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
