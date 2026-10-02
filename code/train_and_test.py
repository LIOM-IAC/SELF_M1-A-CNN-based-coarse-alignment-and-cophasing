"""
Train and evaluate several CNN models.

This script keeps the train/validation/test split together.  The split
indices are saved so that a later run can reproduce exactly the same
partition.

Expected input:
    Feather files containing a column named ``psf_flat`` and the 12
    target columns listed in TARGET_COLUMNS.

The original notebook used:
    - 80/20 train+validation / test split
    - random_state=42
    - 10-fold CV on the training+validation set
    - StandardScaler fitted only on the training+validation targets
    - Adam, lr=1e-3, weight_decay=1e-5
    - batch size 16
    - maximum 200 epochs
    - early stopping patience 10
"""

from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models import build_model, available_models


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

MODEL_NAME = "Custom CNN"

RANDOM_STATE = 42
TEST_SIZE = 0.20

BATCH_SIZE = 16
EPOCHS = 200
K_FOLDS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10

# Fraction of additional noise-augmented samples in each training fold.
AUGMENTATION_FRACTION = 0.0
AUGMENTATION_SNR_MIN = 20.0
AUGMENTATION_SNR_MAX = 40.0

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

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

    return data, images, targets


def add_gaussian_noise_snr(image, snr_db):
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(*image.shape) * np.sqrt(noise_power)
    return image + noise


class augmented_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        targets,
        augmentation_fraction=AUGMENTATION_FRACTION,
        snr_min=AUGMENTATION_SNR_MIN,
        snr_max=AUGMENTATION_SNR_MAX,
        transform=None
    ):
        self.images = images
        self.targets = targets
        self.transform = transform
        self.n_original = len(images)
        self.n_augmented = int(self.n_original * augmentation_fraction)
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __len__(self):
        return self.n_original + self.n_augmented

    def __getitem__(self, idx):
        if idx < self.n_original:
            img = self.images[idx]
            target = self.targets[idx]

        else:
            source_idx = np.random.randint(0, self.n_original)
            img = self.images[source_idx].copy()
            target = self.targets[source_idx]

        img = img.astype(np.float32).reshape(ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE)

        if idx >= self.n_original:
            snr_db = np.random.uniform(self.snr_min, self.snr_max)
            img = add_gaussian_noise_snr(img, snr_db)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.from_numpy(img).float(), torch.tensor(target, dtype=torch.float32).reshape(-1)


def load_or_create_split(n_samples, split_path):
    if split_path.exists():
        split = np.load(split_path)
        return split["trainval_idx"], split["test_idx"]

    indices = np.arange(n_samples)
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    save_split(split_path, trainval_idx, test_idx)
    
    return trainval_idx, test_idx


def save_split(path, trainval_idx, test_idx):
    np.savez(
        path,
        trainval_idx=np.asarray(trainval_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------
# Training / validation functions
# ---------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)

        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss:.4f}")

    return avg_loss

def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation loss: {avg_loss:.4f}")

    return avg_loss

# ---------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------

def predict(model, loader):
    model.eval()

    predictions = []
    targets = []

    times_per_sample = []

    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            outputs = model(inputs)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            batch_time = end - start
            batch_size = inputs.size(0)

            time_per_sample = batch_time / batch_size
            times_per_sample.extend([time_per_sample] * batch_size)

            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    times_per_sample = np.array(times_per_sample)

    return predictions, targets, times_per_sample


def evaluate_test_set(model_name, model_path, test_images, test_targets, scaler):
    dataset = augmented_dataset(test_images, test_targets, augmentation_fraction=AUGMENTATION_FRACTION)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    predictions_scaled, targets_scaled, times_per_sample = predict(model, loader)

    predictions = scaler.inverse_transform(predictions_scaled)
    targets = scaler.inverse_transform(targets_scaled)

    metrics = {}
    for i, label in enumerate(TARGET_COLUMNS):
        metrics[label] = {
            "RMSE": float(np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))),
            "MAE": float(mean_absolute_error(targets[:, i], predictions[:, i])),
            "R2": float(r2_score(targets[:, i], predictions[:, i])),
        }
    
    # Inference time
    mean_time = times_per_sample.mean()
    std_time = times_per_sample.std()

    inference_time = {
        "mean_seconds_per_sample": float(mean_time),
        "std_seconds_per_sample": float(std_time),
        "mean_ms_per_sample": float(mean_time * 1000),
        "std_ms_per_sample": float(std_time * 1000),
    }

    return metrics, inference_time

# -------------------------------------------------
# K-fold training
# -------------------------------------------------

def main():
    
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Available models: {available_models()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data, images, targets = load_data(DATA_DIR)
    print(f"Loaded {len(data)} samples.")

    split_path = OUTPUT_DIR / "data_split.npz"
    trainval_idx, test_idx = load_or_create_split(len(images), split_path)

    print(f"Training + validation samples: {len(trainval_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Split saved to: {split_path}")

    scaler = StandardScaler()
    scaled_trainval_targets = scaler.fit_transform(targets[trainval_idx])
    scaled_test_targets = scaler.transform(targets[test_idx])

    scaler_path = OUTPUT_DIR / "StandardScaler.pkl"
    joblib.dump(scaler, scaler_path)

    trainval_images = images[trainval_idx]
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    all_true_targets = []
    all_predictions = []

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_images)):
        print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

        X_train_fold = trainval_images[train_idx]
        X_val_fold = trainval_images[val_idx]

        y_train_fold = scaled_trainval_targets[train_idx]
        y_val_fold = scaled_trainval_targets[val_idx]

        train_dataset = augmented_dataset(X_train_fold, y_train_fold, augmentation_fraction=AUGMENTATION_FRACTION)
        val_dataset = augmented_dataset(X_val_fold, y_val_fold, augmentation_fraction=AUGMENTATION_FRACTION)

        print("Number of images to use for training:", len(train_dataset))
        print("Number of images to use for validation:", len(val_dataset))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(MODEL_NAME).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_criterion = nn.MSELoss()

        training_loss_history = []
        validation_loss_history = []

        best_val_loss = float('inf')
        patience = PATIENCE
        epochs_no_improve = 0
        
        for epoch in range(EPOCHS):
            train_loss = train(train_loader, model, loss_criterion, optimizer, DEVICE)
            val_loss = validate(val_loader, model, loss_criterion, DEVICE)

            training_loss_history.append(train_loss)
            validation_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0

                best_model_state = model.state_dict()

                save_path = model_dir / f"best_model_fold{fold+1}.pt"
                torch.save(best_model_state, save_path)
                print(f"Saved best model for fold {fold+1} at epoch {epoch+1} to {save_path}")
                
            else:
                epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        avg_train = np.mean(training_loss_history)
        avg_val = np.mean(validation_loss_history)

        print(f"\nFold {fold+1} summary:")
        print(f"Average Train Loss: {avg_train:.4f}")
        print(f"Average Val Loss:   {avg_val:.4f}")

        model.eval()
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                val_preds.append(outputs.cpu().numpy())
                val_trues.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds).ravel()
        val_trues = np.concatenate(val_trues).ravel()

        all_predictions.append(val_preds)
        all_true_targets.append(val_trues)

        fold_results.append({
            'train_loss': training_loss_history,
            'val_loss': validation_loss_history,
            'avg_train_loss': avg_train,
            'avg_val_loss': avg_val
        })

    total_time = time.time() - start_time

    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Total training time: {total_time/3600:.2f} hours")

    print("\n=== K-Fold Summary ===")

    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: Avg Train Loss = {result['avg_train_loss']:.4f}, Avg Val Loss = {result['avg_val_loss']:.4f}")

    overall_train = np.mean([r['avg_train_loss'] for r in fold_results])
    overall_val = np.mean([r['avg_val_loss'] for r in fold_results])
    
    print(f"\nOverall Avg Train Loss: {overall_train:.4f}")
    print(f"Overall Avg Val Loss:   {overall_val:.4f}")

    all_predictions = np.concatenate(all_predictions)
    all_true_targets = np.concatenate(all_true_targets)

    
    # Find the best model among all folds based on validation loss.
    val_losses = [r['avg_val_loss'] for r in fold_results]
    best_fold_idx = int(np.argmin(val_losses))
    best_model_path = model_dir / f"best_model_fold{best_fold_idx + 1}.pt"
    print(f"Best model: Fold {best_fold_idx+1}, Val Loss: {val_losses[best_fold_idx]:.4f}")

    # Test set evaluation
    metrics, inference_time = evaluate_test_set(MODEL_NAME, best_model_path, images[test_idx], scaled_test_targets, scaler)

    print(
        f"Average inference time per sample: "
        f"{inference_time['mean_ms_per_sample']:.3f} ± "
        f"{inference_time['std_ms_per_sample']:.3f} ms"
    )

    print("\nTest-set metrics:")
    for label, values in metrics.items():
        print(
            f"{label:10s} | "
            f"RMSE={values['RMSE']:.6e} | "
            f"MAE={values['MAE']:.6e} | "
            f"R2={values['R2']:.6f}"
        )

    with open(OUTPUT_DIR / "cross_validation_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / "run_config.json", "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "k_folds": K_FOLDS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "patience": PATIENCE,
                "augmentation_fraction": AUGMENTATION_FRACTION,
                "augmentation_snr_min": AUGMENTATION_SNR_MIN,
                "augmentation_snr_max": AUGMENTATION_SNR_MAX,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
