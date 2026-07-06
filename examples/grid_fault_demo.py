from __future__ import annotations

import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_grid_dataset(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    transformer_age = rng.uniform(0, 30, n_samples)
    temperature_c = rng.normal(26, 8, n_samples).clip(-5, 45)
    humidity = rng.uniform(30, 95, n_samples)
    wind_speed = rng.gamma(shape=2.0, scale=1.8, size=n_samples).clip(0, 20)
    base_load = rng.normal(0.62, 0.16, n_samples).clip(0.18, 1.15)

    evening_peak = ((hour >= 18) & (hour <= 22)).astype(float)
    workday = (day_of_week < 5).astype(float)
    temperature_pressure = np.maximum(temperature_c - 30, 0) / 15
    load_rate = (base_load + 0.18 * evening_peak + 0.08 * workday + 0.1 * temperature_pressure).clip(0.05, 1.45)
    voltage_pu = (1.03 - 0.09 * load_rate + rng.normal(0, 0.015, n_samples)).clip(0.86, 1.08)
    reactive_power_ratio = (0.26 + 0.35 * load_rate + rng.normal(0, 0.06, n_samples)).clip(0.05, 0.95)
    power_factor = (1.0 - 0.22 * reactive_power_ratio + rng.normal(0, 0.02, n_samples)).clip(0.72, 1.0)
    recent_maintenance_days = rng.exponential(scale=120, size=n_samples).clip(0, 720)

    risk_score = (
        2.8 * (load_rate > 1.0)
        + 1.8 * (voltage_pu < 0.94)
        + 1.1 * (temperature_c > 34)
        + 0.9 * (transformer_age > 20)
        + 0.8 * (power_factor < 0.86)
        + 0.6 * (recent_maintenance_days > 365)
        + rng.normal(0, 0.7, n_samples)
        - 3.5
    )
    abnormal_probability = 1 / (1 + np.exp(-risk_score))
    is_abnormal = rng.binomial(1, abnormal_probability)

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_workday": workday,
            "transformer_age": transformer_age,
            "temperature_c": temperature_c,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "load_rate": load_rate,
            "voltage_pu": voltage_pu,
            "reactive_power_ratio": reactive_power_ratio,
            "power_factor": power_factor,
            "recent_maintenance_days": recent_maintenance_days,
            "is_abnormal": is_abnormal,
        }
    )


class FaultRiskMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_model() -> None:
    set_seed()
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

    device = choose_device()
    df = make_grid_dataset()
    feature_cols = [col for col in df.columns if col != "is_abnormal"]

    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols].to_numpy(dtype=np.float32),
        df["is_abnormal"].to_numpy(dtype=np.float32),
        test_size=0.2,
        random_state=42,
        stratify=df["is_abnormal"],
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = FaultRiskMLP(n_features=x_train.shape[1]).to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history: list[float] = []
    for epoch in range(1, 41):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(train_ds)
        history.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:02d} loss={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(x_test).to(device)
        logits = model(test_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()

    pred = (prob >= 0.5).astype(int)
    y_true = y_test.astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob),
    }

    print("\nEnvironment check")
    print(f"torch={torch.__version__}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print(f"device={device}")

    print("\nModel metrics")
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")

    print("\nConfusion matrix")
    print(confusion_matrix(y_true, pred))

    print("\nClassification report")
    print(classification_report(y_true, pred, target_names=["normal", "abnormal"]))

    sample = df.head(8).copy()
    sample_scaled = scaler.transform(sample[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    with torch.no_grad():
        sample_prob = torch.sigmoid(model(torch.from_numpy(sample_scaled).to(device))).cpu().numpy()
    sample["predicted_abnormal_prob"] = sample_prob.round(3)
    print("\nSample predictions")
    print(sample[["load_rate", "voltage_pu", "temperature_c", "is_abnormal", "predicted_abnormal_prob"]])

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(history) + 1), history, marker="o", linewidth=1.5)
    plt.title("Grid Fault Risk MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "metrics": metrics,
        },
        output_dir / "fault_risk_mlp.pt",
    )
    df.to_csv(output_dir / "synthetic_grid_fault_data.csv", index=False)

    print(f"\nSaved model and artifacts to: {output_dir}")


if __name__ == "__main__":
    train_model()
