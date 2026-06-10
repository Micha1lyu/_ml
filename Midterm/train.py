"""
train.py
用 PyTorch MLP 訓練音樂類型分類模型

執行方式：
    python train.py

輸入：features.csv（由 extract_features.py 產生）
輸出：model.pth、training_curve.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非互動式，不開視窗
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── 設定 ─────────────────────────────────────────────────────
CSV_FILE   = "features.csv"
MODEL_FILE = "model.pth"
CURVE_FILE = "training_curve.png"

EPOCHS     = 200
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42
PATIENCE   = 20  # early stopping patience

torch.manual_seed(SEED)
np.random.seed(SEED)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ─── 資料載入 ─────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(CSV_FILE)

    # 去掉 label 欄，保留所有數值特徵
    drop_cols = ["label"]
    X = df.drop(columns=drop_cols).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    le.fit(GENRES)
    y = le.transform(y_raw).astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y, le, scaler


# ─── 模型架構 ─────────────────────────────────────────────────
class MusicMLP(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── 訓練 ────────────────────────────────────────────────────
def train():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到 {CSV_FILE}")
        return

    print("📂 載入資料 ...")
    X, y, le, scaler = load_data()
    print(f"   資料量：{len(X)} 筆，特徵維度：{X.shape[1]}，類別：{len(le.classes_)}")

    # Train/Val/Test split：70 / 15 / 15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    print(f"   Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    def make_loader(Xd, yd, shuffle=True):
        ds = TensorDataset(torch.tensor(Xd), torch.tensor(yd))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    # 儲存 test set 供 evaluate.py 使用
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用裝置：{device}")

    model = MusicMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 開始訓練（最多 {EPOCHS} epochs，patience={PATIENCE}）...\n")

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        t_loss, t_correct = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(Xb)
            t_correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        # --- Val ---
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                v_loss += criterion(logits, yb).item() * len(Xb)
                v_correct += (logits.argmax(1) == yb).sum().item()

        t_loss /= len(X_train); t_acc = t_correct / len(X_train)
        v_loss /= len(X_val);   v_acc = v_correct / len(X_val)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": X.shape[1],
                "label_encoder_classes": le.classes_.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }, MODEL_FILE)
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            star = "⭐" if v_acc == best_val_acc else ""
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss={t_loss:.4f} Acc={t_acc:.3f} | "
                  f"Val Loss={v_loss:.4f} Acc={v_acc:.3f} {star}")

        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping 觸發（{PATIENCE} epochs 無改善），停在 epoch {epoch}")
            break

    print(f"\n✅ 訓練完成！最佳 Val Acc：{best_val_acc:.4f}")
    print(f"   模型儲存至：{MODEL_FILE}")

    # ── 畫訓練曲線 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Music Genre Classification — Training History", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train Loss", color="#4C9BE8")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#E84C4C")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Loss Curve")

    axes[1].plot(history["train_acc"], label="Train Acc", color="#4C9BE8")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#E84C4C")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(CURVE_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   訓練曲線儲存至：{CURVE_FILE}")


if __name__ == "__main__":
    train()
