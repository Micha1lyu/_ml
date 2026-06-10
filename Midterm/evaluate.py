"""
evaluate.py
評估訓練好的音樂類型分類模型

執行方式：
    python evaluate.py

輸入：model.pth、X_test.npy、y_test.npy
輸出：confusion_matrix.png、mfcc_visualization.png、classification_report（終端機）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import librosa
import librosa.display
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matplotlib.rcParams["font.family"] = "DejaVu Sans"  # 避免中文字型問題

# ─── 模型定義（需與 train.py 一致）────────────────────────────
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


# ─── 載入模型 ─────────────────────────────────────────────────
def load_model():
    if not os.path.exists("model.pth"):
        raise FileNotFoundError("❌ 找不到 model.pth，請先執行 train.py")

    ckpt = torch.load("model.pth", map_location="cpu")
    input_dim = ckpt["input_dim"]
    classes   = ckpt["label_encoder_classes"]

    model = MusicMLP(input_dim=input_dim, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes


# ─── 評估 ────────────────────────────────────────────────────
def evaluate():
    model, classes = load_model()

    if not os.path.exists("X_test.npy") or not os.path.exists("y_test.npy"):
        raise FileNotFoundError("❌ 找不到 X_test.npy / y_test.npy，請重新執行 train.py")

    X_test = np.load("X_test.npy").astype(np.float32)
    y_test = np.load("y_test.npy")

    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        y_pred = logits.argmax(1).numpy()

    acc = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix (Test Acc = {acc*100:.2f}%)", fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ confusion_matrix.png 已儲存")


# ─── MFCC 視覺化（用一首範例音檔）──────────────────────────────
def visualize_mfcc():
    """
    找第一首找得到的 .wav 畫 MFCC 熱圖，放進報告用
    """
    DATA_DIR = os.path.join("Data", "genres_original")
    wav_path = None

    genres = ["blues", "classical", "jazz", "rock", "pop"]
    for g in genres:
        d = os.path.join(DATA_DIR, g)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".wav")]
            if files:
                wav_path = os.path.join(d, files[0])
                genre_name = g
                break

    if wav_path is None:
        print("⚠ 找不到音檔，跳過 MFCC 視覺化")
        return

    y, sr = librosa.load(wav_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#4C9BE8")
    axes[0].set_title(f"Waveform — {genre_name} / {os.path.basename(wav_path)}", fontsize=12)
    axes[0].set_xlabel("Time (s)")

    # MFCC
    img = librosa.display.specshow(mfcc, x_axis="time", ax=axes[1], cmap="coolwarm")
    axes[1].set_title("MFCC (20 coefficients)", fontsize=12)
    axes[1].set_ylabel("MFCC Coefficient")
    fig.colorbar(img, ax=axes[1])

    plt.tight_layout()
    plt.savefig("mfcc_visualization.png", dpi=150)
    plt.close()
    print("✅ mfcc_visualization.png 已儲存")


# ─── 各類型 MFCC 對比 ─────────────────────────────────────────
def visualize_genre_comparison():
    """
    每個類型取一首歌，畫 MFCC 均值長條圖對比
    """
    DATA_DIR = os.path.join("Data", "genres_original")
    GENRES = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
    genre_mfcc = {}

    for g in GENRES:
        d = os.path.join(DATA_DIR, g)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".wav")]
        if not files:
            continue
        wav_path = os.path.join(d, files[0])
        try:
            y, sr = librosa.load(wav_path, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            genre_mfcc[g] = np.mean(mfcc, axis=1)
        except Exception:
            pass

    if not genre_mfcc:
        print("⚠ 無法載入音檔，跳過類型對比圖")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(13)
    for i, (g, mfcc_mean) in enumerate(genre_mfcc.items()):
        ax.plot(x, mfcc_mean, marker="o", label=g, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"MFCC {i+1}" for i in x], fontsize=9)
    ax.set_title("Mean MFCC Coefficients by Genre", fontsize=14)
    ax.set_ylabel("Mean Value")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("genre_mfcc_comparison.png", dpi=150)
    plt.close()
    print("✅ genre_mfcc_comparison.png 已儲存")


# ─── 主程式 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎵 音樂類型分類 — 評估報告\n")
    evaluate()
    print("\n🎨 產生視覺化圖表 ...\n")
    visualize_mfcc()
    visualize_genre_comparison()
    print("\n✅ 所有圖表已產生！可放入報告中使用。")
