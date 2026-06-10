"""
extract_features.py
從 GTZAN 資料集提取 MFCC 特徵，存成 features.csv

執行方式：
    python extract_features.py

輸入：Data/genres_original/<genre>/*.wav
輸出：features.csv
"""

import os
import csv
import librosa
import numpy as np

# ─── 設定路徑 ────────────────────────────────────────────────
DATA_DIR = os.path.join("Data", "genres_original")
OUTPUT_CSV = "features.csv"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ─── 特徵提取函式 ─────────────────────────────────────────────
def extract_features(file_path):
    """
    提取單一 .wav 檔案的特徵向量（共 59 維）：
      - MFCC mean × 20
      - MFCC std  × 20
      - Chroma mean × 12
      - Spectral Contrast mean × 7
    """
    try:
        y, sr = librosa.load(file_path, duration=30, mono=True)

        # MFCC：20 係數，取均值與標準差
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)   # (20,)
        mfcc_std  = np.std(mfcc, axis=1)    # (20,)

        # Chroma：音高分佈
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # (12,)

        # Spectral Contrast：頻譜對比度
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        sc_mean = np.mean(spec_contrast, axis=1)  # (7,)

        feature = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, sc_mean])
        return feature

    except Exception as e:
        print(f"  ⚠ 跳過 {file_path}：{e}")
        return None


# ─── 主程式 ──────────────────────────────────────────────────
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"❌ 找不到資料夾：{DATA_DIR}")
        print("   請先把 GTZAN 的 genres_original/ 放到 Data/ 底下")
        return

    feature_dim = 20 + 20 + 12 + 7  # = 59
    header = (
        [f"mfcc_mean_{i}" for i in range(20)] +
        [f"mfcc_std_{i}"  for i in range(20)] +
        [f"chroma_{i}"    for i in range(12)] +
        [f"sc_{i}"        for i in range(7)] +
        ["label"]
    )

    rows = []
    total = 0
    skipped = 0

    for genre in GENRES:
        genre_dir = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            print(f"  ⚠ 找不到類型資料夾：{genre_dir}")
            continue

        files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
        print(f"[{genre:10s}] 處理 {len(files)} 首歌 ...", end=" ")

        count = 0
        for fname in files:
            fpath = os.path.join(genre_dir, fname)
            feat = extract_features(fpath)
            if feat is not None:
                rows.append(list(feat) + [genre])
                count += 1
            else:
                skipped += 1
        print(f"✅ {count} 首")
        total += count

    # 寫入 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ 完成！共 {total} 筆，跳過 {skipped} 筆")
    print(f"   輸出：{OUTPUT_CSV}  (特徵維度：{feature_dim})")


if __name__ == "__main__":
    main()
