# AI 協助對話紀錄

> 工具：Antigravity（Claude Sonnet 4.6）  
> 日期：2026-06-10  
> 專案：音樂類型分類（GTZAN Dataset）

---

## 對話內容

---

**我：**  
我要做一個機器學習的期中作業，資料集用這個：  
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification  
要分類 10 種音樂類型，幫我做完整的 pipeline

**AI：**  
收到。GTZAN 總共 1000 首歌（每類 100 首），每首 30 秒的 .wav 檔，10 個類別：blues、classical、country、disco、hiphop、jazz、metal、pop、reggae、rock。

你想走哪條路？  
1. 直接用 Kaggle 提供的 features CSV（已預先抽好 MFCC 等特徵）  
2. 自己從 .wav 抽特徵再訓練

---

**我：**  
兩個都寫，pipeline 要完整，從 wav 到訓練到評估

**AI：**  
好，規劃如下：

- `extract_features.py`：讀 .wav → 抽 MFCC、Chroma、Spectral Contrast → 存成 `features.csv`  
- `train.py`：讀 CSV → StandardScaler → MLP 訓練 → 存 `model.pth`  
- `evaluate.py`：載入模型 → 預測 test set → confusion matrix + 視覺化圖表

特徵維度我建議用：MFCC mean×20 + std×20 + Chroma×12 + Spectral Contrast×7 = 59 維，資訊量夠又不會太高維。

---

**我：**  
MFCC 是什麼？為什麼要用它？

**AI：**  
MFCC（Mel-Frequency Cepstral Coefficients，梅爾頻率倒譜係數）是模擬人耳感知設計的音訊特徵，流程是：

1. 把音訊切成短幀，對每幀做 FFT 轉到頻域  
2. 套梅爾濾波器組，把線性頻率壓縮成對數刻度（符合人耳對低頻更敏感的特性）  
3. 對梅爾頻譜取 log，再做 DCT，去掉頻帶間相關性  
4. 取前 N 個係數（視覺化常用 13，訓練常用 20）

用 MFCC 的理由是：它對音色（timbre）描述能力強，不同樂器、不同 genre 的音色差異會反映在 MFCC 的分佈上，比直接用原始波形有效得多。

---

**我：**  
模型用什麼？

**AI：**  
用 MLP（多層感知機）就夠了，特徵是固定維度的向量，不需要 CNN。架構：

```
Linear(59 → 256) → BatchNorm → ReLU → Dropout(0.3)
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
Linear(128 → 64) → ReLU
Linear(64 → 10)
```

BatchNorm 幫助收斂，Dropout 防止 overfitting，優化器用 Adam + StepLR（每 50 epoch 學習率乘 0.5）。

---

**我：**  
好那就這樣，幫我寫三支腳本

**AI：**  
（生成 `extract_features.py`、`train.py`、`evaluate.py` 完整程式碼）

---

**我：**  
訓練設定用什麼？epoch、batch size、learning rate 怎麼選的？

**AI：**  
設定如下：

| 參數 | 值 | 理由 |
|------|----|------|
| Epochs | 最多 200 | 59 維特徵、1000 筆資料，通常提前收斂 |
| Batch size | 64 | 常見預設，在這個資料量下不會太大也不會太小 |
| Learning rate | 1e-3 | Adam 的標準起始值 |
| Weight decay | 1e-4 | L2 正則化，額外抑制 overfitting |
| Scheduler | StepLR，step=50，gamma=0.5 | 每 50 epoch 學習率減半，讓後期訓練更精細收斂 |
| Early Stopping | patience=20 | val acc 連續 20 epoch 無改善則停止，防止過擬合 |

---

**我：**  
為什麼用 Adam 不用 SGD？

**AI：**  
Adam 是 SGD + 動量 + 自適應學習率的組合，對每個參數獨立調整更新幅度，收斂速度比普通 SGD 快很多，對超參數也不那麼敏感。

SGD 在大型視覺任務（ImageNet 等）搭配精心調整的 schedule 可以達到更好的泛化，但對這種小資料集的 MLP，Adam 是更務實的選擇——同樣的 epoch 數能到更高的準確率，而且不需要花時間調 momentum、LR schedule。

---

**我：**  
Train/Val/Test 怎麼切的？

**AI：**  
70 / 15 / 15 的比例：

```
train_test_split(X, y, test_size=0.30)   → 70% train, 30% tmp
train_test_split(tmp, test_size=0.50)    → 15% val, 15% test
```

切分時有加 `stratify=y`，確保每個類別在三個 split 裡的比例一致（各 genre 都有 100 首，不 stratify 也不會差太多，但加上去是好習慣）。

Val set 用來監控 overfitting 並儲存 best checkpoint，Test set 只在最後評估用，訓練過程完全不碰。

---

**我：**  
跑起來了，test accuracy 大概 72%，有辦法更好嗎？

**AI：**  
72% 對 MLP + 手工特徵來說算正常，以下是幾個方向：

1. **增加特徵**：加 Zero Crossing Rate、Spectral Centroid、Tempo 等，特徵維度從 59 拉到 80+  
2. **資料增強**：對音訊加噪聲、pitch shift，讓 training set 更多樣  
3. **調參**：把 Dropout 從 0.3 調到 0.4，或 epoch 從 200 增到 300  
4. **換模型**：把音訊轉成梅爾頻譜圖（image），改用 CNN，準確率可以到 85%+

短時間內最快有效的是方向 1 或方向 3。

---

**我：**  
好那先不動，幫我生成 confusion matrix 圖跟 MFCC 視覺化

**AI：**  
已在 `evaluate.py` 加入三個視覺化函式：

- `evaluate()`：輸出 confusion matrix（seaborn heatmap）  
- `visualize_mfcc()`：取一首 wav，畫波形圖 + MFCC 熱圖  
- `visualize_genre_comparison()`：每個 genre 取一首，畫 MFCC 均值折線圖對比

---

**我：**  
FFT 是什麼？跟 MFCC 有什麼關係？

**AI：**  
FFT（Fast Fourier Transform，快速傅立葉變換）是把時域訊號轉換到頻域的演算法。

音訊是一串隨時間變化的振幅值（時域），你從波形上看不出這段聲音裡有多少 200 Hz 成分、多少 2000 Hz 成分。FFT 就是做這件事：把時域波形分解成各個頻率的強度。

跟 MFCC 的關係：MFCC 的第一步就是對每個短幀做 FFT，得到功率頻譜，後面的梅爾濾波器才有東西可以套。

---

**我：**  
為什麼要切成短幀？直接對整首歌做 FFT 不行嗎？

**AI：**  
音樂的頻率內容會隨時間變化——歌的開頭可能是鼓聲，中間是人聲，結尾是吉他。如果對整首歌做 FFT，這些資訊就全部混在一起，你只會得到一個「平均頻譜」，根本分不出音色的變化。

切成短幀（通常 20–40 ms）的原因是：在這麼短的時間內，音訊可以視為「準穩態」，頻率特性是穩定的。對每幀各做一次 FFT，再統計所有幀的特徵（取 mean / std），才能捕捉到整首歌的音色資訊。

---

**我：**  
梅爾濾波器組是什麼？為什麼要用它？

**AI：**  
梅爾濾波器組是一組帶通濾波器，排列在梅爾刻度（Mel scale）上。

梅爾刻度是模擬人耳感知的頻率刻度：人耳對低頻的分辨能力遠比高頻強，100 Hz 跟 200 Hz 聽起來差很多，但 8000 Hz 跟 8100 Hz 幾乎聽不出差異。線性頻率刻度不符合這個特性，所以梅爾刻度把低頻拉開、把高頻壓縮。

用梅爾濾波器組的好處：  
1. 把頻譜從幾千個頻率 bin 壓縮成 40 個梅爾頻帶，大幅降維  
2. 壓縮後的表示更接近人耳的感知，對音樂、語音任務效果更好

---

**我：**  
DCT 是什麼？為什麼做完梅爾頻譜還要再做 DCT？

**AI：**  
DCT（Discrete Cosine Transform，離散餘弦變換）的作用是去相關性。

梅爾濾波器組的相鄰頻帶有重疊，所以得出的 40 個梅爾頻帶值之間是有相關性的（相鄰的值通常相近）。DCT 把這 40 個相關的值轉換到一個新的空間，讓各個分量盡量彼此獨立。

結果是：DCT 的低頻分量（前幾個係數）能用少量數字代表頻譜的整體形狀，高頻分量則代表細節。取前 13 或 20 個係數，就是只保留最重要的「頻譜輪廓」，捨棄細節雜訊。

這也是為什麼 MFCC 叫「倒譜」（Cepstrum）——對頻譜再做一次類傅立葉變換，得到「頻率的頻率」。

---

**我：**  
所以 MFCC 為什麼取 20 個不取 13 個？

**AI：**  
13 是語音辨識的傳統慣例，因為人聲的頻率特性主要集中在低倒譜係數。

音樂分類不一樣，樂器的音色比人聲複雜，更高階的係數（13–20）仍然包含有用的音色資訊。論文和實驗普遍顯示，對音樂任務取 20–40 個係數比 13 個效果更好。

視覺化圖表用 13 是因為這樣比較好解釋、也是教科書常見的數字；實際訓練用 20，準確率更高。

---

**我：**  
低序的 MFCC 跟高序的差在哪？

**AI：**  
低序係數（MFCC 1–4）代表頻譜的整體形狀，也就是聲音的「粗輪廓」：

- MFCC 1：近似整體的音量（能量）  
- MFCC 2–4：頻譜的大致斜率、低頻/高頻的能量比例

這幾個係數對區分像 classical（頻譜平坦、高頻豐富）vs. hiphop（低頻能量集中）非常有效。

高序係數（MFCC 10–20）代表頻譜的細節結構：

- 精細的諧波比例、音色微變化  
- 對區分音色相近的 genre（例如 rock vs. metal）有幫助，但也更容易受雜訊影響

實際上，MFCC 1 通常會被捨棄（因為它只是能量，跟音量有關而不是音色），語音辨識有時只用 2–13，音樂分類則通常全取 1–20。

---

**我：**  
README 裡有些資訊跟程式碼不一樣，幫我修

**AI：**  
找到以下不一致：

| 位置 | 問題 | 修正 |
|------|------|------|
| README 標題 | 寫「期末作業」 | 改「期中作業」 |
| 目錄結構 | `final/` + `data/` | 改 `Midterm/` + `Data/` |
| MFCC 說明 | 「取前 13 個係數」 | 改「視覺化 13，訓練 20」 |

（套用修正）

---

**我：**  
幫我在 README 加上使用 Antigravity 的聲明

**AI：**  
已在 README 末尾加入：

> 本專案程式碼與文件撰寫過程中，使用 **Antigravity** 搭配 **Claude Sonnet 4.6**。
