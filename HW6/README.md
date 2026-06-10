# HW6 ── 非 Transformer 語言模型

> 用 **Variable-order Markov Model + Kneser-Ney Smoothing** 實作字元級語言模型，不使用 Attention / Transformer 架構。

## 方法說明

### 為什麼不用 sklearn 分類器？

常見做法是把「前 N 個字」轉成特徵向量，再用 Logistic Regression 預測下一個字。  
這種方式本質上是把語言模型轉成**分類問題**，缺點是：context 沒見過時完全無法處理。

本作業改用**統計語言模型**的標準做法：直接估計條件機率，並用 **Kneser-Ney Smoothing** 解決稀疏問題。

---

### 核心公式

**Kneser-Ney Smoothing（插值版）**：

$$
P_{KN}(w \mid \text{ctx}) = \frac{\max(c(\text{ctx}, w) - d,\ 0)}{c(\text{ctx})} + \lambda(\text{ctx}) \cdot P_{KN}(w \mid \text{ctx}_{1:})
$$

| 符號 | 意義 |
|------|------|
| $c(\text{ctx}, w)$ | (context, w) 的出現次數 |
| $d$ | 折扣值（預設 0.75） |
| $\lambda(\text{ctx})$ | backoff weight，補足折扣掉的機率質量 |
| $P_{KN}(w \mid \text{ctx}_{1:})$ | 遞迴 backoff 到更短的 context |

**Unigram Fallback（continuation probability）**：

$$
P_{KN}(w \mid \emptyset) \propto \bigl|\{v : c(v, w) > 0\}\bigr|
$$

即「$w$ 出現在幾種不同前一個字後面」，比純頻率更能代表詞的普遍性。

---

### 抽樣策略

```
temperature > 1  → 機率分布更平坦，輸出更多樣
temperature < 1  → 機率分布更尖銳，接近 greedy
top_k > 0        → 只從機率最高的 k 個字中抽樣
```

---

## 檔案結構

```
HW6/
├── lm.py    # 語言模型主程式
└── tw.txt   # 訓練語料（繁中短句）
```

## 執行方式

```bash
python lm.py tw.txt
```

程式會：
1. 訓練 Trigram Kneser-Ney 語言模型
2. 印出訓練集 Perplexity
3. 跑幾個固定 prompt 的生成示範
4. 進入互動模式，可自行輸入 prompt 生成文字

### 互動模式範例

```
prompt：小貓
→ 小貓坐在桌上

prompt：天上
→ 天上有白雲

prompt：今天
→ 今天天氣好
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `ORDER` | 3 | n-gram 階數（3 = trigram） |
| `DISCOUNT` | 0.75 | Kneser-Ney 折扣 d |
| `MAX_GEN` | 30 | 最大生成長度 |
| `TEMPERATURE` | 1.0 | 抽樣溫度 |
| `TOP_K` | 0 | top-k 抽樣（0 = 不限） |

## 與 Transformer 的差異

| 面向 | 本作業（Markov + KN） | Transformer |
|------|----------------------|-------------|
| 記憶長度 | 固定 n-1 個字 | 理論上無限（靠 attention） |
| 訓練資料需求 | 極少（幾十句即可） | 需要大量資料 |
| 推論速度 | 極快（查表） | 較慢（矩陣乘法） |
| 泛化能力 | 靠 smoothing | 靠模型權重 |
| 可解釋性 | 高（直接看機率表） | 低 |
