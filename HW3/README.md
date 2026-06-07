# HW3 — nn0.py 深度學習框架學習範例

本作業基於 [nn0.py](https://github.com/ccc-c/c0computer/blob/main/ai/nn0/nn0.py) 這個輕量級自動微分框架，
由淺到深實作三個機器學習範例，涵蓋自動微分、MLP 分類、語言模型三大核心概念。

---

## 檔案結構

```
HW3/
├── nn0.py              # 核心框架（自動微分引擎 + Adam + 神經網路工具函數）
├── ex1_autograd.py     # 範例一：自動微分驗證
├── ex2_mlp.py          # 範例二：MLP 學習 XOR
├── ex3_lm_bigram.py    # 範例三：Bigram 字符語言模型
└── README.md           # 本文件
```

---

## nn0.py — 核心框架

這是整個作業的基礎，不需要安裝任何第三方套件（只用 Python 標準庫的 `math` 與 `random`）。

### 提供的元件

| 元件 | 類型 | 說明 |
|------|------|------|
| `Value` | class | 自動微分節點，支援 `+`, `*`, `**`, `log`, `exp`, `relu`, `tanh`, `sigmoid` |
| `Adam` | class | Adam 優化器，含一階/二階動量與偏差修正 |
| `linear(W, x, b)` | function | 全連接層矩陣乘法 `W·x + b` |
| `softmax(logits)` | function | 數值穩定版 softmax（先減 max 防溢位） |
| `rmsnorm(x, g)` | function | RMS Layer Normalization（Llama 風格，無 bias） |
| `cross_entropy(probs, target_id)` | function | 負對數似然損失 `-log(p[target])` |
| `gd(...)` | function | 通用訓練循環，含 Cosine lr decay + 梯度裁剪 |

### Value 的運算圖原理

```
a = Value(2.0)
b = Value(3.0)
c = a * b + 1   # 自動記錄：c 的父節點是 a, b，局部梯度是 b.data, a.data
c.backward()    # 拓撲排序 → 反向鏈式法則
print(a.grad)   # dc/da = b = 3.0
```

每個 `Value` 儲存：
- `data`：當前數值
- `grad`：反向傳播後累積的梯度
- `_children`：產生此節點的父節點
- `_local_grads`：對應父節點的局部導數

---

## ex1_autograd.py — 範例一：自動微分驗證

**學習目標**：理解運算圖與梯度是如何計算的，並用手算公式驗證。

### 涵蓋內容

1. **純量運算梯度** — `f(x,y) = (x*y+1)²`，驗證 `df/dx = 42`, `df/dy = 28`
2. **鏈式法則** — `f(x) = log(relu(x)+1)`，驗證導數為 `1/3`
3. **單一神經元** — `tanh(w1*x1 + w2*x2 + b)` 的各參數梯度

### 執行

```powershell
python ex1_autograd.py
```

### 預期輸出（節錄）

```
df/dx = 42.0000  (手算: 42)
df/dy = 28.0000  (手算: 28)
[OK] 梯度驗證通過！
```

---

## ex2_mlp.py — 範例二：MLP 學習 XOR

**學習目標**：用 `Value` 手刻多層感知機（MLP），展示神經網路如何學習非線性函數。

### 問題設定

XOR 是線性不可分的問題，單一神經元無法解決：

| x1 | x2 | 目標 |
|----|----|------|
| 0  | 0  | 0    |
| 0  | 1  | 1    |
| 1  | 0  | 1    |
| 1  | 1  | 0    |

### 網路架構

```
輸入 (2) → 隱藏層 (4, tanh) → 輸出 (1, sigmoid)
```

- 損失函數：MSE（均方誤差）
- 優化器：Adam (`lr=0.05`)
- 參數初始化：Xavier Uniform
- 訓練步數：2000 步（每步對全部 4 筆資料算 batch loss）

### 執行

```powershell
python ex2_mlp.py
```

### 預期輸出

```
Step 2000  MSE=0.000016  lr=0.00050

=== 測試結果 ===
(0, 0)   0   0.0039   [OK]
(0, 1)   1   0.9969   [OK]
(1, 0)   1   0.9952   [OK]
(1, 1)   0   0.0040   [OK]

準確率: 4/4 (100%)
```

---

## ex3_lm_bigram.py — 範例三：Bigram 字符語言模型

**學習目標**：完整走過語言模型的訓練流程，包含 Embedding、softmax 機率預測、交叉熵損失、Adam 優化，以及文字生成。

### 模型架構

```
輸入字符 token_id
    ↓
Embedding table E[token_id]   (vocab_size × embed_dim)
    ↓
linear 投影頭 W_head           (embed_dim → vocab_size)
    ↓
softmax → 機率分佈
    ↓
cross_entropy loss → backward → Adam update
```

### 訓練設定

| 項目 | 值 |
|------|----|
| 訓練語料 | `"hello world how are you..."` |
| 詞彙大小 | 19 個字符 |
| Embedding 維度 | 16 |
| 訓練步數 | 2000 |
| 學習率策略 | Cosine Decay |
| 梯度裁剪 | Global Norm Clipping (閾值 5.0) |

### 執行

```powershell
python ex3_lm_bigram.py
```

### 預期輸出（節錄）

```
step  200/2000  loss=2.87
step  600/2000  loss=2.11   ← 接近最佳

'h' 之後最可能的字符：
  'h' → 'e'  機率: 0.637   (正確！hello, how)
  'h' → 'o'  機率: 0.207   (正確！how)
```

訓練後可用機率取樣生成新文字。

---

## 執行方式

> Windows 中文環境需要設定 UTF-8，否則會有 `UnicodeEncodeError`：

```powershell
# 設定一次即可（PowerShell）
$env:PYTHONUTF8=1

# 執行任一範例
python ex1_autograd.py
python ex2_mlp.py
python ex3_lm_bigram.py
```

不需要安裝任何套件，純 Python 標準庫即可執行。

---

## 三個範例的核心對應關係

```
nn0.py 的功能              對應的範例
─────────────────────────────────────────
Value.backward()      →   ex1（梯度驗證）
Adam.step()           →   ex2（XOR 訓練）
softmax + cross_entropy → ex3（語言模型）
gd() 訓練循環         →   ex3（end-to-end）
linear()              →   ex3（投影頭）
```

---

## 參考資料

- 原始 nn0.py：https://github.com/ccc-c/c0computer/blob/main/ai/nn0/nn0.py
- 分支範例：https://github.com/ccc-c/c0computer/tree/testCompilerAi1/ai/nn0
- Micrograd（啟發來源）：https://github.com/karpathy/micrograd
- Adam 論文：Kingma & Ba, 2014, https://arxiv.org/abs/1412.6980
- RMSNorm：Zhang & Sennrich, 2019, https://arxiv.org/abs/1910.07467
