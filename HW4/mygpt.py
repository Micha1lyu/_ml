"""
mygpt.py - 自己實作的迷你 GPT
參考架構：Karpathy microgpt (https://karpathy.github.io/2026/02/12/microgpt/)

完整包含：
  - 自動微分引擎 (Autograd)
  - Character-level Tokenizer
  - GPT-2 風格架構 (Multi-head Attention + MLP)
  - Adam 優化器
  - 訓練迴圈
  - 文字生成 (Inference)
"""

import os
import math
import random

random.seed(1337)

# ─────────────────────────────────────────
# 1. 資料集
# ─────────────────────────────────────────
# 自動下載 32,000 個英文名字（每個名字是一個 document）
if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    print("下載資料集...")
    urllib.request.urlretrieve(url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"文件數量 (num docs): {len(docs)}")

# ─────────────────────────────────────────
# 2. Tokenizer（字元級）
# ─────────────────────────────────────────
# 把資料集裡所有唯一字元收集起來，每個字元對應一個整數 ID
uchars = sorted(set(''.join(docs)))   # 26 個小寫英文字母
BOS = len(uchars)                     # 特殊 token：Beginning of Sequence (句子開始/結束標記)
vocab_size = len(uchars) + 1          # 總詞彙量 = 26 + 1(BOS) = 27
print(f"詞彙量 (vocab size): {vocab_size}")

# 輔助函式：字串 → token id 列表
def encode(s):
    return [uchars.index(c) for c in s]

# 輔助函式：token id 列表 → 字串
def decode(ids):
    return ''.join(uchars[i] for i in ids if i != BOS)

# ─────────────────────────────────────────
# 3. 自動微分引擎 (Autograd)
# ─────────────────────────────────────────
# 訓練神經網路需要對每個參數求梯度：「把這個數字調大一點，loss 會怎麼變？」
# 我們用計算圖 (computation graph) + 反向傳播 (backpropagation) 來做到這件事。
#
# Value 類別：把每個純量數字包裝起來，讓它能追蹤計算過程
class Value:
    # __slots__ 節省記憶體（因為我們會有非常多 Value 物件）
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data            # 這個節點的數值（正向傳播計算出來的）
        self.grad = 0               # 這個節點對 loss 的偏微分（反向傳播填入）
        self._children = children   # 計算圖中的子節點
        self._local_grads = local_grads  # 對子節點的局部偏微分

    # ── 基本運算定義（同時記錄 local gradient，供反向傳播使用）──

    def __add__(self, other):
        # z = x + y  →  dz/dx = 1, dz/dy = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # z = x * y  →  dz/dx = y, dz/dy = x
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # z = x^n  →  dz/dx = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # z = ln(x)  →  dz/dx = 1/x
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # z = e^x  →  dz/dx = e^x
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # z = max(0, x)  →  dz/dx = 1 if x>0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ── Python 運算符支援 ──
    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        反向傳播：從 loss 開始，用鏈式法則遞迴計算所有節點的梯度。
        步驟：
          1. 建立拓撲排序（確保父節點在子節點之前）
          2. 設定輸出節點 grad = 1（loss 對自己的導數）
          3. 倒序遍歷，把梯度乘以 local_grad 累積到子節點
        """
        # 建立拓撲排序
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 反向傳播
        self.grad = 1  # d(loss)/d(loss) = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # chain rule: d(loss)/d(child) += d(loss)/d(v) * d(v)/d(child)
                child.grad += local_grad * v.grad

# ─────────────────────────────────────────
# 4. 模型參數初始化
# ─────────────────────────────────────────
# GPT 超參數（這些數字決定模型大小）
n_layer    = 1      # Transformer 層數（越多越強，但越慢）
n_embd     = 16     # 嵌入維度（每個 token 用 16 維向量表示）
block_size = 16     # 最大上下文長度（最長的名字是 15 個字）
n_head     = 4      # 注意力頭數
head_dim   = n_embd // n_head  # 每個頭的維度 = 16 // 4 = 4

# 隨機初始化一個矩陣（nout × nin），用高斯分布
def make_matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict 存放所有模型參數
# wte: token embedding table（每個 token → 16 維向量）
# wpe: position embedding table（每個位置 → 16 維向量）
# lm_head: 最後的線性層，把 16 維 → vocab_size 個 logit
state_dict = {
    'wte':     make_matrix(vocab_size, n_embd),
    'wpe':     make_matrix(block_size, n_embd),
    'lm_head': make_matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    # Attention 的 Q, K, V, O 投影矩陣
    state_dict[f'layer{i}.attn_wq'] = make_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = make_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = make_matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = make_matrix(n_embd, n_embd)
    # MLP 的兩層線性變換（中間維度擴大 4 倍）
    state_dict[f'layer{i}.mlp_fc1'] = make_matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = make_matrix(n_embd, 4 * n_embd)

# 把所有參數攤平成一個 list，方便優化器統一更新
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"參數總數 (num params): {len(params)}")

# ─────────────────────────────────────────
# 5. 模型架構（前向傳播）
# ─────────────────────────────────────────

def linear(x, W):
    """線性變換：y = W @ x
    x: list[Value]，長度 nin
    W: list[list[Value]]，形狀 nout × nin
    回傳: list[Value]，長度 nout
    """
    return [sum(wij * xi for wij, xi in zip(row, x)) for row in W]

def softmax(logits):
    """Softmax：把 logits 轉成機率分布
    減掉 max 避免 overflow（數學上等價，但數值穩定）
    """
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMS Normalization：對向量做正規化
    比 LayerNorm 簡單，不需要減均值
    scale = 1 / sqrt(mean(x^2) + eps)
    """
    ms = sum(xi * xi for xi in x) / len(x)      # mean of squares
    scale = (ms + 1e-5) ** -0.5                  # 1 / RMS
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    """GPT 前向傳播（一次處理一個 token）
    
    token_id: 當前 token 的 id
    pos_id:   當前位置（0-indexed）
    keys:     KV cache 中的 Key（每層一個 list）
    values:   KV cache 中的 Value（每層一個 list）
    
    回傳: logits（vocab_size 個數值，未正規化的下一個 token 機率）
    """
    # ── Embedding 層 ──
    # 從 embedding table 查出 token 和 position 的向量，相加
    tok_emb = state_dict['wte'][token_id]   # shape: [n_embd]
    pos_emb = state_dict['wpe'][pos_id]     # shape: [n_embd]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    # 先做一次 RMSNorm（這在 residual stream 開始時很重要）
    x = rmsnorm(x)

    # ── Transformer Layers ──
    for li in range(n_layer):

        # ── 1. Multi-Head Self-Attention ──
        x_residual = x          # 先存住，待會做殘差連接
        x = rmsnorm(x)

        # 計算 Query, Key, Value（對當前 token）
        q = linear(x, state_dict[f'layer{li}.attn_wq'])   # [n_embd]
        k = linear(x, state_dict[f'layer{li}.attn_wk'])   # [n_embd]
        v = linear(x, state_dict[f'layer{li}.attn_wv'])   # [n_embd]

        # 把 K, V 加到 KV cache（這樣可以看到前面所有 token）
        keys[li].append(k)
        values[li].append(v)

        # 對每個 attention head 分開計算
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim   # 這個 head 在向量中的起始位置

            # 切出這個 head 的 q, k, v
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]   # 所有過去位置的 k
            v_h = [vi[hs : hs + head_dim] for vi in values[li]] # 所有過去位置的 v

            # Attention score = Q · K^T / sqrt(head_dim)
            # （除以 sqrt(head_dim) 讓梯度不要太大或太小）
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # Softmax → attention weights（代表「應該注意哪些過去的 token」）
            attn_weights = softmax(attn_logits)

            # 加權求和 Value → 這個 head 的輸出
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        # 把多頭輸出合併後做線性投影
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        # 殘差連接：讓梯度可以直接流回去（解決深層網路的梯度消失問題）
        x = [a + b for a, b in zip(x, x_residual)]

        # ── 2. MLP 前饋網路 ──
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 升維：16 → 64
        x = [xi.relu() for xi in x]                       # 非線性激活
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 降維：64 → 16
        x = [a + b for a, b in zip(x, x_residual)]       # 殘差連接

    # ── 輸出層 ──
    # 把最終向量投影到 vocab_size 維，得到每個 token 的 logit
    logits = linear(x, state_dict['lm_head'])
    return logits

# ─────────────────────────────────────────
# 6. Adam 優化器
# ─────────────────────────────────────────
# Adam 是 SGD 的改進版，維護每個參數的一階矩（momentum）和二階矩（RMS）
# 讓每個參數的學習率自適應調整
learning_rate = 0.01
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8

m_buf = [0.0] * len(params)   # 一階矩（梯度的指數移動平均）
v_buf = [0.0] * len(params)   # 二階矩（梯度平方的指數移動平均）

# ─────────────────────────────────────────
# 7. 訓練迴圈
# ─────────────────────────────────────────
num_steps = 1000  # 訓練步數（每步處理一個名字）

print(f"\n開始訓練 ({num_steps} 步)...")
for step in range(num_steps):

    # 取一個文件（名字），循環使用
    doc = docs[step % len(docs)]

    # Tokenize：字元 → id，頭尾包 BOS
    # 例如 "emma" → [BOS, 4, 12, 12, 0, BOS]
    tokens = [BOS] + encode(doc) + [BOS]
    n = min(block_size, len(tokens) - 1)  # 實際要訓練的長度

    # ── 前向傳播 ──
    # 每次從 pos=0 開始，用 KV cache 逐步建立
    kv_keys   = [[] for _ in range(n_layer)]
    kv_values = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id = tokens[pos_id]       # 當前 token
        target_id = tokens[pos_id + 1]  # 目標（下一個 token）

        logits = gpt(token_id, pos_id, kv_keys, kv_values)
        probs = softmax(logits)

        # Cross-entropy loss = -log(正確 token 的機率)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 平均 loss（對整個名字序列）
    loss = (1 / n) * sum(losses)

    # ── 反向傳播 ──
    loss.backward()

    # ── Adam 參數更新 ──
    lr_t = learning_rate * (1 - step / num_steps)  # 線性 LR decay
    for i, p in enumerate(params):
        g = p.grad
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * g
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * g**2
        m_hat = m_buf[i] / (1 - beta1**(step + 1))  # bias correction
        v_hat = v_buf[i] / (1 - beta2**(step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0   # 清空梯度，準備下一步

    print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}", end='\r')

print()  # 換行

# ─────────────────────────────────────────
# 8. 生成（Inference）
# ─────────────────────────────────────────
# 訓練完後，用模型生成新名字
# 從 BOS 開始，每次預測下一個 token，直到再次出現 BOS（代表名字結束）
temperature = 0.5  # 溫度：越低越保守，越高越有創意

print("\n─── 生成的名字 ───")
for sample_idx in range(20):
    kv_keys   = [[] for _ in range(n_layer)]
    kv_values = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, kv_keys, kv_values)
        # 溫度縮放：logits / temperature，再 softmax
        probs = softmax([l / temperature for l in logits])
        # 按機率分布隨機抽樣下一個 token
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break  # 遇到 BOS 代表名字結束
        sample.append(uchars[token_id])

    print(f"  sample {sample_idx+1:2d}: {''.join(sample)}")
