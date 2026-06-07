"""
ex3_lm_bigram.py — Character-level Bigram Language Model

使用 nn0.py 的 softmax、cross_entropy、Adam 與 gd() 訓練循環，
完整示範語言模型訓練流程。

架構：
  Embedding: vocab_size → embed_dim (用 linear 實作)
  Head:      embed_dim  → vocab_size (softmax → 機率)

資料：自定義短句子（可替換成任意文字）

訓練目標：給定當前字符，預測下一個字符（Bigram LM）

訓練後可生成新文字。
"""

import random
import math
from nn0 import Value, Adam, linear, softmax, cross_entropy, gd

random.seed(0)

# ────────────────────────────────────────────────────────────
# 1. 資料準備
# ────────────────────────────────────────────────────────────

text = "hello world how are you i am fine thank you very much"

chars = sorted(set(text))
vocab_size = len(chars)
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for c, i in ctoi.items()}

print(f"詞彙表大小: {vocab_size}")
print(f"字符集: {chars}\n")

# 建立 Bigram 資料集 (前一字 → 後一字)
data = [(ctoi[text[i]], ctoi[text[i+1]]) for i in range(len(text) - 1)]
print(f"訓練樣本數: {len(data)}")

# ────────────────────────────────────────────────────────────
# 2. 模型定義：Bigram Embedding LM
# ────────────────────────────────────────────────────────────

EMBED = 16   # Embedding 維度

def _rand_val():
    return Value(random.gauss(0, 0.02))   # 更小的初始化

# Embedding table: vocab_size × EMBED
E = [[_rand_val() for _ in range(EMBED)] for _ in range(vocab_size)]

# 投影頭：EMBED → vocab_size
W_head = [[_rand_val() for _ in range(EMBED)] for _ in range(vocab_size)]
b_head = [Value(0.0) for _ in range(vocab_size)]

# 所有參數
params = (
    [E[i][j]       for i in range(vocab_size) for j in range(EMBED)] +
    [W_head[i][j]  for i in range(vocab_size) for j in range(EMBED)] +
    b_head
)
print(f"參數總數: {len(params)}\n")


def model_fn(x_ids):
    """
    x_ids: [token_id]（bigram 只用前一個字）
    回傳: list[Value], shape (vocab_size,)，已 softmax
    """
    tok = x_ids[0]
    # Lookup embedding
    emb = E[tok]                            # list[Value], shape (EMBED,)
    # Head 層
    logits = linear(W_head, emb, b_head)   # list[Value], shape (vocab_size,)
    probs  = softmax(logits)
    return probs


# ────────────────────────────────────────────────────────────
# 3. 訓練（使用 nn0.gd()）
# ────────────────────────────────────────────────────────────

print("=== 開始訓練 Bigram 語言模型 ===")
losses = gd(
    model_fn   = model_fn,
    dataset    = [([src], tgt) for src, tgt in data],
    params     = params,
    num_steps  = 2000,
    lr         = 3e-3,    # 更保守的學習率
    verbose    = True,
    log_every  = 200,
)

print(f"\n最終平均 loss: {sum(losses[-100:]) / 100:.4f}")
print(f"（理論最差 loss = log({vocab_size}) ≈ {math.log(vocab_size):.4f}）\n")


# ────────────────────────────────────────────────────────────
# 4. 文字生成
# ────────────────────────────────────────────────────────────

def generate(start_char, max_len=40):
    """從 start_char 開始，用貪婪取樣生成文字。"""
    tok = ctoi.get(start_char, 0)
    result = [start_char]
    for _ in range(max_len):
        probs = model_fn([tok])
        # 機率取樣（multinomial）
        r = random.random()
        cumsum = 0.0
        for next_tok, p in enumerate(probs):
            cumsum += p.data
            if r <= cumsum:
                break
        result.append(itoc[next_tok])
        tok = next_tok
    return "".join(result)


print("=== 文字生成 ===")
for start in ['h', 'w', 'a', 'y']:
    sample = generate(start, max_len=30)
    print(f"  起始='{start}': {sample}")


# ────────────────────────────────────────────────────────────
# 5. 分析：顯示 Bigram 機率矩陣（部分）
# ────────────────────────────────────────────────────────────

print("\n=== 部分 Bigram 機率（'h' 之後最可能的字符）===")
probs_h = model_fn([ctoi['h']])
ranked = sorted(enumerate(probs_h), key=lambda x: -x[1].data)[:5]
for idx, p in ranked:
    print(f"  'h' → '{itoc[idx]}'  機率: {p.data:.4f}")
