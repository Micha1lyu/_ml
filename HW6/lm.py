"""
lm.py  ── Variable-order Markov Language Model with Kneser-Ney Smoothing
=========================================================================

核心想法：
  不用 sklearn 分類器，直接估計條件機率：
    P(w | w_{n-2}, w_{n-1})

  但單純計數有稀疏問題（某些 context 沒出現過）。
  解決方法：Kneser-Ney Smoothing + Backoff：
    P_KN(w | u) = max(c(u,w) - d, 0) / c(u)
                + λ(u) * P_KN(w | shorter_context)

  其中 d = discount（固定折扣），λ 為 backoff weight。

  終極 fallback：unigram continuation probability
    P_continuation(w) ∝ 「w 出現在幾種不同右側 context」

執行：
    python lm.py tw.txt
"""

import sys, random, math
from collections import defaultdict

# ── 參數 ─────────────────────────────────────────────────────
CORPUS_FILE = "tw.txt"
ORDER       = 3        # 最高階數（trigram = 3）
DISCOUNT    = 0.75     # Kneser-Ney 折扣 d
MAX_GEN     = 30       # 生成最大長度
TEMPERATURE = 1.0      # 抽樣溫度
TOP_K       = 0        # top-k（0=不限）
SEED        = 42
# ─────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════════

def load_corpus(path: str) -> list[str]:
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens += list(line) + ["<EOS>"]
    return tokens


def sample(dist: dict[str, float], temperature=1.0, top_k=0) -> str:
    """從機率字典中做 temperature + top-k 抽樣。"""
    items = list(dist.items())
    # temperature scaling
    scores = {w: math.exp(math.log(p + 1e-30) / temperature)
              for w, p in items}
    # top-k
    if top_k > 0:
        items_sorted = sorted(scores, key=scores.get, reverse=True)[:top_k]
        scores = {w: scores[w] for w in items_sorted}
    total = sum(scores.values())
    r = random.random() * total
    cumul = 0.0
    for w, s in scores.items():
        cumul += s
        if r <= cumul:
            return w
    return list(scores)[-1]


# ════════════════════════════════════════════════════════════
#  統計計數
# ════════════════════════════════════════════════════════════

class NgramStats:
    """儲存所有階層的 n-gram 計數，以及 Kneser-Ney 所需的 continuation 計數。"""

    def __init__(self, order: int):
        self.order = order
        # count[(ctx_tuple, w)] = 出現次數（最高階 raw count）
        self.count: dict[tuple, int] = defaultdict(int)
        # ctx_count[ctx_tuple] = 該 context 總出現次數
        self.ctx_count: dict[tuple, int] = defaultdict(int)
        # continuation[(w,)] = w 出現在幾種不同的 (history, w) 組合（unigram KN）
        self.continuation: dict[str, int] = defaultdict(int)
        # 記錄 (ctx, w) 有出現的集合，用來算 lambda（有幾個 w 使 c(ctx,w)>0）
        self.ctx_vocab: dict[tuple, set] = defaultdict(set)

    def fit(self, tokens: list[str]):
        seen_bigrams: set[tuple] = set()

        for i in range(len(tokens) - 1):
            w_next = tokens[i + 1]
            # continuation：用 bigram 估算（w 接在哪些字後面）
            pair = (tokens[i], w_next)
            if pair not in seen_bigrams:
                self.continuation[w_next] += 1
                seen_bigrams.add(pair)

        # 所有階層的 count
        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n):
                ctx = tuple(tokens[i : i + n - 1])   # length n-1
                w   = tokens[i + n - 1]
                self.count[(ctx, w)] += 1
                self.ctx_count[ctx]  += 1
                self.ctx_vocab[ctx].add(w)

        print(f"[Stats] 訓練完成，最高階={self.order}")
        for n in range(1, self.order + 1):
            ctx_len = n - 1
            entries = [(k, v) for k, v in self.count.items() if len(k[0]) == ctx_len]
            print(f"  {n}-gram：{len(entries)} 條記錄")


# ════════════════════════════════════════════════════════════
#  Kneser-Ney 語言模型
# ════════════════════════════════════════════════════════════

class KneserNeyLM:
    """
    Variable-order Markov 模型，以 Kneser-Ney Smoothing 做插值。

    遞迴公式：
        P_KN(w | ctx) = [max(c(ctx,w) - d, 0) / c(ctx)]
                      + λ(ctx) * P_KN(w | ctx[1:])

        P_KN(w | ∅)   = continuation(w) / Σ continuation(*)   ← unigram fallback
    """

    def __init__(self, order: int = ORDER, discount: float = DISCOUNT):
        self.order    = order
        self.d        = discount
        self.stats    = NgramStats(order)

    def train(self, tokens: list[str]):
        self.stats.fit(tokens)
        # 計算 unigram continuation 總數
        self._cont_total = sum(self.stats.continuation.values()) or 1

    # ── 核心遞迴 ──────────────────────────────────────────────

    def _pkn(self, w: str, ctx: tuple) -> float:
        """計算 P_KN(w | ctx)，ctx 是 tuple，可為空（unigram fallback）。"""
        if not ctx:
            # unigram continuation probability
            return self.stats.continuation.get(w, 0) / self._cont_total

        c_ctx_w = self.stats.count.get((ctx, w), 0)
        c_ctx   = self.stats.ctx_count.get(ctx, 0)
        vocab   = self.stats.ctx_vocab.get(ctx, set())

        if c_ctx == 0:
            # context 未見過，直接 backoff
            return self._pkn(w, ctx[1:])

        # 折扣項
        numerator = max(c_ctx_w - self.d, 0) / c_ctx
        # backoff weight λ(ctx) = d * |{w : c(ctx,w)>0}| / c(ctx)
        lam = self.d * len(vocab) / c_ctx
        return numerator + lam * self._pkn(w, ctx[1:])

    # ── 取得整個詞彙的機率分布 ────────────────────────────────

    def _vocab(self) -> set[str]:
        """所有出現過的詞（包含 <EOS>）。"""
        return set(self.stats.continuation.keys())

    def next_dist(self, ctx: tuple) -> dict[str, float]:
        """回傳 {word: prob} 字典（未 normalize，用於抽樣）。"""
        # 找最短有效 context（不超過 order-1）
        ctx = ctx[-(self.order - 1):]
        vocab = self._vocab()
        dist = {w: self._pkn(w, ctx) for w in vocab}
        return dist

    # ── 生成 ──────────────────────────────────────────────────

    def generate(self, prompt: list[str], max_len=MAX_GEN,
                 temperature=TEMPERATURE, top_k=TOP_K) -> str:
        seq = list(prompt)
        for _ in range(max_len):
            ctx  = tuple(seq[-(self.order - 1):])
            dist = self.next_dist(ctx)
            nxt  = sample(dist, temperature, top_k)
            if nxt == "<EOS>":
                break
            seq.append(nxt)
        return "".join(seq[len(prompt):])

    # ── 評估 ──────────────────────────────────────────────────

    def log_prob(self, tokens: list[str]) -> float:
        """計算整個序列的 log probability。"""
        lp = 0.0
        for i in range(self.order - 1, len(tokens)):
            ctx = tuple(tokens[i - (self.order - 1) : i])
            w   = tokens[i]
            p   = self._pkn(w, ctx)
            lp += math.log(p + 1e-30)
        return lp

    def perplexity(self, tokens: list[str]) -> float:
        n  = len(tokens) - (self.order - 1)
        lp = self.log_prob(tokens)
        return math.exp(-lp / n) if n > 0 else float("inf")


# ════════════════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════════════════

def main():
    corpus_file = sys.argv[1] if len(sys.argv) > 1 else CORPUS_FILE
    random.seed(SEED)

    print("=" * 55)
    print(f"  Kneser-Ney Variable-order Markov Language Model")
    print(f"  語料：{corpus_file}  |  order={ORDER}  |  discount={DISCOUNT}")
    print("=" * 55)

    tokens = load_corpus(corpus_file)
    print(f"總 token 數：{len(tokens)}")

    model = KneserNeyLM(order=ORDER, discount=DISCOUNT)
    model.train(tokens)

    ppl = model.perplexity(tokens)
    print(f"\n訓練集困惑度（Perplexity）：{ppl:.2f}  ← 越低越好")

    # ── 固定 prompt 示範 ──────────────────────────────────────
    prompts = [["小", "貓"], ["天", "上"], ["我", "喜"], ["今", "天"]]
    print(f"\n【生成示範】temperature={TEMPERATURE}  top_k={TOP_K}")
    print("-" * 45)
    for p in prompts:
        gen = model.generate(p, temperature=TEMPERATURE, top_k=TOP_K)
        print(f"  {''.join(p)} → {''.join(p)}{gen}")

    # ── 互動模式 ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  互動模式（輸入至少兩個字，Ctrl-C 離開）")
    print("=" * 55)
    try:
        t_in = input(f"溫度 temperature [{TEMPERATURE}]：").strip()
        t = float(t_in) if t_in else TEMPERATURE
        k_in = input(f"top_k [{TOP_K}]（0=不限）：").strip()
        k = int(k_in) if k_in else TOP_K
        while True:
            user = input("\nprompt：").strip()
            if not user:
                continue
            p = list(user)
            if len(p) < ORDER - 1:
                print(f"  ⚠ 請至少輸入 {ORDER - 1} 個字")
                continue
            gen = model.generate(p, temperature=t, top_k=k)
            print(f"  → {user}{gen}")
    except (KeyboardInterrupt, EOFError):
        print("\n再見！")


if __name__ == "__main__":
    main()
