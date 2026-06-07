"""
nn0.py — 自動微分引擎 (Value) 與 Adam 優化器

提供：
  class Value  — 純 Python autograd 節點
  class Adam   — Adam optimizer
  linear()     — 矩陣乘法 (W @ x + b)
  softmax()    — 數值穩定 softmax
  rmsnorm()    — RMS Normalization
  gd()         — 訓練循環（梯度下降 + Adam + lr decay）
"""

import math
import random

# ────────────────────────────────────────────────────────────
# 1. 自動微分節點
# ────────────────────────────────────────────────────────────

class Value:
    """純 Python 的自動微分節點，支援反向傳播。"""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        val = math.exp(self.data)
        return Value(val, (self,), (val,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def __neg__(self):          return self * -1
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-other)
    def __rsub__(self, other):  return other + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):   return self * other ** -1
    def __rtruediv__(self, other):  return other * self ** -1

    def backward(self):
        """反向傳播：拓撲排序後逐層計算梯度。"""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def zero_grad(self):
        self.grad = 0

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"


# ────────────────────────────────────────────────────────────
# 2. Adam 優化器
# ────────────────────────────────────────────────────────────

class Adam:
    """Adam optimizer，支援 learning rate 線性衰減。"""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(self.params)  # 一階動量
        self.v = [0.0] * len(self.params)  # 二階動量

    def step(self, lr_t=None):
        """更新所有參數。lr_t 為當前步驟的有效學習率（可做 decay）。"""
        if lr_t is None:
            lr_t = self.lr
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= lr_t * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0


# ────────────────────────────────────────────────────────────
# 3. 神經網路基礎運算
# ────────────────────────────────────────────────────────────

def linear(W, x, b=None):
    """全連接層: out[i] = sum_j W[i][j] * x[j] + b[i]
    W: list[list[Value]], shape (out, in)
    x: list[Value], shape (in,)
    b: list[Value] or None, shape (out,)
    returns: list[Value], shape (out,)
    """
    out = []
    for i in range(len(W)):
        s = sum(W[i][j] * x[j] for j in range(len(x)))
        if b is not None:
            s = s + b[i]
        out.append(s)
    return out


def softmax(logits):
    """數值穩定 softmax，回傳 list[Value]。"""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(e.data for e in exps)
    total_v = Value(total)
    return [e / total_v for e in exps]


def rmsnorm(x, g):
    """RMS Layer Normalization（不含 bias）。
    x: list[Value]
    g: list[Value]  (可學習的縮放參數)
    """
    n = len(x)
    ms = sum(xi * xi for xi in x) * (1 / n)
    rms = (ms + Value(1e-8)) ** 0.5
    return [g[i] * (x[i] / rms) for i in range(n)]


def cross_entropy(probs, target_id):
    """負對數似然損失。"""
    return -probs[target_id].log()


# ────────────────────────────────────────────────────────────
# 4. 訓練循環
# ────────────────────────────────────────────────────────────

def gd(model_fn, dataset, params, num_steps=500, lr=1e-2, verbose=True,
       log_every=50, clip_grad=5.0):
    """
    通用梯度下降訓練循環。

    Args:
        model_fn:  callable(x_ids) -> list[Value] probs
        dataset:   list of (input_token_ids, target_id)
        params:    list[Value]  (模型所有可學習參數)
        num_steps: 訓練步數
        lr:        初始學習率
        verbose:   是否印 loss
        log_every: 每幾步印一次
        clip_grad: 梯度裁剪閾值（防止梯度爆炸）

    Returns:
        losses: list[float]
    """
    optimizer = Adam(params, lr=lr)
    losses = []

    for step in range(1, num_steps + 1):
        # Cosine lr 衰減（比線性衰減更平滑）
        t_ratio = (step - 1) / max(num_steps - 1, 1)
        lr_t = lr * 0.5 * (1 + math.cos(math.pi * t_ratio))
        lr_t = max(lr_t, lr * 0.01)

        # 隨機取樣一筆資料
        x_ids, target_id = random.choice(dataset)

        # Forward
        probs = model_fn(x_ids)
        loss = cross_entropy(probs, target_id)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（Global Gradient Clipping）
        if clip_grad is not None:
            total_norm = math.sqrt(sum(p.grad ** 2 for p in params))
            if total_norm > clip_grad:
                scale = clip_grad / (total_norm + 1e-8)
                for p in params:
                    p.grad *= scale

        # Update
        optimizer.step(lr_t)

        loss_val = loss.data
        if math.isnan(loss_val):
            print(f"警告：step {step} loss=NaN，停止訓練")
            break
        losses.append(loss_val)
        if verbose and step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step:4d}/{num_steps}  loss={avg:.4f}  lr={lr_t:.5f}")

    return losses
