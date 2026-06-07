"""
ex2_mlp.py — 用 nn0.py 手刻 MLP 學習 XOR

架構：2 → 4 → 1  (hidden layer 用 tanh, output 用 sigmoid)
損失：MSE (Mean Squared Error)
優化：Adam

XOR 真值表：
  (0,0) → 0
  (0,1) → 1
  (1,0) → 1
  (1,1) → 0
"""

import random
import math
from nn0 import Value, Adam

random.seed(42)

# ────────────────────────────────────────────────────────────
# 1. 初始化參數（Xavier 初始化）
# ────────────────────────────────────────────────────────────

def xavier(fan_in, fan_out):
    limit = math.sqrt(6 / (fan_in + fan_out))
    return Value(random.uniform(-limit, limit))

# Layer 1: 2 → 4
W1 = [[xavier(2, 4) for _ in range(2)] for _ in range(4)]
b1 = [Value(0.0) for _ in range(4)]

# Layer 2: 4 → 1
W2 = [[xavier(4, 1) for _ in range(4)] for _ in range(1)]
b2 = [Value(0.0) for _ in range(1)]

# 蒐集所有可學習參數
params = (
    [W1[i][j] for i in range(4) for j in range(2)] +
    b1 +
    [W2[0][j] for j in range(4)] +
    b2
)
print(f"參數總數: {len(params)}")

# ────────────────────────────────────────────────────────────
# 2. 前向傳播
# ────────────────────────────────────────────────────────────

def mlp_forward(x0, x1):
    """兩個輸入值（float），回傳 Value（0~1 之間）"""
    x = [Value(float(x0)), Value(float(x1))]

    # Hidden layer
    h = []
    for i in range(4):
        net = W1[i][0] * x[0] + W1[i][1] * x[1] + b1[i]
        h.append(net.tanh())

    # Output layer
    out_net = sum(W2[0][j] * h[j] for j in range(4)) + b2[0]
    return out_net.sigmoid()


# ────────────────────────────────────────────────────────────
# 3. 訓練資料（XOR）
# ────────────────────────────────────────────────────────────

dataset = [
    ((0, 0), 0.0),
    ((0, 1), 1.0),
    ((1, 0), 1.0),
    ((1, 1), 0.0),
]

# ────────────────────────────────────────────────────────────
# 4. 訓練迴圈（MSE + Adam）
# ────────────────────────────────────────────────────────────

optimizer = Adam(params, lr=0.05)
num_steps = 2000

print("\n=== 開始訓練 XOR MLP ===")
for step in range(1, num_steps + 1):
    # 線性學習率衰減
    lr_t = 0.05 * max(1 - step / num_steps, 0.01)

    total_loss = Value(0.0)
    for (x0, x1), target in dataset:
        pred = mlp_forward(x0, x1)
        t = Value(target)
        err = pred - t
        total_loss = total_loss + err * err   # MSE per sample

    mse = total_loss * (1 / 4)

    optimizer.zero_grad()
    mse.backward()
    optimizer.step(lr_t)

    if step % 200 == 0:
        print(f"Step {step:4d}  MSE={mse.data:.6f}  lr={lr_t:.5f}")


# ────────────────────────────────────────────────────────────
# 5. 測試結果
# ────────────────────────────────────────────────────────────

print("\n=== 測試結果 ===")
print(f"{'輸入':^12}  {'目標':^6}  {'預測':^8}  {'判斷'}")
print("-" * 40)
correct = 0
for (x0, x1), target in dataset:
    pred = mlp_forward(x0, x1)
    guess = 1 if pred.data >= 0.5 else 0
    ok = "✅" if guess == int(target) else "❌"
    if guess == int(target):
        correct += 1
    print(f"({x0}, {x1})        {int(target)}      {pred.data:.4f}    {ok}")

print(f"\n準確率: {correct}/{len(dataset)} ({100*correct/len(dataset):.0f}%)")
