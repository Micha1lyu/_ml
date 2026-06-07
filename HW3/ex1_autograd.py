"""
ex1_autograd.py — 示範 Value 自動微分引擎

目標：理解 nn0.py 的核心機制
      ✅ 建構運算圖
      ✅ 反向傳播計算梯度
      ✅ 與手算結果比對驗證正確性

範例：f(x, y) = (x * y + 1)^2 在 x=2, y=3 的梯度
  df/dx = 2*(x*y+1)*y = 2*(7)*3 = 42
  df/dy = 2*(x*y+1)*x = 2*(7)*2 = 28
"""

from nn0 import Value

def demo_basic():
    print("=" * 50)
    print("【範例 1】純量運算與梯度驗證")
    print("=" * 50)

    x = Value(2.0)
    y = Value(3.0)

    # 建構運算圖: f = (x*y + 1)^2
    z = x * y + 1      # z = 7
    f = z ** 2         # f = 49

    print(f"x={x.data}, y={y.data}")
    print(f"z = x*y + 1 = {z.data}")
    print(f"f = z^2     = {f.data}")

    # 反向傳播
    f.backward()

    print(f"\ndf/dx = {x.grad:.4f}  (手算: 42)")
    print(f"df/dy = {y.grad:.4f}  (手算: 28)")
    assert abs(x.grad - 42) < 1e-9
    assert abs(y.grad - 28) < 1e-9
    print("✅ 梯度驗證通過！\n")


def demo_chain_rule():
    print("=" * 50)
    print("【範例 2】鏈式法則：relu + log")
    print("=" * 50)
    # f(x) = log(relu(x) + 1)，x = 2
    # relu(2) = 2, relu'(2) = 1
    # f = log(3), f' = 1/3 * 1 = 1/3
    x = Value(2.0)
    f = (x.relu() + 1).log()
    f.backward()
    print(f"f = log(relu(2)+1) = {f.data:.4f}  (應為 {__import__('math').log(3):.4f})")
    print(f"df/dx = {x.grad:.4f}  (手算: 1/3 ~= {1/3:.4f})")
    assert abs(x.grad - 1/3) < 1e-9
    print("✅ 鏈式法則驗證通過！\n")


def demo_neuron():
    print("=" * 50)
    print("【範例 3】單一神經元前向傳播")
    print("=" * 50)
    # neuron: out = tanh(w1*x1 + w2*x2 + b)
    x1 = Value(1.0);  x2 = Value(0.5)
    w1 = Value(0.8);  w2 = Value(-0.5)
    b  = Value(0.1)

    out = (w1 * x1 + w2 * x2 + b).tanh()
    print(f"輸入: x1={x1.data}, x2={x2.data}")
    print(f"權重: w1={w1.data}, w2={w2.data}, b={b.data}")
    print(f"net = {(0.8*1 + -0.5*0.5 + 0.1):.4f}")
    print(f"out = tanh(net) = {out.data:.4f}")

    out.backward()
    print(f"\ndout/dw1 = {w1.grad:.4f}")
    print(f"dout/dw2 = {w2.grad:.4f}")
    print(f"dout/db  = {b.grad:.4f}")
    print("✅ 神經元梯度計算完成！\n")


if __name__ == "__main__":
    demo_basic()
    demo_chain_rule()
    demo_neuron()
