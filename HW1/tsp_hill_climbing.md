# 旅行推銷員問題 — 爬山演算法 (2-opt)

## 問題定義

**旅行推銷員問題 (Travelling Salesman Problem, TSP)**：

> 給定 N 個城市和彼此間的距離，找出一條經過每個城市**恰好一次**、最後回到起點的**最短路徑**。

本程式用 **爬山演算法 (Hill Climbing)** 搭配 **2-opt 鄰居操作**來求解。

---

## 演算法架構

```
初始解
  │
  ▼
height(current)
  │
  └─► 產生 neighbor(current)
          │
          ▼
       height(neighbor) > height(current)?
          │ Yes                  │ No
          ▼                     ▼
    current = neighbor      捨棄，繼續
          │
          ▼
    重複直到 max_iter 用完
```

---

## 三個核心函式

### 1. `height(tour, cities)` — 評估函式

```python
def height(tour, cities):
    return -total_dist(tour, cities)
```

| 概念 | 說明 |
|------|------|
| **爬山目標** | 最大化 height |
| **等價目標** | 最小化總距離 |
| **為何取負** | 爬山演算法找極大值，距離要極小，所以乘 -1 轉換 |

總距離計算：

$$\text{total\_dist} = \sum_{i=0}^{n-1} \sqrt{(x_{t_i} - x_{t_{i+1}})^2 + (y_{t_i} - y_{t_{i+1}})^2}$$

其中 $t_{n} = t_0$（最後一個城市接回起點）。

---

### 2. `neighbor(tour)` — 鄰居產生 (2-opt)

```python
def neighbor(tour):
    i = random.randint(0, n - 3)
    j = random.randint(i + 2, n - 1)
    new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
    return new_tour
```

**2-opt 操作說明**：

隨機選兩條不相鄰的邊：

```
edge1 = (a, b) = (tour[i],   tour[i+1])
edge2 = (c, d) = (tour[j],   tour[j+1])
```

刪掉這兩條邊，將中間段 `tour[i+1 .. j]` **反轉**，等效於：

```
刪除：(a, b)  和  (c, d)
新增：(a, c)  和  (b, d)
```

**圖示**：

```
原本：  ... → a → b → b+1 → ... → c → d → ...
                  ↑___反轉這段___↑

之後：  ... → a → c → c-1 → ... → b → d → ...
```

> **為什麼不是 `(a,d)(b,c)`？**  
> 從一個 cycle 刪掉兩條非相鄰邊後，只剩下兩條獨立路徑。  
> 唯一合法的重連方式是讓兩條路徑**頭尾相接**，即 `(a,c)(b,d)`。  
> 若選擇 `(a,d)(b,c)` 則兩條路徑各自形成子環，**不構成合法的 Hamiltonian cycle**。

**選邊限制**：

- `j >= i + 2`：確保兩條邊不相鄰（相鄰邊反轉後等於沒變）
- 排除 `i=0, j=n-1`：這組等同於選到繞回頭的那條邊，也跳過

---

### 3. `hill_climbing(cities, max_iter=100_000)` — 主演算法

```python
def hill_climbing(cities, max_iter=100_000):
    current = list(range(n))      # 初始解: 0→1→2→...→n-1→(回0)
    cur_h   = height(current, cities)

    for _ in range(max_iter):
        nbr   = neighbor(current)
        nbr_h = height(nbr, cities)
        if nbr_h > cur_h:
            current, cur_h = nbr, nbr_h

    return current, -cur_h, improved
```

| 項目 | 說明 |
|------|------|
| **初始解** | 照城市編號順序：0 → 1 → 2 → … → n-1 → 0 |
| **接受條件** | 嚴格改善（`nbr_h > cur_h`） |
| **終止條件** | 執行 `max_iter` 次後停止 |
| **缺點** | 容易卡在 local optimum，無法跳出 |

---

## 執行結果範例

環境：20 個城市，座標隨機生成（seed=42），max_iter=100,000

```
城市數:   20
初始距離: 1090.33
最短距離: 452.50
改善幅度: 58.5%
接受改善: 35 次
最佳路徑: 0 → 4 → 13 → 6 → 5 → 8 → 16 → 15 → 10 → 18 → 14 → 19 → 12 → 2 → 7 → 17 → 1 → 11 → 3 → 9 → 0
```

初始解（照編號走）距離 1090，爬山後降到 452，改善近六成。

---

## 限制與改進方向

| 問題 | 改進方法 |
|------|---------|
| 容易卡 local optimum | **模擬退火 (SA)**：以機率接受較差的解來跳出 |
| 結果不穩定（依初始解） | **隨機重啟 (Random Restart)**：多次執行取最佳 |
| 2-opt 鄰居是隨機選一個 | **最佳 2-opt**：每輪掃描所有可能，選最好的 |
| 城市數多時 local optimum 更嚴重 | **基因演算法 / 蟻群演算法** |

---

## 執行方式

```bash
python tsp_hill_climbing.py
```

只需要 Python 標準函式庫，不需要安裝任何套件。
