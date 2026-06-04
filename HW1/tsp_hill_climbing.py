import random
import math

# ── 城市生成 ─────────────────────────────────────────────────────────────────
def create_cities(n, seed=42):
    random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]

# ── 總距離 ───────────────────────────────────────────────────────────────────
def total_dist(tour, cities):
    n = len(tour)
    return sum(
        math.hypot(
            cities[tour[i]][0] - cities[tour[(i + 1) % n]][0],
            cities[tour[i]][1] - cities[tour[(i + 1) % n]][1],
        )
        for i in range(n)
    )

# ── Height: 距離越短 height 越高 ─────────────────────────────────────────────
def height(tour, cities):
    """height = -總距離，爬山就是最小化距離"""
    return -total_dist(tour, cities)

# ── Neighbor: 2-opt swap ─────────────────────────────────────────────────────
def neighbor(tour):
    """
    隨機選兩條邊:
      edge1 = (a, b) = (tour[i], tour[i+1])
      edge2 = (c, d) = (tour[j], tour[j+1])   (j+1 wraps mod n)

    2-opt 操作：反轉中間段 tour[i+1 .. j]
    等效於：刪掉 (a,b) 和 (c,d)，改連 (a,c) 和 (b,d)

    ※ 注意：從 cycle 刪掉兩條非相鄰邊後，唯一合法的重連方式是
      (a,c)(b,d)，而非 (a,d)(b,c)（後者會分裂成兩個子環）
      所以實作上用反轉中間段来達成。
    """
    n = len(tour)
    # 隨機選 i < j，且至少相差 2（不然是相鄰邊，等於沒動）
    i = random.randint(0, n - 3)
    j = random.randint(i + 2, n - 1)

    # 如果選到 i=0, j=n-1 這組，等於兩條相鄰邊 (繞回頭)，跳過
    if i == 0 and j == n - 1:
        return tour[:]

    new_tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
    return new_tour

# ── 爬山演算法 ───────────────────────────────────────────────────────────────
def hill_climbing(cities, max_iter=100_000):
    """
    Simple Hill Climbing：只要 neighbor height > current height 就走過去
    缺點：容易卡在 local optimum
    """
    n = len(cities)
    current = list(range(n))          # 初始解: 0→1→2→...→n-1→(回0)
    cur_h   = height(current, cities)

    improved = 0
    for step in range(max_iter):
        nbr   = neighbor(current)
        nbr_h = height(nbr, cities)
        if nbr_h > cur_h:
            current, cur_h = nbr, nbr_h
            improved += 1

    return current, -cur_h, improved

# ── 主程式 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 20
    cities = create_cities(N)

    init_tour = list(range(N))
    init_d    = total_dist(init_tour, cities)

    print(f"城市座標 (前5筆):")
    for i, (x, y) in enumerate(cities[:5]):
        print(f"  城市 {i}: ({x:.2f}, {y:.2f})")
    print(f"  ...")
    print()

    best_tour, best_d, improved_count = hill_climbing(cities)

    print(f"城市數:   {N}")
    print(f"初始距離: {init_d:.2f}")
    print(f"最短距離: {best_d:.2f}")
    print(f"改善幅度: {(init_d - best_d) / init_d * 100:.1f}%")
    print(f"接受改善: {improved_count} 次")
    print(f"最佳路徑: {' → '.join(map(str, best_tour))} → {best_tour[0]}")
