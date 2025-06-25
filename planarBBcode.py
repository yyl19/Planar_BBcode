import numpy as np
import matplotlib.pyplot as plt
import pulp
from bposd.css import css_code
import pickle
def generate_combined_xtile_ztile_edges(xtile, ztile, g=10, w=10, l=2, r=2, up=2, down=2, tile_size=4):
    # 计算实际网格尺寸
    cols = w + l + r + (tile_size - 1)
    rows = g + up + down + (tile_size - 1)
    total_points = rows * cols
    Hx = np.zeros((total_points, total_points), dtype=int)
    Hz = np.zeros((total_points, total_points), dtype=int)

    def point_id(x, y):
        return y * cols + x

    # === 添加 xtile：从每个 (x, y) in 全体逻辑区域为原点，向右上放置 xtile ===
    for base_y in range(rows - tile_size + 1):  # up/down 扩展区域
        for base_x in range(l, l + w):  # x-tile 在宽度方向 w 上平移
            for i in range(tile_size * tile_size):
                for j in range(tile_size * tile_size):
                    if xtile[i, j] == 1:
                        x1 = base_x + (i % tile_size)
                        y1 = base_y + (i // tile_size)
                        x2 = base_x + (j % tile_size)
                        y2 = base_y + (j // tile_size)
                        if 0 <= x1 < cols and 0 <= y1 < rows and 0 <= x2 < cols and 0 <= y2 < rows:
                            u = point_id(x1, y1)
                            v = point_id(x2, y2)
                            Hx[u, v] = 1

    # === 添加 ztile：从每个 (x, y) in 全体逻辑区域为原点，向右上放置 ztile ===
    for base_y in range(down, down + g):  # ztile 沿纵轴平移
        for base_x in range(cols - tile_size + 1):  # l/r 扩展区域
            for i in range(tile_size * tile_size):
                for j in range(tile_size * tile_size):
                    if ztile[i, j] == 1:
                        x1 = base_x + (i % tile_size)
                        y1 = base_y + (i // tile_size)
                        x2 = base_x + (j % tile_size)
                        y2 = base_y + (j // tile_size)
                        if 0 <= x1 < cols and 0 <= y1 < rows and 0 <= x2 < cols and 0 <= y2 < rows:
                            u = point_id(x1, y1)
                            v = point_id(x2, y2)
                            Hz[u, v] = 1

    return Hx, Hz, rows, cols
# 分别提取 xtile 与 ztile 应用的 base_x, base_y 原点 (x, y) 坐标
def collect_separate_xtile_ztile_origin_positions(g=10, w=10, l=2, r=2, up=2, down=2, tile_size=4):
    cols = w + l + r + (tile_size - 1)
    rows = g + up + down + (tile_size - 1)

    xtile_origins = []
    ztile_origins = []

    # xtile 原点位置：base_y in [0, rows - tile_size], base_x in [l, l+w)
    for base_y in range(rows - tile_size + 1):
        for base_x in range(l, l + w):
            xtile_origins.append((base_x, base_y))

    # ztile 原点位置：base_y in [down, down+g), base_x in [0, cols - tile_size + 1)
    for base_y in range(down, down + g):
        for base_x in range(cols - tile_size+1 ):
            ztile_origins.append((base_x, base_y))

    return xtile_origins, ztile_origins
def get_origin_index_by_xy(x, y, origin_positions):
    """
    根据 (x, y) 坐标查找其在 origin_positions 中的编号（左下到右上编号）。
    origin_positions 是已排序的 [(x1, y1), (x2, y2), ...]
    """
    try:
        return origin_positions.index((x, y))
    except ValueError:
        return -1  # 表示该坐标不在 origin_positions 中
def number_edges_with_xy_coordinates(xalltile, zalltile,g=10 ,w=6,tile_size=4, l=2, r=2, up=2, down=2):
    rows=2+g+2+tile_size-1
    cols=2+w+2+tile_size-1
    core_left=l
    core_right=l+w+tile_size-1
    core_bottom=down
    core_top=down+g+tile_size-1
    radius = 0.1
    positions = {}
    edge_labels = {}
    edge_id = 0

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            positions[idx] = (j, i)

    def has_edge(u, v):
        return (xalltile[u, v] == 1 or xalltile[v, u] == 1 or
                zalltile[u, v] == 1 or zalltile[v, u] == 1)

    for i in range(core_bottom, core_top):
        for j in range(core_left, core_right):
            u = i * cols + j
            x1, y1 = positions[u]

            # 横边：右边界横边
            if j < core_right or (j == core_right and i < core_top):
                v = u + 1
                if has_edge(u, v):
                    x2, y2 = positions[v]
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    edge_labels[edge_id] = ((x1, y1), (x2, y2))
                    edge_id += 1

            # 竖边：上边界竖边
            if (i < core_top) or (i == core_top and j < core_right):
                v = u + cols
                if v < rows * cols and has_edge(u, v):
                    x2, y2 = positions[v]
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    edge_labels[edge_id] = ((x1, y1), (x2, y2))
                    edge_id += 1

    for idx, (x, y) in positions.items():
        circle = plt.Circle((x, y), radius, fill=True, facecolor='white',
                            edgecolor='black', linewidth=1.0, zorder=3)


    return edge_labels
def get_edge_index_by_coordinates(p1, p2, edge_labels):
    """
    给定两个坐标元组 (x1, y1), (x2, y2)，返回它在 edge_labels 中对应的边编号。
    边是无向的，因此 (p1, p2) 和 (p2, p1) 都视为同一条边。
    """
    key_set = frozenset([p1, p2])
    for edge_id, (q1, q2) in edge_labels.items():
        if frozenset([q1, q2]) == key_set:
            return edge_id
    return -1  # 未找到
def generate_Hx_from_xtile_placement(
    xtile,
    origin_positions,          # [(x,y), …] 已按左下→右上排序
    edge_labels_custom_xy,     # {edge_id: ((x1,y1),(x2,y2)), …}
    cols,
    rows,
    tile_size=4,
    g=10, w=10, l=2, r=2, up=2, down=2
):
    n_pts   = len(origin_positions)
    n_edges = len(edge_labels_custom_xy)
    Hx = np.zeros((n_pts, n_edges), dtype=int)

    # ——— 工具函数 ——————————————————————————
    def origin_index(x, y):
        try:
            return origin_positions.index((x, y))
        except ValueError:
            return -1

    def edge_index(p1, p2):
        s = frozenset((p1, p2))
        for eid, (q1, q2) in edge_labels_custom_xy.items():
            if frozenset((q1, q2)) == s:
                return eid
        return -1
    # ————————————————————————————————

    #  遍历 g × w 个 xtile 原点
    for base_y in range(rows - tile_size + 1):     # y方向平移范围
        for base_x in range(l, l + w):             # x方向平移范围
            row_idx = origin_index(base_x, base_y)
            if row_idx == -1:
                continue  # 跳过未包含的原点

            # 扫 xtile 中所有为 1 的项
            for i in range(tile_size * tile_size):
                for j in range(tile_size * tile_size):
                    if xtile[i, j] == 0:
                        continue

                    x1, y1 = base_x + (i % tile_size), base_y + (i // tile_size)
                    x2, y2 = base_x + (j % tile_size), base_y + (j // tile_size)

                    if not (0 <= x1 < cols and 0 <= y1 < rows and
                            0 <= x2 < cols and 0 <= y2 < rows):
                        continue

                    col_idx = edge_index((x1, y1), (x2, y2))
                    if col_idx != -1:
                        Hx[row_idx, col_idx] = 1

    return Hx
def generate_Hz_from_ztile_placement(
    ztile,
    origin_positions,
    edge_labels_custom_xy,
    cols,
    rows,
    tile_size=4,
    g=10, w=6, l=2, r=2, up=2, down=2,
):
    n_pts = len(origin_positions)
    n_edges = len(edge_labels_custom_xy)
    Hz = np.zeros((n_pts, n_edges), dtype=int)

    def get_origin_index_by_xy(x, y):
        try:
            return origin_positions.index((x, y))
        except ValueError:
            return -1

    def get_edge_index_by_coordinates(p1, p2):
        key_set = frozenset([p1, p2])
        for edge_id, (q1, q2) in edge_labels_custom_xy.items():
            if frozenset([q1, q2]) == key_set:
                return edge_id
        return -1

    #  遍历 g × w 个原点位置
    for base_y in range(down, down + g):        # y方向平移
        for base_x in range(0, l + w+r):           # x方向平移
            row_idx = get_origin_index_by_xy(base_x, base_y)
            if row_idx == -1:
                continue

            for i in range(tile_size * tile_size):
                for j in range(tile_size * tile_size):
                    if ztile[i, j] == 0:
                        continue

                    x1, y1 = base_x + (i % tile_size), base_y + (i // tile_size)
                    x2, y2 = base_x + (j % tile_size), base_y + (j // tile_size)

                    if not (0 <= x1 < cols and 0 <= y1 < rows and 0 <= x2 < cols and 0 <= y2 < rows):
                        continue

                    edge_idx = get_edge_index_by_coordinates((x1, y1), (x2, y2))
                    if edge_idx != -1:
                        Hz[row_idx, edge_idx] = 1

    return Hz
def distance_test(stab, logicOp):
    # 稀疏矩阵安全转换
    if hasattr(logicOp, "toarray"):
        logicOp = logicOp.toarray().flatten()
    else:
        logicOp = np.array(logicOp).flatten()

    if hasattr(stab, "toarray"):
        stab = stab.toarray()
    else:
        stab = np.array(stab)
    n = stab.shape[1]
    m = stab.shape[0]

    wstab = int(np.max(np.sum(stab, axis=1)))
    wlog = int(np.count_nonzero(logicOp))

    num_anc_stab = int(np.ceil(np.log2(wstab)))
    num_anc_logical = int(np.ceil(np.log2(wlog)))
    num_var = n + m * num_anc_stab + num_anc_logical

    model = pulp.LpProblem("Minimum_Weight_Logical", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(num_var)]

    # 目标函数：最小化前 n 个变量（代表 qubit 是否作用 Z）
    model += pulp.lpSum(x[i] for i in range(n))

    # 约束：stab @ x % 2 == 0
    for row in range(m):
        weight = [0] * num_var
        for q in np.nonzero(stab[row])[0]:
            weight[q] = 1
        cnt = 1
        for q in range(num_anc_stab):
            weight[n + row * num_anc_stab + q] = -(1 << cnt)
            cnt += 1
        model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 0

    # 约束：logicOp @ x % 2 == 1
    weight = [0] * num_var
    for q in np.nonzero(logicOp)[0]:
        weight[q] = 1
    cnt = 1
    for q in range(num_anc_logical):
        weight[n + m * num_anc_stab + q] = -(1 << cnt)
        cnt += 1
    model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 1

    # model.solve(pulp.PulpSolverDefault(msg=False))
    model.solve(pulp.PULP_CBC_CMD(msg=False))


    opt_val = sum([pulp.value(x[i]) for i in range(n)])
    return int(opt_val)
if __name__ == "__main__":
    wid=7
    leng=7
    xtile = np.zeros((16, 16), dtype=int)
    xtile[0, 1] = 1
    xtile[2, 6] = 1
    xtile[6, 7] = 1
    xtile[10, 11] = 1
    xtile[8, 12] = 1
    xtile[9, 13] = 1
    ztile = np.zeros((16, 16), dtype=int)
    ztile[0, 4] = 1
    ztile[4, 8] = 1
    ztile[8, 9] = 1
    ztile[10, 14] = 1
    ztile[1, 2] = 1
    ztile[2, 3] = 1
    xalltile, zalltile, combined_rows, combined_cols = generate_combined_xtile_ztile_edges(
        xtile, ztile, g=leng, w=wid, l=2, r=2, up=2, down=2, tile_size=4
    )
    xtile_origins, ztile_origins = collect_separate_xtile_ztile_origin_positions(
        g=leng, w=wid, l=2, r=2, up=2, down=2, tile_size=4
    )
    # 返回 (坐标, 坐标) 的边映射
    edge_labels_custom_xy = number_edges_with_xy_coordinates(
        xalltile, zalltile,g=leng ,w=wid,tile_size=4, l=2, r=2, up=2, down=2
    )
    Hx_xtile = generate_Hx_from_xtile_placement(
        xtile=xtile,
        origin_positions=xtile_origins,  # 长度应为 g × w
        edge_labels_custom_xy=edge_labels_custom_xy,
        cols=combined_cols,
        rows=combined_rows,
        tile_size=4,
        g=leng, w=wid, l=2, r=2, up=2, down=2
    )
    Hz_ztile = generate_Hz_from_ztile_placement(
        ztile=ztile,
        origin_positions=ztile_origins,
        edge_labels_custom_xy=edge_labels_custom_xy,
        cols=combined_cols,
        rows=combined_rows,
        tile_size=4,
        g=leng, w=wid, l=2, r=2, up=2, down=2,
    )
    qcode=css_code(Hx_xtile,Hz_ztile)
    print('Testing CSS code...')
    qcode.test()
    print('Done')
    lz = qcode.lz
    lx = qcode.lx
    k = lz.shape[0]
    print('Computing code distance...')
    d = Hx_xtile.shape[1]
    for i in range(k):
        w = distance_test(Hx_xtile,lx[i,:])
        print('Logical qubit=',i,'Distance=',w)
        d = min(d,w)
    print('Code parameters: n,k,d=',Hx_xtile.shape[1],k,d)
    filename = f"./data/hxhz_{wid}_{leng}.pkl"
    with open(filename, "wb") as f:
        pickle.dump({
            "xposition": xtile_origins,
            "zposition": ztile_origins,
            "edge_map": edge_labels_custom_xy,
            "Hx": Hx_xtile,
            "Hz": Hz_ztile
        }, f)
    print(f"The [[{Hx_xtile.shape[1]},{k},{d}]] code has been saved to ''{filename}''.")

