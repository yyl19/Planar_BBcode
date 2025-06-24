import numpy as np
import pulp
from bposd.css import css_code
import pickle
import galois
import os
import sys
GF = galois.GF(2)
def distance_test(stab, logicOp):
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
    model += pulp.lpSum(x[i] for i in range(n))

    for row in range(m):
        weight = [0] * num_var
        for q in np.nonzero(stab[row])[0]:
            weight[q] = 1
        cnt = 1
        for q in range(num_anc_stab):
            weight[n + row * num_anc_stab + q] = -(1 << cnt)
            cnt += 1
        model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 0

    weight = [0] * num_var
    for q in np.nonzero(logicOp)[0]:
        weight[q] = 1
    cnt = 1
    for q in range(num_anc_logical):
        weight[n + m * num_anc_stab + q] = -(1 << cnt)
        cnt += 1
    model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    opt_val = sum([pulp.value(x[i]) for i in range(n)])
    return int(opt_val)
def gf2_rank_galois(matrix):
    """
    使用 galois 库在 GF(2) 上计算矩阵秩
    参数:
        matrix: numpy.ndarray 类型，元素为 0 或 1
    返回:
        GF(2) 域下的秩
    """
    GF = galois.GF(2)
    mat = GF(matrix)
    rank = np.linalg.matrix_rank(mat)  # galois 的 ndarray 可与 numpy 兼容
    return rank
def solve_gf2(AT, s):
    """
    在 GF(2) 中求解 AT x = s，处理可能的奇异情况
    
    参数:
        AT: 转置后的矩阵 (n×m 维)，可以是普通 numpy 数组
        s: 右侧向量 (m 维)，可以是普通 numpy 数组
    
    返回:
        x: 解向量 (n 维)，如果无解则返回 None
    """
    # 创建 GF(2) 域
    GF = galois.GF(2)
    
    # 将输入转换为 GF(2) 数组
    AT_gf2 = GF(AT)
    s_gf2 = GF(s)
    
    # 构造增广矩阵 [AT | s]
    augmented = np.hstack((AT_gf2, s_gf2.reshape(-1, 1)))
    
    # 行最简形（RREF）
    augmented_rref = augmented.row_reduce()
    
    # 检查是否有解（是否有形如 [0 ... 0 | 1] 的行）
    for row in augmented_rref:
        if np.all(row[:-1] == 0) and row[-1] != 0:
            print("方程组无解！")
            return None
    
    # 提取解（如果有自由变量，返回一个特解）
    n = AT.shape[1]  # 解向量的维度
    x = GF.Zeros(n)
    
    # 找出主元列
    pivot_cols = []
    for i in range(augmented_rref.shape[0]):
        pivot_pos = np.where(augmented_rref[i, :-1] == 1)[0]
        if len(pivot_pos) > 0:
            pivot_cols.append(pivot_pos[0])
    
    # 设置自由变量为 0，求特解
    free_vars = [col for col in range(n) if col not in pivot_cols]
    for col in free_vars:
        x[col] = 0
    
    # 回代求解主元变量
    for i, row in enumerate(augmented_rref):
        pivot_pos = np.where(row[:-1] == 1)[0]
        if len(pivot_pos) == 0:
            continue  # 全零行，跳过
        pivot_col = pivot_pos[0]
        x[pivot_col] = row[-1]
        for j in range(pivot_col + 1, n):
            if row[j] == 1:
                x[pivot_col] -= x[j]
    
    return x
def compute_distance_for_logical(i, Hx, lx_row):
    w = distance_test(Hx, lx_row)
    return i, w
def gf2_inverse_product(jx, jz):
    """
    计算在 GF(2) 域上 jx @ jz.T 的逆矩阵
    :param jx: numpy.ndarray, shape (k, n)
    :param jz: numpy.ndarray, shape (k, n)
    :return: GF(2) 上的逆矩阵
    """
    GF = galois.GF(2)

    # 保证输入合法
    jx = np.array(jx) % 2
    jz = np.array(jz) % 2

    # 计算乘积 M = jx @ jz.T  ∈ GF(2)
    M = (jx @ jz.T) % 2

    if M.shape[0] != M.shape[1]:
        raise ValueError("矩阵不是方阵，无法求逆")

    # 转为 GF(2) 域的矩阵
    M_gf2 = GF(M)

    # 判断是否可逆（行列式非 0）
    if gf2_rank_galois(M) < M.shape[0]:
        raise ValueError("矩阵在 GF(2) 上不可逆")

    # 求逆
    M_inv = np.linalg.inv(M_gf2)

    return M_inv
def gf2_matmul(G, jz):
    """
    在 GF(2) 上计算矩阵乘法 G @ jz
    :param G: numpy.ndarray, shape (m, n)
    :param jz: numpy.ndarray, shape (n, k)
    :return: G @ jz (mod 2)
    """
    G = np.array(G) % 2
    jz = np.array(jz) % 2

    product = (G @ jz) % 2
    return product
def gf2_add(a, b):
    """
    GF(2) 上的向量加法（按位异或）

    参数:
        a, b: numpy.ndarray 类型，shape 相同，元素为 0 或 1

    返回:
        numpy.ndarray: GF(2) 上 a + b 的结果
    """
    return (np.array(a) ^ np.array(b)) % 2
def mod2(A):
    return np.array(A) % 2
def insert_and_maintain_rank(lx5, m,m1):
    """
    将向量 x 插入 lx5 最上方，并删除 lx5 中一行使得新矩阵满秩。
    
    参数:
        lx5: numpy.ndarray, shape=(k, n)，0-1矩阵
        x: numpy.ndarray, shape=(n,) 或 (1, n)，0-1向量
    
    返回:
        lx51: numpy.ndarray, shape=(k, n)，满秩矩阵
        removed_index: 被删除的原 lx5 中的行索引
    """
    x=gf2_add(lx5[m], lx5[m1])
    # x=lx5[m]
    # lx5 = mod2(lx5)
    # x = mod2(x).reshape(1, -1)
    
    # 原矩阵秩
    orig_rank = gf2_rank_galois(lx5 )

    # 合并 x 到顶部
    lx51_temp = np.vstack([x, lx5])

    # 尝试删除每一行（从原 lx5 中）
    for i in range(1, lx51_temp.shape[0]):
        candidate = np.delete(lx51_temp, i, axis=0)
        if gf2_rank_galois(candidate ) == orig_rank:
            return candidate , i - 1  # i-1 是 lx5 中的行索引

    # 如果没有任何一行可以删，返回 None
    return None, None
def check_matrix_embedding(Hz174_sortedall, Hz54_sortedall):
    x, y = Hz54_sortedall.shape

    # 条件 1: 前 x 行 y 列是否与 Hz54_sortedall 相等
    top_left_match = np.array_equal(Hz174_sortedall[:x, :y], Hz54_sortedall)

    # 条件 2: 第 x 行之后的前 y 列是否为 0
    below_is_zero = np.all(Hz174_sortedall[x:, :y] == 0)

    return top_left_match, below_is_zero
def check_hx_matrix_embedding(Hx174_sortedall, Hx54_sortedall):
    x, y = Hx54_sortedall.shape

    # 条件1：前x行前y列与Hx54_sortedall是否相等
    top_left_match = np.array_equal(Hx174_sortedall[:x, :y], Hx54_sortedall)

    # 条件2：前x行第y列之后是否全为0
    right_is_zero = np.all(Hx174_sortedall[:x, y:] == 0)

    return top_left_match, right_is_zero
def compute_augmented_rank_correct(Hx174, J_x174, num_rows=6, seed=None):
    """
    在 GF(2) 上计算矩阵 [Hx174; J_x174; J_x174 + h] 的秩，
    其中 h 为 Hx174 中随机挑的 num_rows 行
    """
    if seed is not None:
        np.random.seed(seed)

    GF = galois.GF(2)
    Hx174 = GF(Hx174)   # 所有矩阵转为 GF(2) 类型
    J_x174 = GF(J_x174)

    assert Hx174.shape[1] == J_x174.shape[1], "列数必须一致"
    assert Hx174.shape[0] >= num_rows, f"Hx174 至少要有 {num_rows} 行"
    assert J_x174.shape[0] >= num_rows, f"J_x174 至少要有 {num_rows} 行"

    # 从 Hx174 中随机选 num_rows 行
    indices = np.random.choice(Hx174.shape[0], num_rows, replace=False)
    h_rows = Hx174[indices, :]  # shape = (num_rows, n)

    # GF(2) 中加法（自动模2）
    J_plus_h_rows = J_x174[:num_rows, :] + h_rows

    # 拼接三个矩阵
    combined = np.vstack([Hx174, J_x174, J_plus_h_rows])

    # 计算 GF(2) 中秩
    rank =  gf2_rank_galois(combined)

    return rank, indices
def pad_matrix_right_gf2(lx, target_num_cols):
    """
    在 GF(2) 上将矩阵 lx 右侧补 0，直到列数为 target_num_cols
    """
    lx = GF(lx)
    current_cols = lx.shape[1]

    if current_cols > target_num_cols:
        raise ValueError("lx 的列数已经超过目标列数")

    if current_cols == target_num_cols:
        return lx

    num_rows = lx.shape[0]
    padding = GF.Zeros((num_rows, target_num_cols - current_cols))
    padded_lx = np.hstack([lx, padding])
    return padded_lx

def solve_gf2_linear_system(lx5_padded, Hx174_sortedall):
    """
    对于每一行 lx5_padded[i]，解 g_i @ H = lx5_padded[i]，即 H.T @ g.T = lx5[i].T
    使用高斯消元法在 GF(2) 上解线性方程组。
    """
    H = GF(Hx174_sortedall)
    lx5 = GF(lx5_padded)
    H_T = H.T  # shape (128, r)
    solutions = []

    for i in range(lx5.shape[0]):
        b = lx5[i]  # shape (128,)
        # 增广矩阵: [H_T | b.T]
        Ab = np.hstack([H_T, b.reshape(-1, 1)])
        Ab_rref = Ab.row_reduce()

        # 检查是否存在无解：最后几行形如 [0 0 ... 0 | 1]（矛盾）
        rank_A = np.linalg.matrix_rank(H_T)
        rank_Ab = np.linalg.matrix_rank(Ab[:, :-1])
        if rank_Ab < np.linalg.matrix_rank(Ab):
            solutions.append(None)
        else:
            # 提取一个解
            num_vars = H.shape[0]  # r rows -> g_i has r components
            x = np.zeros((num_vars,), dtype=int)

            # 只提取主元位置的解
            leading_cols = []
            row_idx = 0
            for col_idx in range(Ab_rref.shape[1] - 1):  # 不包括增广列
                if row_idx < Ab_rref.shape[0] and Ab_rref[row_idx, col_idx] == 1:
                    x[col_idx] = Ab_rref[row_idx, -1]  # 解
                    leading_cols.append(col_idx)
                    row_idx += 1

            solutions.append(GF(x))

    return solutions
def get_zero_indices_exceeding_threshold(solution, threshold):
    """
    从 solution 中找出值为 0 且索引 > threshold 的位置
    """
    if solution is None:
        print("解为 None，跳过处理。")
        return []

    sol_array = np.array(solution, dtype=int)
    zero_indices = np.where(sol_array == 0)[0]

    # 只保留大于阈值的索引
    filtered_indices = zero_indices[zero_indices > threshold]
    return filtered_indices
def remove_rows_and_zero_columns(H, indices_to_remove):
    """
    从 H 中删除指定行，再删除全 0 列，并返回：
        - 处理后的新矩阵 H_new
        - 被删除的列编号列表 removed_col_indices
    """
    # Step 1: 删除指定行
    H_new = np.delete(H, indices_to_remove, axis=0)

    # Step 2: 找出全为 0 的列
    col_sums = np.sum(H_new, axis=0)
    removed_col_indices = np.where(col_sums == 0)[0]

    # Step 3: 删除这些列
    H_new = np.delete(H_new, removed_col_indices, axis=1)

    return H_new, removed_col_indices
def remove_columns_from_matrix(matrix, cols_to_remove):
    """
    从矩阵中删除指定列

    参数:
        matrix: 2D 矩阵 (如 lx5_padded)，可为 numpy.ndarray 或 galois.GF(2) array
        cols_to_remove: 要删除的列索引列表或数组

    返回:
        删除指定列后的新矩阵
    """
    return np.delete(matrix, cols_to_remove, axis=1)
def remove_columns_and_zero_rows(H, cols_to_remove):
    """
    从矩阵 H 中删除指定列，然后删除所有全 0 行。

    参数:
        H: 输入矩阵（numpy.ndarray 或 GF(2) 矩阵）
        cols_to_remove: 要删除的列索引（list 或 numpy array）

    返回:
        - 处理后的矩阵 H_new
        - 删除的行索引（全 0 行的原始行号）
    """
    # 删除指定列
    H_new = np.delete(H, cols_to_remove, axis=1)

    # 查找全 0 行
    row_sums = np.sum(H_new, axis=1)
    zero_row_indices = np.where(row_sums == 0)[0]

    # 删除这些全 0 行
    H_new = np.delete(H_new, zero_row_indices, axis=0)

    return H_new, zero_row_indices
def process_L(Hx174_sortedall, L):
    A, A1, B, B1 = [], [], [], []

    H = GF(Hx174_sortedall)  # convert H to GF(2)
    rank_H = gf2_rank_galois(H)

    for i in range(L.shape[0]):
        Li = L[i]
        H_aug = np.vstack([H, Li])
        new_rank = gf2_rank_galois(H_aug)

        if new_rank > rank_H:
            H = H_aug
            rank_H = new_rank
            B.append(Li)
            B1.append(i)
        else:
            # Solve H.T x = Li.T over GF(2)
            x = solve_gf2(H.T, Li)
            x = np.array(x, dtype=int)

            # 若解中存在对“新加行”索引（超出原 H 的行数）的依赖：
            for j in range(Hx174_sortedall.shape[0], H.shape[0]):
                if x[j] == 1:
                    Li = Li = Li + H[j]  # ✅ 全部在 GF(2) 上，galois 自动处理模2加法
  # Li -= H[j] in GF(2)
            A.append(Li)
            A1.append(i)

    return A, A1, B, B1
def construct_Jx(B, lx17_d, Hx):
    """
    构造 GF(2) 上行满秩矩阵 Jx。初始行为 B，后续从 lx17_d 中选取，
    只要 rank([Jx; Hx; row]) > rank([Jx; Hx]) 就将该行加入 Jx。
    
    参数:
        B (list of list): 初始矩阵，0-1构成的 GF(2) 行向量
        lx17_d (np.ndarray): 候选行矩阵，元素为0或1
        Hx (np.ndarray): 固定稳定子矩阵 Hx
        
    返回:
        Jx (np.ndarray): 构造后的 GF(2) 行满秩矩阵
        selected_indices (list): 从 lx17_d 中选入的行索引
    """
    # GF = GF(2)

    B_array = GF(np.array(B, dtype=int))
    lx17_d_GF = GF(lx17_d)
    Hx_GF = GF(Hx)

    Jx_rows = [row for row in B_array]
    selected_indices = []

    # 初始 [Jx; Hx] 矩阵
    current_matrix = np.vstack([B_array, Hx_GF])
    current_rank = gf2_rank_galois(current_matrix.view(np.ndarray))

    for i, row in enumerate(lx17_d_GF):
        test_matrix = np.vstack([current_matrix, row])
        test_rank =gf2_rank_galois(test_matrix.view(np.ndarray))

        if test_rank > current_rank:
            Jx_rows.append(row)
            selected_indices.append(i)
            current_matrix = test_matrix
            current_rank = test_rank  # 更新秩

        # if len(Jx_rows) >= lx17_d.shape[0]:
        #     break

    Jx = np.vstack(Jx_rows)
    return np.array(Jx, dtype=int), selected_indices,lx17_d[selected_indices, :]
def distance_test1(stab, logicOp):
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
    model += pulp.lpSum(x[i] for i in range(n))

    for row in range(m):
        weight = [0] * num_var
        for q in np.nonzero(stab[row])[0]:
            weight[q] = 1
        cnt = 1
        for q in range(num_anc_stab):
            weight[n + row * num_anc_stab + q] = -(1 << cnt)
            cnt += 1
        model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 0

    weight = [0] * num_var
    for q in np.nonzero(logicOp)[0]:
        weight[q] = 1
    cnt = 1
    for q in range(num_anc_logical):
        weight[n + m * num_anc_stab + q] = -(1 << cnt)
        cnt += 1
    model += pulp.lpSum(weight[i] * x[i] for i in range(num_var)) == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    logical_op = np.array([int(round(pulp.value(x[i]))) for i in range(n)])
    opt_val = int(np.sum(logical_op))

    return opt_val, logical_op
def find_and_remove_w(W, op):
    """
    从二维数组 W 中找到一行向量 w，使得 w @ op % 2 != 0，并将该行从 W 中删除。

    参数:
        W (np.ndarray): 形状为 (n, m) 的二维数组，表示 n 个候选向量
        op (np.ndarray): 形状为 (m,) 的一维数组，目标向量

    返回:
        w (np.ndarray): 满足条件的行向量，或 shape=(0, m) 的空数组
        W_new (np.ndarray): 删除该向量后的 W
    """
    op = np.array(op).flatten() % 2
    W = W % 2  # 保证在 GF(2)

    for j in range(W.shape[0]):
        if np.dot(W[j], op) % 2 != 0:
            w = W[j].copy()
            W_new = np.delete(W, j, axis=0)
            return w, W_new

    # 若无满足条件的行，返回空行向量
    w = np.empty((0, W.shape[1]), dtype=int)
    return w, W
def gf2_add(a, b):
    """
    GF(2) 上的向量加法（按位异或）

    参数:
        a, b: numpy.ndarray 类型，shape 相同，元素为 0 或 1

    返回:
        numpy.ndarray: GF(2) 上 a + b 的结果
    """
    return (np.array(a) ^ np.array(b)) % 2
def loop_update_U_W(U, V, Hz174_cleaned, dt):
    """
    对 U 的每一行进行更新，直到其在 Hz174_cleaned 下的 distance 大于等于 dt。
    每次如果发现 distance 不足，通过从 W 中找可修正向量 w 来更新。

    参数：
        U: (k, n) numpy.ndarray，待更新的逻辑算符矩阵
        V: (m, n) numpy.ndarray，初始 W
        Hz174_cleaned: (r, n) numpy.ndarray，Z stabilizer check matrix
        dt: int，目标距离阈值

    返回：
        U_new: 更新后的 U
        W_new: 更新后的 W
    """
    W = V.copy()
    U = U.copy()

    for i in range(U.shape[0]):
        W = V.copy()
        best_Ui = U[i, :].copy()
        best_d = 0

        d_u, e = distance_test1(Hz174_cleaned, U[i, :])
        if d_u > best_d:
            best_d = d_u
            best_Ui = U[i, :].copy()

        while d_u < dt:
            if np.sum(W @ e % 2) > 0:
                w, W = find_and_remove_w(W, e)
                U[i, :] = gf2_add(w, U[i, :])
                for j in range(W.shape[0]):
                    if np.dot(W[j], e) % 2 != 0:
                        W[j, :] = gf2_add(w, W[j, :])
                d_u, e = distance_test1(Hz174_cleaned, U[i, :])
                if d_u > best_d:
                    best_d = d_u
                    best_Ui = U[i, :].copy()
            else:
                break
        U[i, :] = best_Ui


    return U
def classify_and_sort_edges(edge_map):
    """
    对边进行排序：
    - 首先按照 y 从小到大排序；
    - 对于相同 y，按照 x 从小到大；
    - 然后在每个 (y, x) 层次中，交错排列横边与竖边（即顺序为横边、竖边、横边…）
    """
    from collections import defaultdict

    # 先将所有边按 (y, x) 进行分组（即边的起始位置）
    position_buckets = defaultdict(list)

    for idx, (p1, p2) in edge_map.items():
        x1, y1 = p1
        x2, y2 = p2
        x_min, y_min = min(x1, x2), min(y1, y2)

        if y1 == y2:  # 横边
            edge_type = 0
        elif x1 == x2:  # 竖边
            edge_type = 1
        else:
            raise ValueError(f"非法边: {p1} - {p2}")

        # 分桶按 y_min 排，bucket key 为 (y_min, x_min)
        position_buckets[(y_min, x_min)].append((edge_type, idx))

    # 对 key 排序：y 从小到大，x 从小到大
    sorted_keys = sorted(position_buckets.keys())

    sorted_indices = []
    for key in sorted_keys:
        edges = position_buckets[key]
        # 分成横边、竖边
        hor_edges = [idx for etype, idx in edges if etype == 0]
        ver_edges = [idx for etype, idx in edges if etype == 1]

        # 交错排列
        for i in range(max(len(hor_edges), len(ver_edges))):
            if i < len(hor_edges):
                sorted_indices.append(hor_edges[i])
            if i < len(ver_edges):
                sorted_indices.append(ver_edges[i])

    return sorted_indices

def reorder_matrix_by_edges(H, edge_map):
    sorted_indices = classify_and_sort_edges(edge_map)
    H_sorted = H[:, sorted_indices]
    new_edge_map = {new_idx: edge_map[old_idx] for new_idx, old_idx in enumerate(sorted_indices)}
    return H_sorted, new_edge_map, sorted_indices
def sort_rows_by_position(matrix, xposition): 
    """
    按照点的坐标（先y升序，再x升序）对行进行排序
    """
    # 每个元素：(y, x, original_index)
    indexed_positions = [(y, x, i) for i, (x, y) in enumerate(xposition)]
    # 按行（y）从小到大，再按列（x）从小到大排序
    indexed_positions.sort(key=lambda tup: (tup[0], tup[1]))

    sorted_indices = [idx for _, _, idx in indexed_positions]
    sorted_matrix = matrix[sorted_indices, :]
    sorted_xposition = [xposition[idx] for idx in sorted_indices]

    return sorted_matrix, sorted_xposition, sorted_indices
def process_Z(leng):
    filename = f"./data/hxhz162k0_7_{leng}.pkl"
    if not os.path.exists(filename):
        print(f"[SKIP] File not found: {filename}")
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        xposition174 = data["xposition"]
        zposition174 = data["zposition"]
        edge_map174 = data["edge_map"]
        Hx174 = data["Hx"]
        Hz174 = data["Hz"]
        Hz54_sorted, edge_map54_sorted, idx_order54 = reorder_matrix_by_edges(Hz54, edge_map54)
        Hz174_sorted, edge_map174_sorted, idx_order174 = reorder_matrix_by_edges(Hz174, edge_map174)
        Hx54_sorted, edge_map54_sortedx, idx_order54x = reorder_matrix_by_edges(Hx54, edge_map54)
        Hx174_sorted, edge_map174_sortedx, idx_order174x = reorder_matrix_by_edges(Hx174, edge_map174)
        Hz174_sortedall, zposition174_sorted, idx174 = sort_rows_by_position(Hz174_sorted, zposition174)
        Hz54_sortedall, zposition54_sorted, idx54 = sort_rows_by_position(Hz54_sorted, zposition54)
        Hx174_sortedall, xposition174_sorted, idx174 = sort_rows_by_position(Hx174_sorted, xposition174)
        Hx54_sortedall, xposition54_sorted, idx54 = sort_rows_by_position(Hx54_sorted, xposition54)
        qcode174=css_code(Hx174_sortedall,Hz174_sortedall)
        print('Testing CSS code...')
        qcode174.test()
        print('Done')
        lz5_padded = pad_matrix_right_gf2(Jzz, Hz174_sortedall.shape[1])
        solutions = solve_gf2_linear_system(lz5_padded, Hz174_sortedall)
        threshold = Hz54_sortedall.shape[0]-1  # 获取阈值
        indices = get_zero_indices_exceeding_threshold(solutions[mlx], threshold)
        Hz_new, removed_cols = remove_rows_and_zero_columns(Hz174_sortedall, indices)
        lz5_padded_updated = remove_columns_from_matrix(lz5_padded, removed_cols)
        Hx174_cleaned, removed_row_indices = remove_columns_and_zero_rows(Hx174_sortedall, removed_cols)
        qcode174_d=css_code(Hx174_cleaned,Hz_new)
        print('Testing CSS code...')
        qcode174_d.test()
        print('Done')
        lz174_d = qcode174_d.lz
        lx174_d = qcode174_d.lx
        k_d = lz174_d.shape[0]
        lz17_d=lz174_d.toarray()
        lx17_d=lx174_d.toarray()
        A, A1, B, B1 = process_L(Hz_new, lz5_padded_updated)
        Jz, indices,C = construct_Jx(B, lz17_d, Hz_new)
        GT=gf2_inverse_product(Jz, lx17_d)
        Jx=gf2_matmul(GT.T, lx17_d)
        U = Jx[:len(B), :]
        V=Jx[len(B):, :]
        print("Before opt dx:")
        d0 = Hx174_cleaned.shape[1]
        for i in range(U.shape[0]):
            w = distance_test(Hx174_cleaned,U[i,:])
            print('Logical qubit=',i,'Distance=',w)
            d0 = min(d0,w)
        print("dz:")
        dz = Hz_new.shape[1]
        for i in range(len(B)):
            w = distance_test(Hz_new,B[i])
            print('Logical qubit=',i,'Distance=',w)
            dz = min(dz,w)
        print('Code parameters: n,k,d=',Hx174_cleaned.shape[1],k_d,min(d0,dz))
        print("优化后大码距U：")
        U0=U
        U1=loop_update_U_W(U0, V, Hx174_cleaned, dt=dt0)
        d = Hx174_cleaned.shape[1]
        for i in range(U1.shape[0]):
            w = distance_test(Hx174_cleaned,U1[i,:])
            print('Logical qubit=',i,'Distance=',w)
            d = min(d,w)

        print('Code parameters: n,k,d=',Hx174_sortedall.shape[1],k_d,min(d,dz))
        return (leng, Hx174_sortedall.shape[1],d0,dz,d)

    except Exception as e:
        print(f"[ERROR] leng={leng} => {e}")
        return None
def process_ZZ(leng,w1,w2):
    filename = f"./data/hxhz162k0_7_{leng}.pkl"
    if not os.path.exists(filename):
        print(f"[SKIP] File not found: {filename}")
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        xposition174 = data["xposition"]
        zposition174 = data["zposition"]
        edge_map174 = data["edge_map"]
        Hx174 = data["Hx"]
        Hz174 = data["Hz"]
        Hz54_sorted, edge_map54_sorted, idx_order54 = reorder_matrix_by_edges(Hz54, edge_map54)
        Hz174_sorted, edge_map174_sorted, idx_order174 = reorder_matrix_by_edges(Hz174, edge_map174)
        Hx54_sorted, edge_map54_sortedx, idx_order54x = reorder_matrix_by_edges(Hx54, edge_map54)
        Hx174_sorted, edge_map174_sortedx, idx_order174x = reorder_matrix_by_edges(Hx174, edge_map174)
        Hz174_sortedall, zposition174_sorted, idx174 = sort_rows_by_position(Hz174_sorted, zposition174)
        Hz54_sortedall, zposition54_sorted, idx54 = sort_rows_by_position(Hz54_sorted, zposition54)
        Hx174_sortedall, xposition174_sorted, idx174 = sort_rows_by_position(Hx174_sorted, xposition174)
        Hx54_sortedall, xposition54_sorted, idx54 = sort_rows_by_position(Hx54_sorted, xposition54)
        lz51, removed_idx = insert_and_maintain_rank(Jzz, w1,w2)
        qcode174=css_code(Hx174_sortedall,Hz174_sortedall)
        print('Testing CSS code...')
        qcode174.test()
        print('Done')
        lz5_padded = pad_matrix_right_gf2(lz51, Hz174_sortedall.shape[1])
        solutions = solve_gf2_linear_system(lz5_padded, Hz174_sortedall)
        threshold = Hz54_sortedall.shape[0]-1  # 获取阈值
        indices = get_zero_indices_exceeding_threshold(solutions[0], threshold)
        Hz_new, removed_cols = remove_rows_and_zero_columns(Hz174_sortedall, indices)
        lz5_padded_updated = remove_columns_from_matrix(lz5_padded, removed_cols)
        Hx174_cleaned, removed_row_indices = remove_columns_and_zero_rows(Hx174_sortedall, removed_cols)
        qcode174_d=css_code(Hx174_cleaned,Hz_new)
        print('Testing CSS code...')
        qcode174_d.test()
        print('Done')
        lz174_d = qcode174_d.lz
        lx174_d = qcode174_d.lx
        k_d = lz174_d.shape[0]
        lz17_d=lz174_d.toarray()
        lx17_d=lx174_d.toarray()
        A, A1, B, B1 = process_L(Hz_new, lz5_padded_updated)
        Jz, indices,C = construct_Jx(B, lz17_d, Hz_new)
        GT=gf2_inverse_product(Jz, lx17_d)
        Jx=gf2_matmul(GT.T, lx17_d)
        U = Jx[:len(B), :]
        V=Jx[len(B):, :]
        print("Before opt dx:")
        d0 = Hx174_cleaned.shape[1]
        for i in range(U.shape[0]):
            w = distance_test(Hx174_cleaned,U[i,:])
            print('Logical qubit=',i,'Distance=',w)
            d0 = min(d0,w)
        print("dz:")
        dz = Hz_new.shape[1]
        for i in range(len(B)):
            w = distance_test(Hz_new,B[i])
            print('Logical qubit=',i,'Distance=',w)
            dz = min(dz,w)
        print('Code parameters: n,k,d=',Hx174_cleaned.shape[1],k_d,min(d0,dz))
        print("After opt dx:")
        U0=U
        U1=loop_update_U_W(U0, V, Hx174_cleaned, dt=dt0)
        d = Hx174_cleaned.shape[1]
        for i in range(U1.shape[0]):
            w = distance_test(Hx174_cleaned,U1[i,:])
            print('Logical qubit=',i,'Distance=',w)
            d = min(d,w)

        print('Code parameters: n,k,d=',Hx174_sortedall.shape[1],k_d,min(d,dz))
        return (leng, Hx174_sortedall.shape[1],d0,dz,d)

    except Exception as e:
        print(f"[ERROR] leng={leng} => {e}")
        return None
if __name__ == "__main__":
    mlx=0
    m1lx = np.array([1, 2, 3, 4, 5, 6, 7])
    if mlx in m1lx:
        print("Zi and Zj must be different!!!")
        sys.exit()
    dt0=7
    with open("./data/hxhz_7_7.pkl", "rb") as f:
        data = pickle.load(f)
    xposition54 = data["xposition"]
    zposition54 = data["zposition"]
    edge_map54 = data["edge_map"]
    Hx54 = data["Hx"]
    Hz54 = data["Hz"]
    Jxz = np.loadtxt("./data/Jxz.txt", dtype=int) 
    Jzz = np.loadtxt("./data/Jzz.txt", dtype=int)
    resultsz = []
    for leng in range(9, 16):
        resultz = process_Z(leng)
        if resultz:
            resultsz.append(resultz)
            _, _, _,  _, d = resultz
            if d == dt0:
                print(f"已达到目标 dx = {dt0}，提前退出循环。")
                break
    #calculate ZZ
    results_zz = {}
    for i in range(len(m1lx)):
        results = []
        for leng in range(9,16):
            result = process_ZZ(leng,mlx,m1lx[i])
            if result:
                results.append(result)
                _, _, _,  _, d = result
                if d == dt0:
                    print(f"已达到目标 dx = {dt0}，提前退出循环。")
                    break
        results_zz[i] = results
    print("\n========= Summary Z =========")
    print(f"Z{mlx}")
    print("leng\tn\tdz\t优化前dx\t优化后dx ")
    for wid, n,  d0,dz,d in resultsz:
        print(f"{wid}\t{n}\t{dz}\t{d0}\t{d}")
    print("\n========= Summary ZZ =========")
    for i in range(len(results_zz)):
        print(f"Z{mlx} Z{m1lx[i]}")
        print("leng\tn\tdz\t优化前dx\t优化后dx ")
        for wid, n,  d0,dz,d in results_zz[i]:
            print(f"{wid}\t{n}\t{dz}\t{d0}\t{d}")