"""
algorithms.py — Các thuật toán giải Permutation Flow Shop (tối thiểu makespan).

Bao gồm:
  1. random_search        — baseline ngẫu nhiên (mốc so sánh dưới).
  2. neh                  — heuristic dựng hình NEH (Nawaz-Enscore-Ham, 1983).
  3. simulated_annealing  — luyện kim mô phỏng.
  4. tabu_search          — tìm kiếm Tabu.
  5. genetic_algorithm    — thuật toán di truyền (OX crossover + insert mutation).

Mọi thuật toán trả về AlgoResult(perm, makespan, history, n_eval).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from flowshop import FlowShopInstance, makespan


@dataclass
class AlgoResult:
    """Kết quả chạy một thuật toán."""
    name: str
    perm: list[int]
    makespan: int
    n_eval: int = 0                       # số lần đánh giá makespan (đo chi phí)
    runtime: float = 0.0                  # giây
    history: list[int] = field(default_factory=list)  # makespan tốt nhất theo thời gian


# --------------------------------------------------------------------------
# Tiện ích chung
# --------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def local_insertion_search(instance, perm, best_cmax, eval_counter):
    """Tìm kiếm cục bộ theo lân cận INSERTION (chèn 1 job sang vị trí khác).

    Trả về (perm_cải_thiện, cmax_mới, eval_counter). Dừng khi không cải thiện
    được trong một vòng quét đầy đủ. Đây là "vũ khí" chuẩn cho Flow Shop.
    """
    n = len(perm)
    perm = list(perm)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            job = perm[i]
            rest = perm[:i] + perm[i + 1:]
            best_pos, best_local = i, best_cmax
            best_perm_local = perm
            for pos in range(n):
                cand = rest[:pos] + [job] + rest[pos:]
                c = makespan(instance, cand)
                eval_counter[0] += 1
                if c < best_local:
                    best_local, best_pos, best_perm_local = c, pos, cand
            if best_local < best_cmax:
                perm = best_perm_local
                best_cmax = best_local
                improved = True
    return perm, best_cmax, eval_counter


# --------------------------------------------------------------------------
# 1. Random search (baseline)
# --------------------------------------------------------------------------
def random_search(instance: FlowShopInstance, max_eval=20000, seed=0) -> AlgoResult:
    """Sinh ngẫu nhiên các hoán vị, giữ lời giải tốt nhất. Mốc so sánh dưới."""
    rng = _rng(seed)
    n = instance.n_jobs
    t0 = time.perf_counter()
    best_perm = list(rng.permutation(n))
    best_cmax = makespan(instance, best_perm)
    history = [best_cmax]
    for _ in range(max_eval - 1):
        perm = list(rng.permutation(n))
        c = makespan(instance, perm)
        if c < best_cmax:
            best_perm, best_cmax = perm, c
        history.append(best_cmax)
    return AlgoResult("Random", best_perm, best_cmax, max_eval,
                      time.perf_counter() - t0, history)


# --------------------------------------------------------------------------
# 2. NEH heuristic
# --------------------------------------------------------------------------
def neh(instance: FlowShopInstance) -> AlgoResult:
    """NEH (Nawaz, Enscore, Ham 1983) — heuristic dựng hình tốt nhất cho PFSP.

    Ý tưởng:
      B1. Xếp job theo TỔNG thời gian xử lý GIẢM DẦN.
      B2. Lần lượt lấy từng job, thử CHÈN vào mọi vị trí của chuỗi đang dựng,
          giữ vị trí cho makespan nhỏ nhất.
    Cho lời giải chỉ ~3-5% trên tối ưu mà chạy gần như tức thì O(n^2 m).
    """
    t0 = time.perf_counter()
    proc = instance.proc
    n = instance.n_jobs
    eval_counter = [0]

    total_time = proc.sum(axis=1)
    order = list(np.argsort(-total_time))  # giảm dần theo tổng thời gian

    seq = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_pos, best_cmax = 0, None
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            c = makespan(instance, cand)
            eval_counter[0] += 1
            if best_cmax is None or c < best_cmax:
                best_cmax, best_pos = c, pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]

    cmax = makespan(instance, seq)
    return AlgoResult("NEH", seq, cmax, eval_counter[0],
                      time.perf_counter() - t0, [cmax])


# --------------------------------------------------------------------------
# 3. Simulated Annealing
# --------------------------------------------------------------------------
def simulated_annealing(instance, max_eval=20000, seed=0,
                        T0=None, alpha=0.97, init="neh") -> AlgoResult:
    """Luyện kim mô phỏng với lân cận INSERTION.

    - Khởi tạo bằng NEH (mặc định) để xuất phát từ điểm tốt.
    - Chấp nhận nghiệm xấu hơn với xác suất exp(-delta/T).
    - Hạ nhiệt theo cấp số nhân T <- alpha*T sau mỗi 'chu kỳ' n bước.
    """
    rng = _rng(seed)
    n = instance.n_jobs
    t0 = time.perf_counter()

    if init == "neh":
        cur = neh(instance).perm
    else:
        cur = list(rng.permutation(n))
    cur_cmax = makespan(instance, cur)
    best, best_cmax = list(cur), cur_cmax
    history = [best_cmax]
    n_eval = 1

    # Nhiệt độ ban đầu: ước lượng theo độ lệch trung bình các bước thử.
    if T0 is None:
        deltas = []
        for _ in range(50):
            i, j = rng.integers(0, n, size=2)
            cand = list(cur)
            cand[i], cand[j] = cand[j], cand[i]
            deltas.append(abs(makespan(instance, cand) - cur_cmax))
            n_eval += 1
        mean_delta = np.mean(deltas) if deltas else 1.0
        T0 = max(mean_delta / math.log(1 / 0.8), 1.0)  # accept ~80% ban đầu
    T = T0

    period = max(n, 20)
    while n_eval < max_eval:
        for _ in range(period):
            if n_eval >= max_eval:
                break
            # Lân cận: rút 1 job và chèn sang vị trí khác (insertion move)
            i = int(rng.integers(0, n))
            job = cur[i]
            rest = cur[:i] + cur[i + 1:]
            pos = int(rng.integers(0, n))
            cand = rest[:pos] + [job] + rest[pos:]
            c = makespan(instance, cand)
            n_eval += 1
            delta = c - cur_cmax
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur, cur_cmax = cand, c
                if c < best_cmax:
                    best, best_cmax = list(cand), c
            history.append(best_cmax)
        T *= alpha
        if T < 1e-3:
            T = T0 * 0.5  # reheat nhẹ để tránh đóng băng sớm

    return AlgoResult("SA", best, best_cmax, n_eval,
                      time.perf_counter() - t0, history)


# --------------------------------------------------------------------------
# 4. Tabu Search
# --------------------------------------------------------------------------
def tabu_search(instance, max_eval=20000, seed=0, tenure=None,
                init="neh") -> AlgoResult:
    """Tìm kiếm Tabu với lân cận INSERTION và danh sách cấm theo cặp (job,vị trí).

    Mỗi vòng: duyệt một mẫu lân cận, chọn nước đi tốt nhất KHÔNG bị cấm
    (trừ khi vượt qua được aspiration = cải thiện kỷ lục). Cập nhật tabu list.
    """
    rng = _rng(seed)
    n = instance.n_jobs
    t0 = time.perf_counter()
    if tenure is None:
        tenure = max(5, n // 2)

    if init == "neh":
        cur = neh(instance).perm
    else:
        cur = list(rng.permutation(n))
    cur_cmax = makespan(instance, cur)
    best, best_cmax = list(cur), cur_cmax
    history = [best_cmax]
    n_eval = 1

    tabu = {}  # (job, pos) -> vòng hết hạn
    it = 0
    # giới hạn số lân cận xét mỗi vòng để không vượt ngân sách quá nhanh
    sample = min(n * n, 200)

    while n_eval < max_eval:
        it += 1
        best_move = None
        best_move_cmax = None
        best_move_perm = None
        # Xét một mẫu các nước đi insertion
        for _ in range(sample):
            if n_eval >= max_eval:
                break
            i = int(rng.integers(0, n))
            job = cur[i]
            rest = cur[:i] + cur[i + 1:]
            pos = int(rng.integers(0, n))
            cand = rest[:pos] + [job] + rest[pos:]
            c = makespan(instance, cand)
            n_eval += 1
            move = (job, pos)
            is_tabu = tabu.get(move, 0) > it
            # aspiration: cho phép nếu phá kỷ lục
            if is_tabu and not (c < best_cmax):
                continue
            if best_move_cmax is None or c < best_move_cmax:
                best_move_cmax, best_move, best_move_perm = c, move, cand

        if best_move is None:
            # tất cả đều bị cấm — đa dạng hoá bằng một bước ngẫu nhiên
            cur = list(rng.permutation(n))
            cur_cmax = makespan(instance, cur)
            n_eval += 1
        else:
            cur, cur_cmax = best_move_perm, best_move_cmax
            tabu[best_move] = it + tenure
            if cur_cmax < best_cmax:
                best, best_cmax = list(cur), cur_cmax
        history.append(best_cmax)

    return AlgoResult("Tabu", best, best_cmax, n_eval,
                      time.perf_counter() - t0, history)


# --------------------------------------------------------------------------
# 5. Genetic Algorithm
# --------------------------------------------------------------------------
def _ox_crossover(p1, p2, rng):
    """Order Crossover (OX) — chuẩn cho mã hoá hoán vị."""
    n = len(p1)
    a, b = sorted(rng.integers(0, n, size=2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in set(p1[a:b + 1])]
    idx = 0
    for k in range(n):
        if child[k] is None:
            child[k] = fill[idx]
            idx += 1
    return child


def _insert_mutation(perm, rng):
    """Đột biến: rút một job và chèn vào vị trí ngẫu nhiên."""
    n = len(perm)
    i = int(rng.integers(0, n))
    job = perm[i]
    rest = perm[:i] + perm[i + 1:]
    pos = int(rng.integers(0, n))
    return rest[:pos] + [job] + rest[pos:]


def genetic_algorithm(instance, max_eval=20000, seed=0, pop_size=50,
                      crossover_rate=0.9, mutation_rate=0.2,
                      elite=2, seed_with_neh=True) -> AlgoResult:
    """Thuật toán di truyền cho PFSP.

    - Mã hoá: cá thể là một hoán vị job.
    - Chọn lọc: tournament size 2.
    - Lai ghép: OX. Đột biến: insertion. Giữ 'elite' cá thể tốt nhất.
    - Gieo NEH vào quần thể ban đầu để nâng chất lượng xuất phát.
    """
    rng = _rng(seed)
    n = instance.n_jobs
    t0 = time.perf_counter()
    n_eval = 0

    def evl(p):
        nonlocal n_eval
        n_eval += 1
        return makespan(instance, p)

    # Khởi tạo quần thể
    pop = [list(rng.permutation(n)) for _ in range(pop_size)]
    if seed_with_neh:
        pop[0] = neh(instance).perm
    fitness = [evl(p) for p in pop]

    best_idx = int(np.argmin(fitness))
    best, best_cmax = list(pop[best_idx]), fitness[best_idx]
    history = [best_cmax]

    def tournament():
        i, j = rng.integers(0, pop_size, size=2)
        return pop[i] if fitness[i] <= fitness[j] else pop[j]

    while n_eval < max_eval:
        # Giữ elite
        order = np.argsort(fitness)
        new_pop = [list(pop[order[k]]) for k in range(elite)]
        new_fit = [fitness[order[k]] for k in range(elite)]

        while len(new_pop) < pop_size and n_eval < max_eval:
            p1, p2 = tournament(), tournament()
            child = _ox_crossover(p1, p2, rng) if rng.random() < crossover_rate else list(p1)
            if rng.random() < mutation_rate:
                child = _insert_mutation(child, rng)
            c = evl(child)
            new_pop.append(child)
            new_fit.append(c)
            if c < best_cmax:
                best, best_cmax = list(child), c
            history.append(best_cmax)

        pop, fitness = new_pop, new_fit

    return AlgoResult("GA", best, best_cmax, n_eval,
                      time.perf_counter() - t0, history)


# --------------------------------------------------------------------------
# Biến thể "memetic": NEH + tìm kiếm cục bộ (rất mạnh, ngân sách thấp)
# --------------------------------------------------------------------------
def neh_local_search(instance, **_) -> AlgoResult:
    """NEH rồi tinh chỉnh bằng local insertion search tới hội tụ."""
    t0 = time.perf_counter()
    base = neh(instance)
    eval_counter = [base.n_eval]
    perm, cmax, eval_counter = local_insertion_search(
        instance, base.perm, base.makespan, eval_counter)
    return AlgoResult("NEH+LS", perm, cmax, eval_counter[0],
                      time.perf_counter() - t0, [base.makespan, cmax])


# Bảng tra các thuật toán dùng trong thực nghiệm
ALGORITHMS = {
    "Random": random_search,
    "NEH": neh,
    "NEH+LS": neh_local_search,
    "SA": simulated_annealing,
    "Tabu": tabu_search,
    "GA": genetic_algorithm,
}
