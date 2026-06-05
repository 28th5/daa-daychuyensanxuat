"""
experiment.py — Chạy thực nghiệm so sánh các thuật toán Flow Shop.

Quy trình đánh giá:
  - Chạy mọi thuật toán trên toàn bộ benchmark Taillard (20x5 và 20x10).
  - Thuật toán ngẫu nhiên (SA, Tabu, GA, Random) chạy NHIỀU seed rồi lấy
    trung bình + tốt nhất, để kết quả đáng tin (giảm may rủi).
  - Thước đo chính: RPD (Relative Percentage Deviation) so với best-known:
        RPD = 100 * (Cmax - best_known) / best_known
  - Xuất:
        results/summary_per_instance.csv  — chi tiết từng instance
        results/summary_by_algorithm.csv  — tổng hợp theo thuật toán
        results/convergence.png           — đường hội tụ
        results/avg_gap.png               — gap trung bình theo thuật toán
        results/gantt_ta001.png           — biểu đồ Gantt lời giải tốt nhất ta001
"""

from __future__ import annotations

import csv
import os
import statistics
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # không cần GUI
import matplotlib.pyplot as plt

from instances import all_benchmark, taillard_20x5
from flowshop import completion_matrix
from algorithms import (
    random_search, neh, neh_local_search,
    simulated_annealing, tabu_search, genetic_algorithm,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MAX_EVAL = 20000          # ngân sách đánh giá makespan cho metaheuristic
N_SEEDS = 3               # số lần chạy lặp cho thuật toán ngẫu nhiên

# Thuật toán tất định (chạy 1 lần) và ngẫu nhiên (chạy nhiều seed)
DETERMINISTIC = {
    "NEH": lambda inst, seed: neh(inst),
    "NEH+LS": lambda inst, seed: neh_local_search(inst),
}
STOCHASTIC = {
    "Random": lambda inst, seed: random_search(inst, MAX_EVAL, seed),
    "SA": lambda inst, seed: simulated_annealing(inst, MAX_EVAL, seed),
    "Tabu": lambda inst, seed: tabu_search(inst, MAX_EVAL, seed),
    "GA": lambda inst, seed: genetic_algorithm(inst, MAX_EVAL, seed),
}

# Thứ tự hiển thị
ALGO_ORDER = ["Random", "NEH", "NEH+LS", "SA", "Tabu", "GA"]


def rpd(cmax, best_known):
    return 100.0 * (cmax - best_known) / best_known


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances = all_benchmark()
    per_instance_rows = []
    # gom RPD theo thuật toán để tính trung bình toàn cục
    agg = {a: {"rpd_mean": [], "rpd_best": [], "runtime": []} for a in ALGO_ORDER}

    print(f"Chạy {len(instances)} instance, ngân sách {MAX_EVAL} evals, "
          f"{N_SEEDS} seed/thuật toán ngẫu nhiên\n")

    for inst in instances:
        bk = inst.best_known
        row = {"instance": inst.name, "size": f"{inst.n_jobs}x{inst.n_machines}",
               "best_known": bk}
        print(f"[{inst.name}] {inst.n_jobs}x{inst.n_machines} best_known={bk}")

        for algo in ALGO_ORDER:
            if algo in DETERMINISTIC:
                r = DETERMINISTIC[algo](inst, 0)
                cmaxes = [r.makespan]
                runtimes = [r.runtime]
            else:
                cmaxes, runtimes = [], []
                for s in range(N_SEEDS):
                    r = STOCHASTIC[algo](inst, s)
                    cmaxes.append(r.makespan)
                    runtimes.append(r.runtime)

            c_best = min(cmaxes)
            c_mean = statistics.mean(cmaxes)
            rt = statistics.mean(runtimes)

            row[f"{algo}_best"] = c_best
            row[f"{algo}_mean"] = round(c_mean, 1)
            row[f"{algo}_rpd_best"] = round(rpd(c_best, bk), 2)
            row[f"{algo}_rpd_mean"] = round(rpd(c_mean, bk), 2)

            agg[algo]["rpd_best"].append(rpd(c_best, bk))
            agg[algo]["rpd_mean"].append(rpd(c_mean, bk))
            agg[algo]["runtime"].append(rt)

            print(f"    {algo:8s} best={c_best:5d} ({rpd(c_best,bk):5.2f}%)  "
                  f"mean={c_mean:7.1f} ({rpd(c_mean,bk):5.2f}%)  t={rt:.3f}s")
        per_instance_rows.append(row)
        print()

    _write_per_instance_csv(per_instance_rows)
    by_algo = _write_by_algorithm_csv(agg)
    _print_final_table(by_algo)
    return instances, per_instance_rows, by_algo


def _write_per_instance_csv(rows):
    path = os.path.join(RESULTS_DIR, "summary_per_instance.csv")
    fields = ["instance", "size", "best_known"]
    for a in ALGO_ORDER:
        fields += [f"{a}_best", f"{a}_mean", f"{a}_rpd_best", f"{a}_rpd_mean"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"-> Đã ghi {path}")


def _write_by_algorithm_csv(agg):
    path = os.path.join(RESULTS_DIR, "summary_by_algorithm.csv")
    out = []
    for a in ALGO_ORDER:
        out.append({
            "algorithm": a,
            "avg_rpd_best": round(statistics.mean(agg[a]["rpd_best"]), 3),
            "avg_rpd_mean": round(statistics.mean(agg[a]["rpd_mean"]), 3),
            "max_rpd_mean": round(max(agg[a]["rpd_mean"]), 3),
            "avg_runtime_s": round(statistics.mean(agg[a]["runtime"]), 3),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"-> Đã ghi {path}")
    return out


def _print_final_table(by_algo):
    print("\n" + "=" * 64)
    print("TỔNG HỢP TOÀN BỘ BENCHMARK (RPD = % trên best-known, càng nhỏ càng tốt)")
    print("=" * 64)
    print(f"{'Thuật toán':10s} {'RPD_best':>10s} {'RPD_mean':>10s} "
          f"{'RPD_max':>9s} {'t(s)':>8s}")
    for r in by_algo:
        print(f"{r['algorithm']:10s} {r['avg_rpd_best']:>9.3f}% "
              f"{r['avg_rpd_mean']:>9.3f}% {r['max_rpd_mean']:>8.2f}% "
              f"{r['avg_runtime_s']:>8.3f}")
    print("=" * 64)


# --------------------------------------------------------------------------
# Biểu đồ
# --------------------------------------------------------------------------
def plot_convergence(instance_name="ta011"):
    """Vẽ đường hội tụ của SA, Tabu, GA, Random trên một instance khó (20x10)."""
    from instances import all_benchmark
    inst = next(i for i in all_benchmark() if i.name == instance_name)
    plt.figure(figsize=(8, 5))
    for fn, nm, color in [
        (lambda: random_search(inst, MAX_EVAL, 0), "Random", "gray"),
        (lambda: simulated_annealing(inst, MAX_EVAL, 0), "SA", "tab:blue"),
        (lambda: tabu_search(inst, MAX_EVAL, 0), "Tabu", "tab:green"),
        (lambda: genetic_algorithm(inst, MAX_EVAL, 0), "GA", "tab:red"),
    ]:
        r = fn()
        # Trục x = số eval thực: các thuật toán ghi lịch sử ở nhịp khác nhau
        # (SA/GA ghi mỗi eval, Tabu ghi mỗi vòng lặp), nên quy về cùng thang evals.
        xs = np.linspace(0, r.n_eval, len(r.history))
        plt.plot(xs, r.history, label=nm, color=color, lw=1.3)
    if inst.best_known:
        plt.axhline(inst.best_known, ls="--", color="black", lw=1,
                    label=f"best-known ({inst.best_known})")
    plt.xlabel("Số lần đánh giá makespan")
    plt.ylabel("Makespan tốt nhất tìm được")
    plt.title(f"Đường hội tụ trên {instance_name} ({inst.n_jobs}x{inst.n_machines})")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(RESULTS_DIR, "convergence.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"-> Đã ghi {path}")


def plot_avg_gap(by_algo):
    """Biểu đồ cột gap trung bình theo thuật toán."""
    names = [r["algorithm"] for r in by_algo]
    gaps = [r["avg_rpd_mean"] for r in by_algo]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, gaps, color="tab:blue")
    for b, g in zip(bars, gaps):
        plt.text(b.get_x() + b.get_width() / 2, g, f"{g:.2f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel("RPD trung bình (%) so với best-known")
    plt.title("So sánh chất lượng lời giải trung bình giữa các thuật toán")
    plt.grid(axis="y", alpha=0.3)
    path = os.path.join(RESULTS_DIR, "avg_gap.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"-> Đã ghi {path}")


def plot_gantt(instance_name="ta001"):
    """Vẽ Gantt cho lời giải tốt nhất tìm được (minh hoạ lịch dây chuyền)."""
    inst = next(i for i in all_benchmark() if i.name == instance_name)
    # tìm lời giải tốt bằng SA nhiều seed
    best = min((simulated_annealing(inst, MAX_EVAL, s) for s in range(N_SEEDS)),
               key=lambda r: r.makespan)
    C = completion_matrix(inst, best.perm)
    proc = inst.proc
    n, m = inst.n_jobs, inst.n_machines
    cmap = plt.cm.get_cmap("tab20", n)

    plt.figure(figsize=(11, 5))
    for k, job in enumerate(best.perm):
        for i in range(m):
            start = C[k][i] - proc[job][i]
            plt.barh(i, proc[job][i], left=start, height=0.6,
                     color=cmap(job), edgecolor="black", linewidth=0.3)
    plt.yticks(range(m), [f"Máy {i+1}" for i in range(m)])
    plt.xlabel("Thời gian")
    plt.title(f"Lịch dây chuyền {instance_name} — Cmax={best.makespan} "
              f"(best-known={inst.best_known})")
    plt.gca().invert_yaxis()
    path = os.path.join(RESULTS_DIR, f"gantt_{instance_name}.png")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"-> Đã ghi {path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    instances, rows, by_algo = run_all()
    print("\nĐang vẽ biểu đồ...")
    plot_avg_gap(by_algo)
    plot_convergence("ta011")
    plot_gantt("ta001")
    print(f"\nHoàn tất trong {time.perf_counter()-t0:.1f}s. "
          f"Kết quả trong thư mục results/")
