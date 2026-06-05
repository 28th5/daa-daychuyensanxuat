"""
instances.py — Sinh dữ liệu chuẩn Taillard cho Flow Shop.

Taillard (1993) là bộ benchmark KINH ĐIỂN, được dùng rộng rãi nhất để đánh giá
thuật toán Flow Shop. Điểm hay: dữ liệu không lưu thành file lớn mà được sinh
LẠI từ một bộ sinh số ngẫu nhiên (RNG) xác định cùng các "seed thời gian" công bố.
Nhờ vậy mọi nhà nghiên cứu trên thế giới đều dùng đúng cùng một bộ dữ liệu.

Tham chiếu: E. Taillard, "Benchmarks for basic scheduling problems",
European Journal of Operational Research 64 (1993) 278-285.

Module này:
  - Cài đặt đúng RNG của Taillard.
  - Tái lập 20 instance đầu (ta001-ta020): nhóm 20x5 và 20x10.
  - Kèm makespan tối ưu / tốt nhất đã biết để làm thước đo đánh giá.
"""

from __future__ import annotations

import numpy as np
from flowshop import FlowShopInstance


# ----- Bộ sinh số ngẫu nhiên của Taillard (Lehmer / MINSTD) -----
_A = 16807
_B = 127773
_C = 2836
_M = 2147483647


def _taillard_unif(seed: int, low: int, high: int):
    """Sinh một số nguyên đều trong [low, high] theo đúng giải thuật Taillard.

    Trả về (giá_trị, seed_mới). Phải gọi tuần tự để tái lập đúng ma trận.
    """
    k = seed // _B
    seed = _A * (seed - k * _B) - k * _C
    if seed < 0:
        seed += _M
    value_0_1 = seed * (1.0 / _M)
    return low + int(value_0_1 * (high - low + 1)), seed


def generate_taillard(name: str, n_jobs: int, n_machines: int,
                      time_seed: int, best_known: int | None = None) -> FlowShopInstance:
    """Sinh một instance Taillard từ seed thời gian.

    Thời gian xử lý là số nguyên đều trong [1, 99] — đúng quy ước Taillard.
    Lưu ý thứ tự duyệt: theo MÁY ở vòng ngoài, JOB ở vòng trong (đúng bản gốc).
    """
    seed = time_seed
    proc = np.zeros((n_jobs, n_machines), dtype=np.int64)
    for i in range(n_machines):
        for j in range(n_jobs):
            val, seed = _taillard_unif(seed, 1, 99)
            proc[j][i] = val
    return FlowShopInstance(name=name, proc=proc, best_known=best_known)


# Seed thời gian và makespan tối ưu (đã chứng minh) cho nhóm 20x5 — ta001..ta010.
# Nhóm 20x5 đã được giải tối ưu hoàn toàn trong tài liệu.
_TA_20x5 = [
    ("ta001", 873654221, 1278),
    ("ta002", 379008056, 1359),
    ("ta003", 1866992158, 1081),
    ("ta004", 216771124, 1293),
    ("ta005", 495070989, 1235),
    ("ta006", 402959317, 1195),
    ("ta007", 1369363414, 1234),
    ("ta008", 2021925980, 1206),
    ("ta009", 573109518, 1230),
    ("ta010", 88325120, 1108),
]

# Nhóm 20x10 — ta011..ta020. Giá trị là tốt-nhất-đã-biết (best known upper bound).
_TA_20x10 = [
    ("ta011", 587595453, 1582),
    ("ta012", 1401007982, 1659),
    ("ta013", 873136276, 1496),
    ("ta014", 268827376, 1377),
    ("ta015", 1634173168, 1419),
    ("ta016", 691823909, 1397),
    ("ta017", 73807235, 1484),
    ("ta018", 1273398721, 1538),
    ("ta019", 2065119309, 1593),
    ("ta020", 1672900551, 1591),
]


def taillard_20x5() -> list[FlowShopInstance]:
    """10 instance 20 job x 5 máy (đã biết lời giải tối ưu)."""
    return [generate_taillard(n, 20, 5, s, bk) for (n, s, bk) in _TA_20x5]


def taillard_20x10() -> list[FlowShopInstance]:
    """10 instance 20 job x 10 máy (kèm tốt-nhất-đã-biết)."""
    return [generate_taillard(n, 20, 10, s, bk) for (n, s, bk) in _TA_20x10]


def all_benchmark() -> list[FlowShopInstance]:
    """Toàn bộ benchmark dùng trong thực nghiệm (20 instance)."""
    return taillard_20x5() + taillard_20x10()


def random_instance(n_jobs: int, n_machines: int, seed: int = 0,
                    name: str | None = None) -> FlowShopInstance:
    """Sinh instance ngẫu nhiên (numpy) — dùng để thử nghiệm kích thước tuỳ ý."""
    rng = np.random.default_rng(seed)
    proc = rng.integers(1, 100, size=(n_jobs, n_machines), dtype=np.int64)
    nm = name or f"rand_{n_jobs}x{n_machines}_s{seed}"
    return FlowShopInstance(name=nm, proc=proc, best_known=None)


if __name__ == "__main__":
    # Kiểm tra nhanh: ma trận ta001 phải khớp dữ liệu Taillard công bố.
    ta001 = taillard_20x5()[0]
    print(ta001)
    # Trong file Taillard, mỗi HÀNG là một máy. Kiểm tra theo cột máy:
    print("Máy 0, job 0-9:", ta001.proc[:10, 0].tolist())  # [54,83,15,71,77,36,53,38,27,87]
    print("Máy 1, job 0-9:", ta001.proc[:10, 1].tolist())  # [79, 3,11,99,56,70,99,60, 5,56]
