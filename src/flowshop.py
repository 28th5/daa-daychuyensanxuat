"""
flowshop.py — Mô hình bài toán Permutation Flow Shop Scheduling (PFSP).

Bài toán: có n công việc (job) phải đi qua m máy theo CÙNG một thứ tự máy
(M1 -> M2 -> ... -> Mm). Mỗi công việc j cần thời gian xử lý p[j][i] trên máy i.
Tất cả công việc được xử lý theo cùng một hoán vị (permutation) thứ tự trên mọi máy.

Mục tiêu phổ biến nhất: tối thiểu hoá MAKESPAN (Cmax) — thời điểm công việc cuối
cùng rời khỏi máy cuối cùng.

Đây là mô hình kinh điển của một dây chuyền sản xuất nối tiếp.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FlowShopInstance:
    """Một thể hiện (instance) của bài toán Flow Shop.

    Attributes:
        name: tên định danh instance (vd: 'ta001').
        proc: ma trận thời gian xử lý kích thước (n_jobs, n_machines).
              proc[j][i] = thời gian job j chạy trên máy i.
        best_known: makespan tốt nhất đã biết (nếu có), dùng để đánh giá.
    """

    name: str
    proc: np.ndarray
    best_known: int | None = None

    @property
    def n_jobs(self) -> int:
        return self.proc.shape[0]

    @property
    def n_machines(self) -> int:
        return self.proc.shape[1]

    def __repr__(self) -> str:
        bk = f", best_known={self.best_known}" if self.best_known else ""
        return f"FlowShopInstance({self.name}, {self.n_jobs}x{self.n_machines}{bk})"


def makespan(instance: FlowShopInstance, perm) -> int:
    """Tính makespan (Cmax) cho một hoán vị thứ tự công việc.

    Dùng công thức truy hồi thời điểm hoàn thành C[k][i]:
        C[0][0]   = p[perm[0]][0]
        C[k][0]   = C[k-1][0] + p[perm[k]][0]            (máy đầu)
        C[0][i]   = C[0][i-1] + p[perm[0]][i]            (job đầu)
        C[k][i]   = max(C[k-1][i], C[k][i-1]) + p[perm[k]][i]

    Độ phức tạp: O(n * m). Cài đặt vector hoá theo từng máy để nhanh.
    """
    proc = instance.proc
    m = instance.n_machines
    # completion[i] = thời điểm hoàn thành của job hiện tại trên máy i
    completion = np.zeros(m, dtype=np.int64)
    for job in perm:
        p = proc[job]
        # máy 0: cộng dồn tuần tự
        completion[0] += p[0]
        for i in range(1, m):
            completion[i] = max(completion[i], completion[i - 1]) + p[i]
    return int(completion[-1])


def completion_matrix(instance: FlowShopInstance, perm) -> np.ndarray:
    """Trả về ma trận thời điểm hoàn thành C[k][i] cho toàn bộ lịch.

    Dùng để vẽ biểu đồ Gantt và phân tích chi tiết. Kích thước (n, m).
    """
    proc = instance.proc
    n, m = instance.n_jobs, instance.n_machines
    C = np.zeros((n, m), dtype=np.int64)
    for k, job in enumerate(perm):
        p = proc[job]
        for i in range(m):
            prev_machine = C[k][i - 1] if i > 0 else 0
            prev_job = C[k - 1][i] if k > 0 else 0
            C[k][i] = max(prev_machine, prev_job) + p[i]
    return C


def total_flow_time(instance: FlowShopInstance, perm) -> int:
    """Tổng thời gian hoàn thành (sum of completion times) — mục tiêu phụ phổ biến.

    Phản ánh mức tồn kho bán thành phẩm (WIP) và thời gian chờ trung bình.
    """
    C = completion_matrix(instance, perm)
    return int(C[:, -1].sum())
