<<<<<<< HEAD
# daa-daychuyensanxuat
=======
# Tối ưu dây chuyền sản xuất — Flow Shop Scheduling

Dự án tìm hiểu và **đánh giá thực nghiệm** các thuật toán cho bài toán tối ưu dây
chuyền sản xuất (Permutation Flow Shop Scheduling — tối thiểu makespan), trên bộ
benchmark chuẩn **Taillard**.

📄 **Báo cáo đầy đủ: [BAOCAO.md](BAOCAO.md)**

## Cấu trúc

```
src/
  flowshop.py     # Mô hình bài toán: makespan, completion matrix, total flow time
  instances.py    # Sinh dữ liệu chuẩn Taillard (tái lập đúng dữ liệu công bố)
  algorithms.py   # Random, NEH, NEH+LS, SA, Tabu, GA
  experiment.py   # Chạy benchmark, xuất CSV + biểu đồ
results/          # Kết quả: CSV + PNG (sinh ra sau khi chạy)
BAOCAO.md         # Báo cáo: giới thiệu, dữ liệu, thuật toán, thực nghiệm
```

## Chạy

```powershell
pip install -r requirements.txt
cd src
$env:PYTHONIOENCODING="utf-8"   # Windows: in được tiếng Việt
python instances.py             # kiểm tra bộ sinh dữ liệu
python experiment.py            # chạy toàn bộ thực nghiệm
```

## Thuật toán được đánh giá

| Thuật toán | Loại |
|---|---|
| Random Search | baseline |
| NEH | heuristic dựng hình |
| NEH + Local Search | memetic |
| Simulated Annealing | metaheuristic |
| Tabu Search | metaheuristic |
| Genetic Algorithm | metaheuristic tiến hoá |

Thước đo: **RPD** (% lệch so với lời giải tốt nhất đã biết).
>>>>>>> 515341f (first commit)
