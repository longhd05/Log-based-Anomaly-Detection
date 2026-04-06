# Log-based Anomaly Detection

---

## 📋 Tổng quan

**Log-based Anomaly Detection** là hệ thống phát hiện bất thường trong log hệ thống máy tính sử dụng các kỹ thuật học máy truyền thống. Dự án triển khai và so sánh 7 thuật toán phát hiện bất thường trên tập dữ liệu **HDFS (Hadoop Distributed File System)**.

Quy trình hoạt động:
1. **Nạp dữ liệu** log HDFS (dạng `.csv` hoặc `.npz`) và nhãn bất thường.
2. **Trích xuất đặc trưng** từ chuỗi sự kiện log (Event Count Vector, TF-IDF, Zero-mean normalization).
3. **Huấn luyện mô hình** từ một trong 7 thuật toán được hỗ trợ.
4. **Đánh giá** với các chỉ số Precision, Recall, F1-score.
5. **Dự đoán** nhãn (normal / anomaly) cho dữ liệu log mới.

---

## 🏗️ Kiến trúc hệ thống

```
Log-based-Anomaly-Detection/
│
├── data/
│   └── HDFS/                          # Tập dữ liệu HDFS
│       ├── HDFS.npz                   # Toàn bộ dữ liệu (đã tiền xử lý)
│       ├── HDFS_100k.npz              # Tập con 100k log
│       ├── HDFS_115k.npz              # Tập con 115k log
│       ├── HDFS_100k.log_structured.csv  # Log thô dạng CSV
│       └── anomaly_label.csv          # Nhãn bất thường theo BlockId
│
├── loglizer/                          # Thư viện lõi
│   ├── dataloader.py                  # Nạp và phân chia dữ liệu HDFS
│   ├── preprocessing.py               # Trích xuất đặc trưng (FeatureExtractor)
│   ├── utils.py                       # Hàm tính metrics
│   └── models/                        # Các mô hình phát hiện bất thường
│       ├── PCA.py
│       ├── InvariantsMiner.py
│       ├── IsolationForest.py
│       ├── LR.py
│       ├── LogClustering.py
│       ├── SVM.py
│       └── DecisionTree.py
│
├── benchmarks/                        # Scripts chạy thực nghiệm
│   ├── HDFS_bechmark.py               # Benchmark nhanh (không lưu model)
│   ├── HDFS_benchmark_save_models.py  # Benchmark đầy đủ, lưu model .pkl
│   ├── HDFS_benchmark_time.py         # Đo thời gian huấn luyện
│   ├── benchmark_scaling.py           # Kiểm tra khả năng mở rộng
│   ├── predict.py                     # Dự đoán từ model đã lưu (CSV/NPZ)
│   └── predict_npz.py                 # Dự đoán từ file .npz
│
├── output/                            # Kết quả huấn luyện và dự đoán
│   ├── DecisionTree/                  # Model artifact (DecisionTree.pkl)
│   └── result/
│
├── utils.py                           # Tiện ích dùng chung
├── requirements.txt
└── README.md
```

**Luồng xử lý:**

```
Raw Log (CSV/NPZ)
      │
      ▼
 dataloader.load_HDFS()
      │  (session window, train/test split)
      ▼
 FeatureExtractor.fit_transform()
      │  (Event Count Vector → TF-IDF → Zero-mean)
      ▼
   Model.fit()          ←── Huấn luyện
      │
      ▼
   Model.evaluate()     ←── Precision / Recall / F1
      │
      ▼
   predict.py           ←── Dự đoán dữ liệu mới → prediction_result.csv
```

---

## 🛠️ Công nghệ sử dụng

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| Python | ≥ 3.8 | Ngôn ngữ lập trình chính |
| scikit-learn | latest | Các mô hình ML (LR, SVM, DecisionTree, IsolationForest) |
| NumPy | latest | Xử lý ma trận đặc trưng |
| Pandas | latest | Đọc/ghi dữ liệu CSV |
| SciPy | latest | Hàm sigmoid trong FeatureExtractor |
| regex | latest | Trích xuất BlockId từ log |

---

# ⚙️ Cài đặt

---

### 1. Clone repository

```bash
git clone https://github.com/longhd05/Log-based-Anomaly-Detection.git
cd Log-based-Anomaly-Detection
```

---

### 2. Tạo môi trường ảo

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Linux/macOS)
source venv/bin/activate

# Kích hoạt (Windows)
venv\Scripts\activate
```

---

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 🚀 Chạy ứng dụng

**Bước 1 – Huấn luyện tất cả mô hình và lưu artifact:**

```bash
cd benchmarks
python HDFS_benchmark_save_models.py
```

Script sẽ huấn luyện 7 mô hình trên `data/HDFS/HDFS.npz`, in kết quả Precision/Recall/F1 và lưu từng model vào `output/<ModelName>/<ModelName>.pkl`. Model tốt nhất được lưu thêm dưới tên `best_model_<ModelName>.pkl`.

**Bước 2 – Dự đoán trên dữ liệu mới:**

```bash
cd benchmarks
# Dùng file mặc định (HDFS_100k.log_structured.csv)
python predict.py

# Hoặc truyền file tùy chọn qua --file
python predict.py --file /đường/dẫn/đến/file.csv
python predict.py --file /đường/dẫn/đến/file.npz
```

Kết quả được lưu tại `benchmarks/prediction_result.csv` với các cột `BlockId`, `pred` (0/1), `label` (normal/anomaly).

**Chạy benchmark nhanh (không lưu model):**

```bash
cd benchmarks
python HDFS_bechmark.py
```

---

## 📖 Hướng dẫn sử dụng

### Huấn luyện một mô hình cụ thể

Mở file `benchmarks/HDFS_bechmark.py` và chỉnh dòng:

```python
run_models = ['DecisionTree']   # Thay bằng tên mô hình mong muốn
```

Các tên hợp lệ: `'PCA'`, `'InvariantsMiner'`, `'LogClustering'`, `'IsolationForest'`, `'LR'`, `'SVM'`, `'DecisionTree'`.

### Truyền file dữ liệu đầu vào cho `predict.py` qua dòng lệnh

Dùng option `--file` để chỉ định file log cần phân tích (hỗ trợ `.csv` và `.npz`):

```bash
cd benchmarks

# Dùng file CSV log structured
python predict.py --file ../data/HDFS/HDFS_100k.log_structured.csv

# Dùng file NPZ
python predict.py --file ../data/HDFS/HDFS_115k.npz

# Dùng đường dẫn tuyệt đối
python predict.py --file /absolute/path/to/your_log.csv
```

Nếu không truyền `--file`, script tự dùng file mặc định `data/HDFS/HDFS_100k.log_structured.csv`.

### Thay đổi nguồn dữ liệu mặc định của `predict.py`

Mở file `benchmarks/predict.py` và chỉnh biến `INPUT_FILE`:

```python
INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_100k.log_structured.csv"
# hoặc dùng file .npz:
# INPUT_FILE = ROOT / "data" / "HDFS" / "HDFS_115k.npz"
```

### Sử dụng thư viện loglizer trong code của bạn

```python
from loglizer import dataloader, preprocessing
from loglizer.models import DecisionTree

# 1. Nạp dữ liệu
(x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(
    'data/HDFS/HDFS.npz',
    train_ratio=0.8,
    split_type='uniform'
)

# 2. Trích xuất đặc trưng
fe = preprocessing.FeatureExtractor()
x_train = fe.fit_transform(x_tr, term_weighting='tf-idf')
x_test  = fe.transform(x_te)

# 3. Huấn luyện & đánh giá
model = DecisionTree()
model.fit(x_train, y_train)
model.evaluate(x_test, y_test)
```

---

## 🔧 Chi tiết các module

### `loglizer/dataloader.py`

| Hàm | Mô tả |
|---|---|
| `load_HDFS(log_file, label_file, ...)` | Nạp log HDFS từ `.csv` hoặc `.npz`, chia tập train/test theo `uniform` hoặc `sequential` |
| `slice_hdfs(x, y, window_size)` | Cắt chuỗi sự kiện thành các sliding window |

### `loglizer/preprocessing.py` – `FeatureExtractor`

| Phương thức | Mô tả |
|---|---|
| `fit_transform(X_seq, term_weighting, normalization, oov, min_count)` | Xây dựng Event Count Matrix từ tập train, tuỳ chọn áp dụng TF-IDF và Zero-mean |
| `transform(X_seq)` | Biến đổi tập test dùng tham số đã học từ train |

### `loglizer/models/`

| Mô hình | Loại | Đặc điểm |
|---|---|---|
| `PCA` | Unsupervised | Phát hiện bất thường qua SPE (Q-statistic) trong không gian PCA |
| `InvariantsMiner` | Unsupervised | Khai thác bất biến tuyến tính trong log |
| `LogClustering` | Unsupervised | Gom cụm log, coi các mẫu không thuộc cụm nào là bất thường |
| `IsolationForest` | Unsupervised | Cô lập điểm dị thường bằng cây ngẫu nhiên |
| `LR` | Supervised | Logistic Regression |
| `SVM` | Supervised | Support Vector Machine |
| `DecisionTree` | Supervised | Cây quyết định (mô hình mặc định cho `predict.py`) |

### `benchmarks/predict.py`

Script dự đoán linh hoạt, hỗ trợ cả đầu vào `.csv` (log thô dạng structured hoặc dạng sequence) và `.npz`. Tự động nạp `DecisionTree.pkl` từ `output/DecisionTree/`. Hỗ trợ option `--file <PATH>` để truyền file log từ dòng lệnh.

---

## 🌐 Frontend (Giao diện Web)

Dự án có kèm giao diện web nền tối để upload file log và xem kết quả phát hiện bất thường trực tiếp trên trình duyệt.

### Cấu trúc

```
frontend/
├── app.py                # Flask backend (API + server HTML)
├── requirements.txt      # Chỉ cần thêm flask
└── templates/
    └── index.html        # Giao diện người dùng
```

### 1. Cài đặt thêm Flask

> Yêu cầu đã cài xong `requirements.txt` chính (scikit-learn, pandas, numpy, …).

```bash
pip install -r frontend/requirements.txt
```

### 2. Chuẩn bị model

Cần có file model tại `output/DecisionTree/DecisionTree.pkl`.  
Nếu chưa có, huấn luyện trước:

```bash
python benchmarks/HDFS_benchmark_save_models.py
```

### 3. Chạy server

```bash
# Từ thư mục gốc của dự án
python frontend/app.py

# (tuỳ chọn) bật chế độ debug khi phát triển
FLASK_DEBUG=1 python frontend/app.py
```

Server khởi động tại **http://localhost:5000**.

### 4. Sử dụng

1. Mở trình duyệt, truy cập `http://localhost:5000`.
2. Kéo thả hoặc chọn file log (`.csv` hoặc `.npz`).
3. Nhấn **Bắt đầu Detect**.
4. Kết quả hiển thị:
   - Số lượng mẫu tổng / bất thường / bình thường.
   - Bảng chi tiết — hàng **bất thường bôi đỏ**, hàng bình thường hiển thị xanh.
   - Bộ lọc nhanh theo loại kết quả.

> **Lưu ý:** Với file rất lớn, bảng hiển thị tối đa 2 000 hàng đầu tiên; số liệu thống kê vẫn tính trên toàn bộ dữ liệu.

---

## 🖥️ Giao diện CLI

Dự án hoạt động hoàn toàn qua **giao diện dòng lệnh (CLI)**. Kết quả benchmark được xuất ra console và lưu vào file CSV:

```
====== Input data summary ======
Total: 575061 instances, 16838 anomaly, 558223 normal
Train: 460048 instances, 13470 anomaly, 446578 normal
Test:  115013 instances, 3368 anomaly, 111645 normal

====== Transformed train data summary ======
Train data shape: 460048-by-29

====== Model summary ======
...

====== Evaluation summary ======
Precision: 0.998, Recall: 0.997, F1-measure: 0.997
```

Kết quả dự đoán (`prediction_result.csv`):

```
BlockId,pred,label
blk_-1608999687919862906,0,normal
blk_7503483334202473044,1,anomaly
...
```

---

## 📁 Dữ liệu mẫu

| File | Kích thước | Mô tả |
|---|---|---|
| `data/HDFS/HDFS.npz` | ~toàn bộ dataset | Dữ liệu HDFS đã tiền xử lý (x_data, y_data) |
| `data/HDFS/HDFS_100k.npz` | ~100k sessions | Tập con để thực nghiệm nhanh |
| `data/HDFS/HDFS_115k.npz` | ~115k sessions | Tập con mở rộng |
| `data/HDFS/HDFS_100k.log_structured.csv` | ~100k dòng log | Log HDFS thô đã được parse, gồm các cột `LineId`, `Content`, `EventId` |
| `data/HDFS/anomaly_label.csv` | — | Nhãn bất thường theo `BlockId` (`Normal` / `Anomaly`) |

Dữ liệu nguồn: [HDFS_log_dataset](https://github.com/logpai/loghub) – bộ log của hệ thống Hadoop được thu thập từ môi trường thực tế.

---

## ⚠️ Lưu ý

- File `data/HDFS/HDFS.npz` là bắt buộc để chạy `HDFS_benchmark_save_models.py`. Nếu chưa có, hãy tải từ [loghub](https://github.com/logpai/loghub).
- Script `predict.py` mặc định tìm model tại `output/DecisionTree/DecisionTree.pkl`. Cần chạy `HDFS_benchmark_save_models.py` trước để sinh file model.
- Tỷ lệ phân chia mặc định là **80% train / 20% test** với chiến lược `uniform` (cân bằng tỉ lệ anomaly/normal trong cả hai tập).
- Các mô hình unsupervised (`PCA`, `InvariantsMiner`, `LogClustering`, `IsolationForest`) không sử dụng nhãn khi huấn luyện, riêng `LogClustering` chỉ huấn luyện trên mẫu **normal**.

---

## 📄 License

Dự án được phát triển phục vụ mục đích học tập và nghiên cứu tại **Học viện Công nghệ Bưu chính Viễn thông (PTIT)**.