<div align="center">

# 📊 Dự Đoán Điểm Sinh Viên

### Machine Learning project so sánh nhiều mô hình hồi quy

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

</div>

---

## 🎯 Mục tiêu

Dự án sử dụng Machine Learning để dự đoán điểm cuối kỳ `G3` từ bộ dữ liệu `student-por-v1.csv`. Project tập trung vào quy trình xử lý dữ liệu, huấn luyện mô hình, so sánh kết quả và trực quan hóa báo cáo.

---

## 🧠 Các mô hình sử dụng

| Mô hình | Vai trò |
|---|---|
| Linear Regression | Baseline model đơn giản |
| Ridge Regression | Hồi quy tuyến tính có regularization |
| Decision Tree Regressor | Mô hình cây quyết định |
| Random Forest Regressor | Ensemble model để cải thiện độ ổn định |

---

## 🔄 Pipeline xử lý

```text
📥 Đọc dữ liệu CSV
   ↓
🧹 Tiền xử lý dữ liệu
   ↓
🔢 One-hot encoding biến phân loại
   ↓
✂️ Chia train/test
   ↓
🤖 Huấn luyện 4 mô hình
   ↓
📊 So sánh MAE / MSE / RMSE / R²
   ↓
🖼️ Xuất biểu đồ và báo cáo
```

---

## 🗂️ Cấu trúc thư mục

```text
student-score-prediction/
├── data/
│   ├── raw/          # Dữ liệu gốc
│   └── processed/    # Dữ liệu sau xử lý
├── reports/
│   └── figures/      # Biểu đồ kết quả
├── src/
│   ├── data_loader.py
│   ├── models.py
│   └── visualize_models.py
├── requirements.txt
└── README.md
```

---

## 🚀 Cách chạy

### 1. Cài thư viện

```bash
python -m pip install -r requirements.txt
```

### 2. Chạy mô hình

```bash
python src/models.py
```

---

## 📦 Kết quả đầu ra

| File | Nội dung |
|---|---|
| `data/processed/student-por-v1-processed.csv` | Dữ liệu đã xử lý |
| `reports/model_comparison.csv` | Bảng so sánh mô hình |
| `reports/correlation_with_G3.csv` | Tương quan với điểm G3 |
| `reports/random_forest_feature_importance.csv` | Feature importance |
| `reports/figures/*.png` | Biểu đồ đánh giá và trực quan hóa |

---

## 🧭 Roadmap

- [ ] Thêm biểu đồ preview trực tiếp vào README
- [ ] Thêm notebook giải thích quy trình modeling
- [ ] Thử thêm mô hình Gradient Boosting/XGBoost
- [ ] Thêm phần nhận xét mô hình tốt nhất
- [ ] Chuẩn hóa báo cáo kết quả cuối cùng

---

<div align="center">

Project Machine Learning phục vụ học tập và thực hành quy trình modeling.  
Developed by [Nhữ Trung Hải](https://github.com/nhutrunghai)

</div>
