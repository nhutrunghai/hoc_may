# Student Score Prediction

Dự án dự đoán điểm cuối kỳ `G3` của học sinh/sinh viên từ bộ dữ liệu `student-por-v1.csv`.

Dự án huấn luyện và so sánh 4 mô hình hồi quy:

- Linear Regression
- Ridge Regression, có tìm `alpha` tốt nhất bằng `GridSearchCV`
- Decision Tree Regressor
- Random Forest Regressor

## Cấu trúc thư mục

```text
student_score_prediction/
|-- data/
|   |-- raw/
|   |   `-- student-por-v1.csv
|   `-- processed/
|-- reports/
|   `-- figures/
|-- src/
|   |-- data_loader.py
|   |-- models.py
|   `-- visualize_models.py
|-- requirements.txt
`-- README.md
```

## Cài thư viện

```powershell
python -m pip install -r requirements.txt
```

File `requirements.txt` gồm các thư viện chính:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Chạy dự án

Chạy file chính:

```powershell
python src\models.py
```

Khi chạy xong, chương trình sẽ:

- Đọc dữ liệu từ `data/raw/student-por-v1.csv`
- Mã hóa dữ liệu chữ thành số bằng one-hot encoding
- Chia dữ liệu thành tập train/test
- Huấn luyện 4 mô hình hồi quy
- So sánh các mô hình bằng `MAE`, `MSE`, `RMSE`, `R-squared`
- Lưu kết quả đánh giá vào thư mục `reports/`
- Tạo các biểu đồ trong `reports/figures/`

## Kết quả đầu ra

Sau khi chạy mô hình, các file chính được tạo ra gồm:

- `data/processed/student-por-v1-processed.csv`: dữ liệu sau khi xử lý và one-hot encoding
- `reports/model_comparison.csv`: bảng so sánh kết quả các mô hình
- `reports/correlation_with_G3.csv`: tương quan giữa các biến và điểm `G3`
- `reports/random_forest_feature_importance.csv`: độ quan trọng của các biến theo Random Forest
- `reports/figures/compare.png`: biểu đồ so sánh các chỉ số đánh giá
- `reports/figures/pred.png`: biểu đồ giá trị thật và giá trị dự đoán của mô hình tốt nhất
- `reports/figures/rf.png`: biểu đồ feature importance của Random Forest
- `reports/figures/tree.png`: hình minh họa cây quyết định
- `reports/figures/linear.png`: hệ số của Linear Regression
- `reports/figures/ridge.png`: hệ số của Ridge Regression
