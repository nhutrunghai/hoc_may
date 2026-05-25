# Student Score Prediction

Du an du doan diem cuoi ky `G3` cua hoc sinh/sinh vien tu bo du lieu `student-por-10.csv`.

Du an huan luyen va so sanh 4 mo hinh hoi quy:

- Linear Regression
- Ridge Regression, co tim `alpha` tot nhat bang `GridSearchCV`
- Decision Tree Regressor
- Random Forest Regressor

## Cau truc thu muc

student_score_prediction/
|-- data/
| |-- raw/
| | `-- student-por-10.csv
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
| `-- figures/
|-- src/
|   |-- data_loader.py
|   |-- models.py
|   `-- visualize_models.py
|-- requirements.txt
`-- README.md

```

## Cai thu vien

Cai cac thu vien can thiet:


python -m pip install -r requirements.txt
```

File `requirements.txt` gom cac thu vien chinh:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`

## Chay huan luyen mo hinh

Chay file huan luyen:

python src\models.py

Khi chay xong, chuong trinh se:

- Doc du lieu tu `data/raw/student-por-10.csv`
- Xu ly du lieu va ma hoa bien phan loai
- Chia du lieu thanh tap train/test
- Huan luyen 4 mo hinh hoi quy
- So sanh cac mo hinh bang `MAE`, `RMSE`, `R-squared`
- Luu mo hinh da train vao thu muc `models/`
- Luu ket qua danh gia vao thu muc `reports/`

## Tao hinh truc quan mo hinh

Sau khi train mo hinh, chay:

python src\visualize_models.py

````

## Ket qua dau ra

Sau khi chay mo hinh, cac file chinh duoc tao ra gom:

- `data/processed/student-por-10-processed.csv`: du lieu sau khi xu ly va one-hot encoding
- `models/linear_reg.pkl`: mo hinh Linear Regression
- `models/ridge_reg.pkl`: mo hinh Ridge Regression
- `models/decision_tree.pkl`: mo hinh Decision Tree Regressor
- `models/random_forest.pkl`: mo hinh Random Forest Regressor
- `reports/model_comparison.csv`: bang so sanh ket qua cac mo hinh
- `reports/correlation_with_G3.csv`: tuong quan giua cac bien va diem `G3`
- `reports/random_forest_feature_importance.csv`: do quan trong cua cac bien theo Random Forest
- `reports/figures/actual_vs_predicted_best_model.png`: bieu do gia tri that va gia tri du doan cua mo hinh tot nhat
- `reports/figures/model_comparison_metrics.png`: bieu do so sanh cac chi so danh gia
- `reports/figures/decision_tree_visualization.png`: hinh minh hoa cay quyet dinh
- `reports/figures/linear_regression_coefficients.png`: he so cua Linear Regression
- `reports/figures/ridge_regression_coefficients.png`: he so cua Ridge Regression
- `reports/figures/random_forest_feature_importance.png`: bieu do feature importance cua Random Forest

## Quy trinh chay nhanh

```powershell
cd D:\hocmayfinal\student_score_prediction
python -m pip install -r requirements.txt
python src\models.py
python src\visualize_models.py
````
