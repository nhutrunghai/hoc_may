# Student Score Prediction

Du an du doan diem cuoi ky `G3` cua hoc sinh/sinh vien tu bo du lieu `student-por-v1.csv`.

Du an huan luyen va so sanh 4 mo hinh hoi quy:

- Linear Regression
- Ridge Regression, co tim `alpha` tot nhat bang `GridSearchCV`
- Decision Tree Regressor
- Random Forest Regressor

## Cau truc thu muc

student_score_prediction/
|-- data/
| |-- raw/
| | `-- student-por-v1.csv
|   `-- processed/
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

## Chay huan luyen mo hinh

Chay file huan luyen:

python src\models.py

Khi chay xong, chuong trinh se:

- Doc du lieu tu `data/raw/student-por-v1.csv`
- Xu ly du lieu va ma hoa bien phan loai
- Chia du lieu thanh tap train/test
- Huan luyen 4 mo hinh hoi quy
- So sanh cac mo hinh bang `MAE`, `RMSE`, `R-squared`
- Luu ket qua danh gia vao thu muc `reports/`
- Tao cac bieu do trong `reports/figures/`

## Ket qua dau ra

Sau khi chay mo hinh, cac file chinh duoc tao ra gom:

- `data/processed/student-por-v1-processed.csv`: du lieu sau khi xu ly va one-hot encoding
- `reports/model_comparison.csv`: bang so sanh ket qua cac mo hinh
- `reports/correlation_with_G3.csv`: tuong quan giua cac bien va diem `G3`
- `reports/random_forest_feature_importance.csv`: do quan trong cua cac bien theo Random Forest
- `reports/figures/pred.png`: bieu do gia tri that va gia tri du doan cua mo hinh tot nhat
- `reports/figures/compare.png`: bieu do so sanh cac chi so danh gia
- `reports/figures/tree.png`: hinh minh hoa cay quyet dinh
- `reports/figures/linear.png`: he so cua Linear Regression
- `reports/figures/ridge.png`: he so cua Ridge Regression
- `reports/figures/rf.png`: bieu do feature importance cua Random Forest

## Quy trinh chay nhanh

```powershell
cd D:\hocmayfinal\student_score_prediction
python -m pip install -r requirements.txt
python src\models.py
````
