import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from data_loader import (
    RANDOM_STATE,
    TARGET_COLUMN,
    correlation_with_target,
    load_data,
    missing_value_report,
    save_processed_dataset,
    split_features_target,
    train_test_split_data,
)

# Đường dẫn chính của dự án
DATA_PATH = "data/raw/student-por-v1.csv"
PROCESSED_PATH = "data/processed/student-por-v1-processed.csv"
REPORTS = "reports"
FIGURES = "reports/figures"


# Bước 5: Khai báo 4 mô hình cần so sánh
def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": GridSearchCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

# Bước 6: Đánh giá mô hình bằng MAE, RMSE và R-squared
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": r2_score(y_test, y_pred),
    }


# Hàm chính: đọc dữ liệu, xử lý, train và đánh giá các mô hình
def train_all_models(return_models=False):
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    # Bước 1: Đọc dữ liệu từ file CSV
    df = load_data(DATA_PATH)
    print("Kich thuoc du lieu:", df.shape)
    print("\nGia tri thieu theo cot:")
    print(missing_value_report(df))

    # Bước 2: Mã hóa cột chữ thành cột số và lưu ra file processed
    encoded_df = save_processed_dataset(df, PROCESSED_PATH)

    # Bước 3: Xem các cột nào tương quan mạnh với điểm G3
    corr = correlation_with_target(df)
    corr.to_csv(f"{REPORTS}/correlation_with_G3.csv", header=["correlation"])
    print("\nTop 15 bien tuong quan manh nhat voi G3:")
    print(corr.head(15))

    # Bước 4: Tách X là dữ liệu đầu vào, y là điểm cần dự đoán
    X, y = split_features_target(encoded_df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    feature_names = X_train.columns.tolist()
    models = build_models()

    # Bước 7: Lần lượt train từng mô hình và lưu kết quả đánh giá
    results = []
    fitted_models = {}
    ridge_best_alpha = None
    for model_name, model in models.items():
        print(f"\nDang huan luyen: {model_name}")
        model.fit(X_train, y_train)

        best_alpha = None
        fitted_model = model
        if isinstance(model, GridSearchCV):
            best_alpha = model.best_params_["alpha"]
            ridge_best_alpha = best_alpha
            fitted_model = model.best_estimator_

        fitted_models[model_name] = fitted_model

        metrics = evaluate_model(model, X_test, y_test)
        results.append({"Model": model_name, **metrics})

    # Bước 8: Sắp xếp kết quả theo RMSE tăng dần để tìm mô hình tốt nhất
    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    results_df.to_csv(f"{REPORTS}/model_comparison.csv", index=False)

    best_model_name = results_df.loc[0, "Model"]
    print("\nBang so sanh mo hinh tren tap Test:")
    print(results_df.to_string(index=False))
    print(f"\nMo hinh tot nhat theo RMSE: {best_model_name}")
    if ridge_best_alpha is not None:
        print(f"Best alpha cua Ridge Regression: {ridge_best_alpha}")
    if return_models:
        return results_df, fitted_models, X_test, y_test, best_model_name, feature_names
    return results_df


def main():
    _, fitted_models, X_test, y_test, best_model_name, feature_names = train_all_models(return_models=True)
    from visualize_models import generate_visualizations

    generate_visualizations(fitted_models, X_test, y_test, best_model_name, feature_names)

if __name__ == "__main__":
    main()
