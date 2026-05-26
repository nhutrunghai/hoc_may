import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None

try:
    from .data_loader import (
        RANDOM_STATE,
        TARGET_COLUMN,
        correlation_with_target,
        load_data,
        missing_value_report,
        save_processed_dataset,
        split_features_target,
        train_test_split_data,
    )
except ImportError:
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


# Duong dan chinh cua du an
DATA_PATH = "data/raw/student-por-v1.csv"
PROCESSED_PATH = "data/processed/student-por-v1-processed.csv"
REPORTS = "reports"
FIGURES = "reports/figures"


# Buoc 5: Khai bao 4 mo hinh can so sanh
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

# Buoc 6: Danh gia mo hinh bang MAE, RMSE va R-squared
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if root_mean_squared_error is None:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    else:
        rmse = root_mean_squared_error(y_test, y_pred)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
        "R-squared": r2_score(y_test, y_pred),
    }


# Ham chinh: doc du lieu, xu ly, train va danh gia cac mo hinh
def train_all_models(return_models=False):
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    # Buoc 1: Doc du lieu tu file CSV
    df = load_data(DATA_PATH)
    print("Kich thuoc du lieu:", df.shape)
    print("\nGia tri thieu theo cot:")
    print(missing_value_report(df))

    # Buoc 2: Ma hoa cot chu thanh cot so va luu ra file processed
    encoded_df = save_processed_dataset(df, PROCESSED_PATH)

    # Buoc 3: Xem cac cot nao tuong quan manh voi diem G3
    corr = correlation_with_target(df)
    corr.to_csv(f"{REPORTS}/correlation_with_G3.csv", header=["correlation"])
    print("\nTop 15 bien tuong quan manh nhat voi G3:")
    print(corr.head(15))

    # Buoc 4: Tach X la du lieu dau vao, y la diem can du doan
    X, y = split_features_target(encoded_df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    feature_names = X_train.columns.tolist()
    models = build_models()

    # Buoc 7: Lan luot train tung mo hinh va luu ket qua danh gia
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

    # Buoc 8: Sap xep ket qua theo RMSE tang dan de tim mo hinh tot nhat
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


# Chay file nay se train model va tao bieu do
def main():
    _, fitted_models, X_test, y_test, best_model_name, feature_names = train_all_models(return_models=True)
    try:
        from .visualize_models import generate_visualizations
    except ImportError:
        from visualize_models import generate_visualizations

    generate_visualizations(fitted_models, X_test, y_test, best_model_name, feature_names)

if __name__ == "__main__":
    main()
