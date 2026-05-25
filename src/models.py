from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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
        make_preprocessor,
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
        make_preprocessor,
        missing_value_report,
        save_processed_dataset,
        split_features_target,
        train_test_split_data,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por-10.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student-por-10-processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def build_models(preprocessor) -> dict:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", Ridge()),
                ]
            ),
            param_grid={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        ),
        "Decision Tree Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
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


def save_actual_vs_predicted_plot(
    y_test: pd.Series,
    y_pred,
    model_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="#2563eb", s=55)
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#dc2626", linewidth=2)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance_plot(random_forest_model, output_csv: Path, output_png: Path) -> None:
    preprocessor = random_forest_model.named_steps["preprocessor"]
    regressor = random_forest_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": regressor.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )
    importance_df.to_csv(output_csv, index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", legend=False)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def train_all_models() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(RAW_DATA_PATH)
    print("Kich thuoc du lieu:", df.shape)
    print("\nGia tri thieu theo cot:")
    print(missing_value_report(df))

    save_processed_dataset(df, PROCESSED_DATA_PATH)

    corr = correlation_with_target(df)
    corr.to_csv(REPORT_DIR / "correlation_with_G3.csv", header=["correlation"])
    print("\nTop 15 bien tuong quan manh nhat voi G3:")
    print(corr.head(15))

    X, y = split_features_target(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    preprocessor = make_preprocessor(X_train)
    models = build_models(preprocessor)

    results = []
    fitted_models = {}
    model_file_names = {
        "Linear Regression": "linear_reg.pkl",
        "Ridge Regression": "ridge_reg.pkl",
        "Decision Tree Regressor": "decision_tree.pkl",
        "Random Forest Regressor": "random_forest.pkl",
    }

    for model_name, model in models.items():
        print(f"\nDang huan luyen: {model_name}")
        model.fit(X_train, y_train)

        best_alpha = None
        model_to_save = model
        if isinstance(model, GridSearchCV):
            best_alpha = model.best_params_["model__alpha"]
            model_to_save = model.best_estimator_

        fitted_models[model_name] = model_to_save
        joblib.dump(model_to_save, MODEL_DIR / model_file_names[model_name])

        metrics = evaluate_model(model, X_test, y_test)
        results.append({"Model": model_name, **metrics, "Best alpha": best_alpha})

    results_df = (
        pd.DataFrame(results)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.loc[0, "Model"]
    best_model = fitted_models[best_model_name]
    save_actual_vs_predicted_plot(
        y_test=y_test,
        y_pred=best_model.predict(X_test),
        model_name=best_model_name,
        output_path=FIGURE_DIR / "actual_vs_predicted_best_model.png",
    )
    save_feature_importance_plot(
        random_forest_model=fitted_models["Random Forest Regressor"],
        output_csv=REPORT_DIR / "random_forest_feature_importance.csv",
        output_png=FIGURE_DIR / "random_forest_feature_importance.png",
    )

    print("\nBang so sanh mo hinh tren tap Test:")
    print(results_df.to_string(index=False))
    print(f"\nMo hinh tot nhat theo RMSE: {best_model_name}")
    return results_df


if __name__ == "__main__":
    train_all_models()
