from pathlib import Path
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

try:
    from .models import MODEL_DIR, PROJECT_ROOT, REPORT_DIR, FIGURE_DIR, train_all_models
except ImportError:
    from models import MODEL_DIR, PROJECT_ROOT, REPORT_DIR, FIGURE_DIR, train_all_models


MODEL_FILES = {
    "Linear Regression": "linear_reg.pkl",
    "Ridge Regression": "ridge_reg.pkl",
    "Decision Tree Regressor": "decision_tree.pkl",
    "Random Forest Regressor": "random_forest.pkl",
}

OPEN_IMAGES_AFTER_SAVE = True


def load_trained_models() -> dict:
    missing_files = [
        file_name
        for file_name in MODEL_FILES.values()
        if not (MODEL_DIR / file_name).exists()
    ]
    if missing_files:
        print("Chua co du model .pkl, dang train lai cac mo hinh...")
        train_all_models()

    return {
        model_name: joblib.load(MODEL_DIR / file_name)
        for model_name, file_name in MODEL_FILES.items()
    }


def print_report_tables() -> None:
    comparison_path = REPORT_DIR / "model_comparison.csv"
    correlation_path = REPORT_DIR / "correlation_with_G3.csv"
    feature_importance_path = REPORT_DIR / "random_forest_feature_importance.csv"

    if comparison_path.exists():
        results_df = pd.read_csv(comparison_path)
        print("\n=== Bang so sanh 4 mo hinh tren tap Test ===")
        print(results_df.to_string(index=False))

        best_model = results_df.sort_values("RMSE", ascending=True).iloc[0]
        print(f"\nMo hinh tot nhat theo RMSE: {best_model['Model']}")
        print(f"RMSE: {best_model['RMSE']:.4f}")
        print(f"MAE: {best_model['MAE']:.4f}")
        print(f"R-squared: {best_model['R-squared']:.4f}")

    if correlation_path.exists():
        correlation_df = pd.read_csv(correlation_path)
        print("\n=== Top 15 bien tuong quan manh nhat voi G3 ===")
        print(correlation_df.head(15).to_string(index=False))

    if feature_importance_path.exists():
        importance_df = pd.read_csv(feature_importance_path)
        print("\n=== Top 20 Feature Importance cua Random Forest ===")
        print(importance_df.head(20).to_string(index=False))


def get_feature_names(model) -> list[str]:
    return model.named_steps["preprocessor"].get_feature_names_out().tolist()


def plot_model_comparison(output_path: Path) -> None:
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        train_all_models()

    results_df = pd.read_csv(comparison_path)
    melted_df = results_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE", "R-squared"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric")
    plt.title("Model Comparison on Test Set")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_decision_tree_model(model, output_path: Path) -> None:
    feature_names = get_feature_names(model)
    tree_model = model.named_steps["model"]

    plt.figure(figsize=(24, 12))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree Regressor - First 3 Levels")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_linear_coefficients(model, model_name: str, output_path: Path, top_n: int = 20) -> None:
    feature_names = get_feature_names(model)
    coefficients = model.named_steps["model"].coef_

    coef_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": abs(coefficients),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .head(top_n)
        .sort_values("coefficient", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    colors = ["#dc2626" if value < 0 else "#2563eb" for value in coef_df["coefficient"]]
    plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    plt.axvline(0, color="#111827", linewidth=1)
    plt.title(f"Top {top_n} Coefficients - {model_name}")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    models = load_trained_models()
    print_report_tables()

    output_images = [
        FIGURE_DIR / "model_comparison_metrics.png",
        FIGURE_DIR / "decision_tree_visualization.png",
        FIGURE_DIR / "linear_regression_coefficients.png",
        FIGURE_DIR / "ridge_regression_coefficients.png",
        FIGURE_DIR / "random_forest_feature_importance.png",
        FIGURE_DIR / "actual_vs_predicted_best_model.png",
    ]

    plot_model_comparison(output_images[0])
    plot_decision_tree_model(
        models["Decision Tree Regressor"],
        output_images[1],
    )
    plot_linear_coefficients(
        models["Linear Regression"],
        "Linear Regression",
        output_images[2],
    )
    plot_linear_coefficients(
        models["Ridge Regression"],
        "Ridge Regression",
        output_images[3],
    )

    print("Da tao cac hinh truc quan mo hinh trong thu muc:")
    print(FIGURE_DIR)
    print("- model_comparison_metrics.png")
    print("- decision_tree_visualization.png")
    print("- linear_regression_coefficients.png")
    print("- ridge_regression_coefficients.png")
    print("- random_forest_feature_importance.png")
    print("- actual_vs_predicted_best_model.png")

    if OPEN_IMAGES_AFTER_SAVE:
        print("\nDang mo cac hinh anh...")
        for image_path in output_images:
            if image_path.exists():
                os.startfile(image_path)


if __name__ == "__main__":
    main()
