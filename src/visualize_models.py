from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree

try:
    from .models import REPORT_DIR, FIGURE_DIR
except ImportError:
    from models import REPORT_DIR, FIGURE_DIR

OPEN_IMAGES_AFTER_SAVE = True


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
        raise FileNotFoundError(f"Chua co file bao cao: {comparison_path}")

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


def save_actual_vs_predicted_plot(model, X_test, y_test, model_name: str, output_path: Path) -> None:
    y_pred = model.predict(X_test)
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
        pd.DataFrame({"feature": feature_names, "importance": regressor.feature_importances_})
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

def generate_visualizations(models: dict, X_test, y_test, best_model_name: str, open_images: bool = OPEN_IMAGES_AFTER_SAVE) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_images = [
        FIGURE_DIR / "compare.png",
        FIGURE_DIR / "tree.png",
        FIGURE_DIR / "linear.png",
        FIGURE_DIR / "ridge.png",
        FIGURE_DIR / "rf.png",
        FIGURE_DIR / "pred.png",
    ]

    plot_model_comparison(output_images[0])
    plot_decision_tree_model(models["Decision Tree Regressor"], output_images[1])
    plot_linear_coefficients(models["Linear Regression"], "Linear Regression", output_images[2])
    plot_linear_coefficients(models["Ridge Regression"], "Ridge Regression", output_images[3])
    save_feature_importance_plot(
        models["Random Forest Regressor"],
        REPORT_DIR / "random_forest_feature_importance.csv",
        output_images[4],
    )
    save_actual_vs_predicted_plot(models[best_model_name], X_test, y_test, best_model_name, output_images[5])

    print("\nDa tao cac hinh truc quan mo hinh trong thu muc:")
    print(FIGURE_DIR)
    for image_path in output_images:
        print(f"- {image_path.name}")

    if open_images:
        print("\nDang mo cac hinh anh...")
        for image_path in output_images:
            if image_path.exists():
                os.startfile(image_path)

if __name__ == "__main__":
    raise SystemExit("Hay chay python src\\models.py de train va tao bieu do.")
