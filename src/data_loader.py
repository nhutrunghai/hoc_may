from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "G3"
RANDOM_STATE = 42


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw student dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {csv_path}")
    return pd.read_csv(csv_path)


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Return missing value counts for every column."""
    return df.isna().sum()


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Du lieu phai co cot muc tieu '{target_column}'")
    return df.drop(columns=[target_column]), df[target_column]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline while keeping all original features."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def correlation_with_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """Calculate correlation after one-hot encoding categorical columns."""
    encoded_df = pd.get_dummies(df, drop_first=False)
    return (
        encoded_df.corr(numeric_only=True)[target_column]
        .drop(target_column)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Save a cleaned and one-hot encoded dataset for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean = df.copy()
    for column in df_clean.columns:
        if df_clean[column].isna().any():
            if df_clean[column].dtype == "object":
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            else:
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    encoded_df = pd.get_dummies(df_clean, drop_first=False)
    encoded_df.to_csv(output_path, index=False)
    return encoded_df
