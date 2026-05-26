from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Cot can du doan la diem cuoi ky G3
TARGET_COLUMN = "G3"
RANDOM_STATE = 42

# Doc file CSV thanh bang du lieu pandas
def load_data(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {csv_path}")
    return pd.read_csv(csv_path)

# Dem so gia tri bi thieu trong tung cot
def missing_value_report(df):
    return df.isna().sum()

# Xu ly du lieu thieu va ma hoa cot chu thanh cot so
def encode_data(df):
    df_clean = df.copy()

    for column in df_clean.columns:
        if df_clean[column].isna().any():
            if df_clean[column].dtype == "object":
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            else:
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    return pd.get_dummies(df_clean, drop_first=False)

# Tach du lieu thanh X va y
# X: cac cot dau vao, y: cot diem G3 can du doan
def split_features_target(df, target_column=TARGET_COLUMN):
    if target_column not in df.columns:
        raise ValueError(f"Du lieu phai co cot muc tieu '{target_column}'")
    return df.drop(columns=[target_column]), df[target_column]

# Chia du lieu thanh 80% train va 20% test
def train_test_split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Tinh tuong quan giua cac cot voi diem G3
def correlation_with_target(df, target_column=TARGET_COLUMN):
    encoded_df = encode_data(df)
    return (
        encoded_df.corr(numeric_only=True)[target_column]
        .drop(target_column)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

# Luu du lieu sau khi xu ly ra file CSV de xem lai
def save_processed_dataset(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_df = encode_data(df)
    encoded_df.to_csv(output_path, index=False)
    return encoded_df
