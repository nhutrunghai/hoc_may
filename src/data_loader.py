from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Cột cần dự đoán là điểm cuối kỳ G3
TARGET_COLUMN = "G3"
RANDOM_STATE = 42

# Đọc file CSV thành bảng dữ liệu pandas
def load_data(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_path}")
    return pd.read_csv(csv_path)


# Mã hóa cột chữ thành cột số 
def encode_data(df):
    return pd.get_dummies(df, drop_first=False, dtype=int)

def split_features_target(df, target_column=TARGET_COLUMN):
    if target_column not in df.columns:
        raise ValueError(f"Dữ liệu không có cột mục tiêu '{target_column}'")
    return df.drop(columns=[target_column]), df[target_column]

# Chia dữ liệu thành 80% train và 20% test
def train_test_split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Tính tương quan giữa các cột với điểm G3
def target_corr(df, target_column=TARGET_COLUMN):
    encoded_df = encode_data(df)
    return (
        encoded_df.corr(numeric_only=True)[target_column]
        .drop(target_column)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

# Lưu dữ liệu sau khi xử lý ra file CSV để xem lại
def save_processed_dataset(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_df = encode_data(df)
    encoded_df.to_csv(output_path, index=False)
    return encoded_df
