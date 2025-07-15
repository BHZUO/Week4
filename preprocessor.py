import pandas as pd
from sklearn.preprocessing import TargetEncoder

def add_date_features(df):
    """添加日期相关特征"""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsFall'] = df['Month'].isin([9, 10, 11]).astype(int)
    return df

def create_target_variable(df):
    """创建目标变量 Avg_Price 和 Price_Range"""
    df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2
    df['Price_Range'] = df['High Price'] - df['Low Price']
    return df

def handle_categorical_columns(df, categorical_cols):
    """处理分类变量缺失值并统一格式"""
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown').astype(str).str.upper().str.strip()
    return df

def apply_target_encoding(model_data, categorical_cols, target='Avg_Price'):
    """应用目标编码"""
    encoders = {}
    for col in categorical_cols:
        te = TargetEncoder(random_state=42)
        model_data[col] = te.fit_transform(model_data[[col]], model_data[target])
        encoders[col] = te
    return model_data, encoders

def prepare_features_and_target(df, features, target='Avg_Price'):
    """提取最终特征和目标变量"""
    model_data = df[features + [target]].dropna()
    X = model_data[features]
    y = model_data[target]
    return X, y