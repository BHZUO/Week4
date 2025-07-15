import pandas as pd
from datetime import datetime

def load_data(filepath):
    """
    加载数据并进行基本类型转换和日期解析
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    return df

def print_data_info(df):
    """
    打印数据集基本信息
    """
    print(f"\n数据集形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据概览:")
    print(df.info())
    print("\n描述性统计:")
    print(df.describe(include='all'))