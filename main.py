import os
from modules.data_loader import load_data, print_data_info
from modules.preprocessor import (
    add_date_features, create_target_variable, handle_categorical_columns,
    apply_target_encoding, prepare_features_and_target
)
from modules.explorer import plot_price_distribution, plot_price_by_variety
from modules.model_trainer import train_and_evaluate_models
from modules.analyzer import analyze_predictions
from modules.saver import setup_temp_dir, save_model_and_encoders, cleanup_temp_dir
import joblib
import json
import lightgbm as lgb

def main():
    print("1. 数据加载...")
    df = load_data('data/US-pumpkins.csv')
    print_data_info(df)

    print("\n2. 添加日期特征...")
    df = add_date_features(df)

    print("\n3. 创建目标变量...")
    df = create_target_variable(df)

    print("\n4. 处理分类变量...")
    categorical_cols = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color']
    df = handle_categorical_columns(df, categorical_cols)

    print("\n5. 应用目标编码...")
    features = ['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Color',
                'Year', 'Month', 'IsFall', 'DayOfYear']
    target = 'Avg_Price'
    model_data, encoders = apply_target_encoding(df.copy(), categorical_cols, target)

    print("\n6. 准备特征和目标...")
    X, y = prepare_features_and_target(model_data, features, target)

    print("\n7. 划分数据集...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n8. 训练 LGBM 和 XGBoost 模型...")
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    print("\n9. 保存实验结果到 output/experiment_results.json")
    os.makedirs("output", exist_ok=True)
    with open('output/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n10. 样本分析...")
    best_model = lgb.LGBMRegressor().fit(X_train, y_train)  # 可替换为实际最佳模型
    predictions = best_model.predict(X_test)
    analysis = analyze_predictions(X_test, y_test, predictions, df_original=df)

    print("\n11. 保存样本分析到 output/sample_analysis.json")
    with open('output/sample_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)

    print("\n✅ 所有任务完成！")
    print("实验结果已保存至 output/experiment_results.json")
    print("样本分析已保存至 output/sample_analysis.json")

if __name__ == '__main__':
    main()