import os
from modules.data_loader import load_data, print_data_info
from modules.preprocessor import (
    add_date_features,
    create_target_variable,
    handle_categorical_columns,
    apply_target_encoding,
    prepare_features_and_target
)
from modules.explorer import plot_price_distribution, plot_price_by_variety
from modules.model_trainer import (
    split_data,
    evaluate_models,
    train_best_model,
    evaluate_model
)
from modules.model_interpreter import plot_feature_importance, explain_with_shap
from modules.saver import setup_temp_dir, save_model_and_encoders, cleanup_temp_dir

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

    print("\n7. 探索性数据分析...")
    plot_price_distribution(model_data['Avg_Price'])
    plot_price_by_variety(model_data)

    print("\n8. 划分数据集...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n9. 模型比较...")
    evaluate_models(X_train, y_train)

    print("\n10. 训练最佳模型...")
    best_model, best_params = train_best_model(X_train, y_train)
    print(f"\n最佳参数: {best_params}")

    print("\n11. 模型评估...")
    evaluate_model(best_model, X_test, y_test)

    print("\n12. 特征重要性与模型解释...")
    plot_feature_importance(best_model, features)
    explain_with_shap(best_model, X_test, features)

    print("\n13. 保存模型...")
    save_model_and_encoders(best_model, encoders)

    print("\n14. 清理临时文件夹...")
    temp_dir = setup_temp_dir()
    cleanup_temp_dir(temp_dir)

    print("\n✅ 分析完成！结果已保存到当前目录。")

if __name__ == '__main__':
    main()