from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import numpy as np

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    使用 LGBM 和 XGBoost 进行训练，并返回符合格式的实验记录
    """

    models = {
        "LGBM": lgb.LGBMRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42)
    }

    experiment_results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_records = []
        fold_index = 0

        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)

            y_pred_train = model.predict(X_train_fold)
            y_pred_val = model.predict(X_val_fold)

            # 性能指标计算
            train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_pred_train))
            train_mae = mean_absolute_error(y_train_fold, y_pred_train)
            train_r2 = r2_score(y_train_fold, y_pred_train)

            val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
            val_mae = mean_absolute_error(y_val_fold, y_pred_val)
            val_r2 = r2_score(y_val_fold, y_pred_val)

            fold_record = {
                f"{fold_index}_fold_train_data": [len(y_train_fold), len(X_train_fold.columns)],
                f"{fold_index}_fold_test_data": [len(y_val_fold), len(X_val_fold.columns)],
                f"{fold_index}_fold_train_performance": {
                    "rmse": round(train_rmse, 2),
                    "mae": round(train_mae, 2),
                    "r2": round(train_r2, 2)
                },
                f"{fold_index}_fold_test_performance": {
                    "rmse": round(val_rmse, 2),
                    "mae": round(val_mae, 2),
                    "r2": round(val_r2, 2)
                }
            }
            fold_records.append(fold_record)
            fold_index += 1

        # 最终在完整训练集上评估模型
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # 收集每个 fold 的性能指标并求平均
        train_rmses = [r[f"{i}_fold_train_performance"]["rmse"] for i, r in enumerate(fold_records)]
        train_maes = [r[f"{i}_fold_train_performance"]["mae"] for i, r in enumerate(fold_records)]
        train_r2s = [r[f"{i}_fold_train_performance"]["r2"] for i, r in enumerate(fold_records)]

        test_rmses = [r[f"{i}_fold_test_performance"]["rmse"] for i, r in enumerate(fold_records)]
        test_maes = [r[f"{i}_fold_test_performance"]["mae"] for i, r in enumerate(fold_records)]
        test_r2s = [r[f"{i}_fold_test_performance"]["r2"] for i, r in enumerate(fold_records)]

        avg_train_rmse = round(np.mean(train_rmses), 2)
        avg_train_mae = round(np.mean(train_maes), 2)
        avg_train_r2 = round(np.mean(train_r2s), 2)

        avg_test_rmse = round(np.mean(test_rmses), 2)
        avg_test_mae = round(np.mean(test_maes), 2)
        avg_test_r2 = round(np.mean(test_r2s), 2)

        # 合并所有 fold 信息为一个完整的实验记录
        full_result = {
            "model_name": name,
            "model_params": str(model.get_params()),
            "fea_encoding": "target",
            **{k: v for fold in fold_records for k, v in fold.items()},
            "average_train_performance": {
                "rmse": avg_train_rmse,
                "mae": avg_train_mae,
                "r2": avg_train_r2
            },
            "average_test_performance": {
                "rmse": round(test_rmse, 2),
                "mae": round(test_mae, 2),
                "r2": round(test_r2, 2)
            }
        }

        experiment_results.append(full_result)

    return experiment_results