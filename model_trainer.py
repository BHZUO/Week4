from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_models(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge()
    }

    print("\n模型交叉验证比较:")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        print(f"{name} 平均R²: {scores.mean():.3f} (±{scores.std():.3f})")

def train_best_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n最佳模型性能:")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- MAE: {mae:.2f}")
    print(f"- R²: {r2:.2f}")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('实际 vs 预测价格')
    plt.savefig('prediction_results.png', bbox_inches='tight', dpi=300)
    plt.close()

    return y_pred