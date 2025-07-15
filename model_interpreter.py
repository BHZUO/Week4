import shap
import matplotlib.pyplot as plt

def plot_feature_importance(model, features):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.title('特征重要性')
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

def explain_with_shap(model, X_test, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()