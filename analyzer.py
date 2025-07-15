import pandas as pd
import json

def analyze_predictions(X_test, y_test, predictions, df_original=None, top_n=2):
    """
    分析预测结果中的正确和错误样本
    """
    errors = abs(predictions - y_test.values)
    error_df = pd.DataFrame({
        'index': X_test.index,
        'true_value': y_test.values,
        'predicted_value': predictions,
        'error': errors
    })

    correct_samples = error_df[error_df['error'] < 50].head(top_n)
    incorrect_samples = error_df[error_df['error'] >= 50].sort_values('error', ascending=False).head(top_n)

    analysis = {
        "correct_samples": correct_samples.to_dict(orient='records'),
        "incorrect_samples": incorrect_samples.to_dict(orient='records')
    }

    with open('sample_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)

    return analysis