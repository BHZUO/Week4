import joblib
import os
import tempfile
import shutil

def setup_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix='joblib_')
    os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
    return temp_dir

def save_model_and_encoders(model, encoders):
    joblib.dump(model, 'pumpkin_model_gb.pkl')
    joblib.dump(encoders, 'target_encoders.pkl')

def cleanup_temp_dir(temp_dir):
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"清理临时文件夹失败: {e}")
