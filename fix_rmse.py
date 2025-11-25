import re

with open(r'c:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4\TrainingModel1.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'rmse_raw = mean_squared_error(y_true_raw, y_pred_raw, squared=False)',
    'rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))'
)

with open(r'c:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4\TrainingModel1.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed RMSE calculation")
