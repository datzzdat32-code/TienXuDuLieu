# Bài 2: (2đ) Chỉ số bệnh nhân
# - Khám phá dữ liệu, thống kê, trực quan hóa (histogram,
# boxplot).
# - Phát hiện ngoại lệ (bệnh nhân cực cao/nhỏ, huyết áp cực đoan).
# - Chuẩn hóa bằng Min-Max và Z-Score.
# - So sánh phân phối trước và sau chuẩn hóa.
# - Nhận xét: biến nào bị ảnh hưởng nhiều bởi ngoại lệ, và phương
# pháp chuẩn hóa nào phù hợp hơn?

# Bài Làm:
# Bài 2: Chỉ số bệnh nhân
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# Nạp dữ liệu
patients = pd.read_csv('ITA105_Lab_3_Health.csv')
# Kiểm tra missing values
print(patients.isnull().sum())
# Thống kê mô tả
print(patients.describe())
# Vẽ histogram và boxplot cho từng biến
for i, column in enumerate(patients.columns, 1):  # Bỏ cột ID
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(patients[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.subplot(1, 2, 2)
    plt.boxplot(patients[column], vert=False)
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.savefig(f'patient_plot_histbox_{i}_{column}.png')
    plt.close()
# Phát hiện ngoại lệ (bệnh nhân cực cao/nhỏ, huyết áp cực đoan)
for column in patients.columns:
    Q1 = patients[column].quantile(0.25)
    Q3 = patients[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = patients[(patients[column] < lower_bound) | (patients[column] > upper_bound)]
    print(f'\nOutliers in {column}:\n{outliers}')
# Chuẩn hóa bằng Min-Max Scaling
patients_numeric = patients
patients_minmax = (patients_numeric - patients_numeric.min()) / (patients_numeric.max() - patients_numeric.min())
patients_minmax_df = patients_minmax.copy()
# Chuẩn hóa bằng Z-Score Normalization
patients_zscore = (patients_numeric - patients_numeric.mean()) / patients_numeric.std()
patients_zscore_df = patients_zscore.copy()
# Vẽ biểu đồ so sánh phân phối trước và sau chuẩn hóa
for i, column in enumerate(patients.columns, 1):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(patients[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Original {column}')
    plt.subplot(1, 3, 2)
    plt.hist(patients_minmax_df[column], bins=20, color='green', alpha=0.7)
    plt.title(f'Min-Max Scaled {column}')
    plt.subplot(1, 3, 3)
    plt.hist(patients_zscore_df[column], bins=20, color='red', alpha=0.7)
    plt.title(f'Z-Score Normalized {column}')
    plt.tight_layout()
    plt.savefig(f'patient_plot_comparison_{i}_{column}.png')
    plt.close()
print("\n=== Dữ liệu sau khi chuẩn hóa Min-Max ===")
print(patients_minmax_df.head())
print("\n=== Dữ liệu sau khi chuẩn hóa Z-Score ===")
print(patients_zscore_df.head())
