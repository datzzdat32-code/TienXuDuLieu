# Bài 1: (2đ) Thông số vận động viên
# - Nạp dữ liệu, kiểm tra missing values, thống kê mô tả.
# - Vẽ histogram và boxplot cho từng biến để quan sát scale và
# phân phối.
# - Chuẩn hóa từng biến bằng Min-Max Scaling → đưa về [0,1].
# - Chuẩn hóa từng biến bằng Z-Score Normalization → mean = 0, std
# = 1.
# - Vẽ biểu đồ so sánh phân phối dữ liệu trước và sau chuẩn hóa
# (Min-Max và Z-Score).

# Bài Làm:

# Bài 1: Thông số vận động viên
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# Nạp dữ liệu
athletes = pd.read_csv('ITA105_Lab_3_Sports.csv')
# Kiểm tra missing values
print(athletes.isnull().sum())
# Thống kê mô tả
print(athletes.describe())
# Vẽ histogram và boxplot cho từng biến
for i, column in enumerate(athletes.columns[1:], 1):  # Bỏ cột tên
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(athletes[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.subplot(1, 2, 2)
    plt.boxplot(athletes[column], vert=False)
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.savefig(f'plot_histbox_{i}_{column}.png')
    plt.close()
# Chuẩn hóa bằng Min-Max Scaling
athletes_numeric = athletes.iloc[:, 1:]
athletes_minmax = (athletes_numeric - athletes_numeric.min()) / (athletes_numeric.max() - athletes_numeric.min())
athletes_minmax_df = athletes_minmax.copy()
# Chuẩn hóa bằng Z-Score Normalization
athletes_zscore = (athletes_numeric - athletes_numeric.mean()) / athletes_numeric.std()
athletes_zscore_df = athletes_zscore.copy()
# Vẽ biểu đồ so sánh phân phối dữ liệu trước và sau chuẩn hóa
for i, column in enumerate(athletes.columns[1:], 1):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(athletes[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Original {column}')
    plt.subplot(1, 3, 2)
    plt.hist(athletes_minmax_df[column], bins=20, color='green', alpha=0.7)
    plt.title(f'Min-Max Scaled {column}')
    plt.subplot(1, 3, 3)
    plt.hist(athletes_zscore_df[column], bins=20, color='red', alpha=0.7)
    plt.title(f'Z-Score Normalized {column}')
    plt.tight_layout()
    plt.savefig(f'plot_comparison_{i}_{column}.png')
    plt.close()

print("\n=== Dữ liệu sau khi chuẩn hóa Min-Max ===")
print(athletes_minmax_df.head())
print("\n=== Dữ liệu sau khi chuẩn hóa Z-Score ===")
print(athletes_zscore_df.head())

