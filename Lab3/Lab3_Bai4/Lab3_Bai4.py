# Bài 4: (2đ) Người chơi trực tuyến
# - Khám phá dữ liệu, kiểm tra missing values, trực quan hóa phân
# phối.
# - Chuẩn hóa bằng Min-Max và Z-Score.
# - Vẽ histogram so sánh phân phối trước và sau chuẩn hóa.
# - Thảo luận: một số người chơi cực kỳ “cày cuốc” → ngoại lệ,
# phương pháp nào ổn hơn?
# - Chuẩn hóa dữ liệu để chuẩn bị cho mô hình clustering hoặc KNN
# (highlight lý do chọn phương pháp).

# Bài Làm:
# Bài 4: Người chơi trực tuyến
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
# Nạp dữ liệu
players = pd.read_csv(r'C:\Users\PC\OneDrive\Desktop\Lab3\ITA105_Lab_3_Gaming.csv')  # dùng file có sẵn
# Khám phá dữ liệu, kiểm tra missing values, trực quan hóa phân phối
print(players.isnull().sum())
print(players.describe())
for i, column in enumerate(players.columns[1:], 1):  # Bỏ cột tên
    plt.figure(figsize=(12, 5))
    plt.hist(players[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.savefig(f'player_hist_{i}_{column}.png')
    plt.close()
# Chuẩn hóa bằng Min-Max Scaling
players_numeric = players.iloc[:, 1:]
players_minmax = (players_numeric - players_numeric.min()) / (players_numeric.max() - players_numeric.min())
players_minmax_df = players_minmax.copy()
# Chuẩn hóa bằng Z-Score Normalization
players_zscore = (players_numeric - players_numeric.mean()) / players_numeric.std()
players_zscore_df = players_zscore.copy()
# Vẽ histogram so sánh phân phối trước và sau chuẩn hóa
for i, column in enumerate(players.columns[1:], 1):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(players[column], bins=20, color='blue', alpha=0.7)
    plt.title(f'Original {column}')
    plt.subplot(1, 3, 2)
    plt.hist(players_minmax_df[column], bins=20, color='green', alpha=0.7)
    plt.title(f'Min-Max Scaled {column}')
    plt.subplot(1, 3, 3)
    plt.hist(players_zscore_df[column], bins=20, color='red', alpha=0.7)
    plt.title(f'Z-Score Normalized {column}')
    plt.tight_layout()
    plt.savefig(f'player_hist_comparison_{i}_{column}.png')
    plt.close()
print("\n=== Dữ liệu sau khi chuẩn hóa Min-Max ===")
print(players_minmax_df.head())
print("\n=== Dữ liệu sau khi chuẩn hóa Z-Score ===")
print(players_zscore_df.head())
# - Thảo luận: một số người chơi cực kỳ “cày cuốc” → ngoại lệ, phương pháp nào ổn hơn?
# - Chuẩn hóa dữ liệu để chuẩn bị cho mô hình clustering hoặc KNN (highlight lý do chọn phương pháp).
