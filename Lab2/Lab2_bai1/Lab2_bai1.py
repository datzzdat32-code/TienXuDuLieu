import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Nạp dữ liệu và kiểm tra [cite: 33]
df_housing = pd.read_csv('ITA105_Lab_2_Housing.csv')
print(df_housing.shape)
print(df_housing.isnull().sum())

# 2. Thống kê mô tả [cite: 34]
print(df_housing.describe())

# 3. Vẽ boxplot cho từng biến numeric [cite: 36]
plt.figure(figsize=(12, 4))
for i, col in enumerate(['dien_tich', 'gia', 'so_phong']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df_housing[col])
    plt.title(f'Boxplot {col}')
plt.show()

# 4. Vẽ scatterplot diện tích và giá [cite: 37]
plt.scatter(df_housing['dien_tich'], df_housing['gia'])
plt.xlabel('Diện tích')
plt.ylabel('Giá')
plt.title('Scatter Plot: Diện tích vs Giá')
plt.show()

# 5 & 6. Tính IQR và Z-score để xác định ngoại lệ [cite: 38, 39]
# Ví dụ cho cột 'gia'
Q1 = df_housing['gia'].quantile(0.25)
Q3 = df_housing['gia'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df_housing[(df_housing['gia'] < (Q1 - 1.5 * IQR)) | (df_housing['gia'] > (Q3 + 1.5 * IQR))]

z_scores = stats.zscore(df_housing[['dien_tich', 'gia', 'so_phong']])
outliers_z = df_housing[(abs(z_scores) > 3).any(axis=1)]

# 9. Xử lý ngoại lệ (Ví dụ: dùng Clip giá trị) [cite: 44, 46]
df_housing['gia_clipped'] = df_housing['gia'].clip(lower=df_housing['gia'].quantile(0.05), 
                                                  upper=df_housing['gia'].quantile(0.95))


