# Bài 3: (2đ) Chỉ số công ty
# - Khám phá dataset, vẽ boxplot để quan sát scale khác nhau và
# ngoại lệ (công ty cực lớn).
# - Chuẩn hóa bằng Min-Max và Z-Score.
# - Vẽ scatterplot so sánh 2 biến trước và sau chuẩn hóa (ví dụ:
# Doanh thu và Lợi nhuận).
# - Nhận xét: dữ liệu có ngoại lệ lớn → Min-Max có phù hợp không?
# - Thảo luận: chọn phương pháp chuẩn hóa cho mô hình dự đoán tài
# chính (Linear Regression, KNN).

# Bài Làm:
# Bài 3: (2đ) Chỉ số công ty
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Nạp dữ liệu, tìm file CSV có sẵn
csv_file = 'ITA105_Lab_3_Companies.csv'
if not os.path.exists(csv_file):
    candidates = glob.glob('ITA105_Lab_3_*.csv')
    if not candidates:
        raise FileNotFoundError('Không tìm thấy file ITA105_Lab_3_*.csv')
    csv_file = candidates[0]

companies = pd.read_csv(csv_file)
print(f"Dang file: {csv_file}")

# Lấy cột số để chuẩn hóa (bỏ cột không-numeric nếu có)
companies_numeric = companies.select_dtypes(include=[np.number])

# Vẽ boxplot để quan sát scale và ngoại lệ
for i, column in enumerate(companies_numeric.columns, 1):
    plt.figure(figsize=(12, 5))
    plt.boxplot(companies_numeric[column].dropna(), vert=False)
    plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.savefig(f'company_boxplot_{i}_{column}.png')
    plt.close()

# Chuẩn hóa bằng Min-Max Scaling
companies_minmax = (companies_numeric - companies_numeric.min()) / (companies_numeric.max() - companies_numeric.min())

# Chuẩn hóa bằng Z-Score Normalization
companies_zscore = (companies_numeric - companies_numeric.mean()) / companies_numeric.std()

# Chọn hai biến để vẽ scatter plot (nếu có ít nhất 2 biến)
if companies_numeric.shape[1] >= 2:
    x_col, y_col = companies_numeric.columns[0], companies_numeric.columns[1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(companies_numeric[x_col], companies_numeric[y_col], color='blue', alpha=0.7)
    plt.title(f'Original {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.subplot(1, 2, 2)
    plt.scatter(companies_minmax[x_col], companies_minmax[y_col], color='green', alpha=0.7)
    plt.title(f'Min-Max Scaled {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.tight_layout()
    plt.savefig('company_scatter_comparison.png')
    plt.close()
else:
    print('Không đủ biến để vẽ scatter plot. Cần ít nhất 2 biến numeric.')

print('\n=== Du lieu sau khi chuan hoa Min-Max ===')
print(companies_minmax.head())
print('\n=== Du lieu sau khi chuan hoa Z-Score ===')
print(companies_zscore.head())

print('\n=== Nhan xet ===')
print('- Neu dataset co ngoai le lon, Min-Max co the bi bien dang do gia tri ngoai lai anh huong den gioi han [0,1].')
print('- Z-Score on dinh hon khi co ngoai le, ket qua van the hien khoang cach so voi trung binh.')
print('- Ung dung: Linear Regression va KNN thuong uu tien chuan hoa; KNN dac biet nhay voi scale nen Z-Score hoac robust scaler thuong phu hop hon khi co ngoai le.')

