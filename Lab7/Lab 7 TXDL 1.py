# Tính skewness cho các cột số
numerical_cols = df.select_dtypes(include=[np.number]).columns
skewness = df[numerical_cols].skew().sort_values(ascending=False)

# Lập bảng top 10 cột lệch nhất (trong dataset này có khoảng 5 cột số chính)
print("Top các cột lệch nhất:\n", skewness.head(10))

# Vẽ biểu đồ Histogram + KDE cho 3 cột lệch mạnh nhất (vd: SalePrice, LotArea, NegSkewIncome)
top_3_skewed = skewness.index[:3]
plt.figure(figsize=(15, 5))
for i, col in enumerate(top_3_skewed):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Phân phối của {col}\nSkew: {df[col].skew():.2f}')
plt.tight_layout()
plt.show()