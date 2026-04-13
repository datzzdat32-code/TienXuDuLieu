# 1. np.log() cho cột dương
df['SalePrice_log'] = np.log1p(df['SalePrice'])

# 2. scipy.stats.boxcox() cho cột dương
df['LotArea_boxcox'], _ = stats.boxcox(df['LotArea'])

# 3. PowerTransformer (Yeo-Johnson) cho cột có giá trị âm
pt = PowerTransformer(method='yeo-johnson')
df['NegSkewIncome_pt'] = pt.fit_transform(df[['NegSkewIncome']])

# Vẽ biểu đồ so sánh trước và sau transform
# (Bạn có thể lặp lại bước vẽ biểu đồ như ở Bài 1 để so sánh)