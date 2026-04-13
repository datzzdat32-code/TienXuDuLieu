# Tạo dữ liệu giả lập có category mới và missing
test_data = X.iloc[0:5].copy()
test_data.loc[0, 'Neighborhood'] = 'New_Zone' # Unseen category
test_data.loc[1, 'LotArea'] = np.nan           # Missing value

try:
    processed_test = full_pipeline.transform(test_data)
    print("Kiểm thử thành công: Pipeline xử lý được dữ liệu lỗi/mới.")
except Exception as e:
    print(f"Lỗi: {e}")