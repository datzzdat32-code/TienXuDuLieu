# 1. Lưu pipeline
joblib.dump(rf_pipeline, 'house_price_pipeline.pkl')

# 2. Hàm dự báo thực tế
def predict_price(new_data_path):
    model = joblib.load('house_price_pipeline.pkl')
    new_data = pd.read_csv(new_data_path)
    # Lấy các cột cần thiết
    predictions = model.predict(new_data)
    return predictions

# Lưu kết quả test
y_pred = rf_pipeline.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")