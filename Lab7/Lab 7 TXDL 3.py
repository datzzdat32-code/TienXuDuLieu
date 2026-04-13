from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Chuẩn bị X, y (Ví dụ đơn giản với các cột số)
X = df[['LotArea', 'HouseAge', 'Rooms']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Version A: Dữ liệu gốc
model_a = LinearRegression().fit(X_train, y_train)
pred_a = model_a.predict(X_test)
print(f"Version A - RMSE: {np.sqrt(mean_squared_error(y_test, pred_a))}")

# Version B: Log biến mục tiêu (SalePrice)
y_train_log = np.log1p(y_train)
model_b = LinearRegression().fit(X_train, y_train_log)
pred_b_log = model_b.predict(X_test)
pred_b = np.expm1(pred_b_log) # Dịch ngược kết quả
print(f"Version B - RMSE: {np.sqrt(mean_squared_error(y_test, pred_b))}")