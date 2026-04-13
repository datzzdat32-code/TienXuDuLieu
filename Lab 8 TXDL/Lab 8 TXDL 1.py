import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dữ liệu
df = pd.read_csv('ITA105_Lab_8.csv')

# Tách features và target
X = df.drop(columns=['SalePrice', 'ImagePath']) # Bỏ ImagePath vì không dùng trong model cơ bản
y = df['SalePrice']

# Xử lý cột ngày tháng (Time Series) trước khi đưa vào pipeline
def extract_date_features(df):
    df = df.copy()
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    df['SaleMonth'] = df['SaleDate'].dt.month
    df['SaleQuarter'] = df['SaleDate'].dt.quarter
    return df.drop(columns=['SaleDate'])

# 1. Nhóm các cột
numeric_features = ['LotArea', 'Rooms', 'NoiseFeature']
categorical_features = ['Neighborhood', 'Condition', 'HasGarage']
text_feature = 'Description'

# 2. Xây dựng transformer cho từng loại
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Kết hợp vào ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', TfidfVectorizer(stop_words='english', max_features=50), text_feature)
    ]
)

# Pipeline tổng hợp (Smoke test với 10 dòng)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
demo_data = X.head(10)
output_demo = full_pipeline.fit_transform(demo_data)
print(f"Shape sau pipeline: {output_demo.shape}")