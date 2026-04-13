# =========================================
# 0. INSTALL
# =========================================
!pip install scikit-learn plotly seaborn --quiet

# =========================================
# 1. IMPORT
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# 2. CREATE DATA (SAFE VERSION)
# =========================================
np.random.seed(42)

n = 1000

df = pd.DataFrame({
    'price': np.random.randint(500, 5000, n),
    'area': np.random.randint(20, 200, n),
    'rooms': np.random.randint(1, 6, n),
    'city': np.random.choice(['Hanoi', 'HCM', 'Danang'], n),
    'description': np.random.choice([
        'luxury apartment with nice view',
        'cozy small house',
        'modern villa near center',
        'cheap house need renovation'
    ], n),
    'date': pd.date_range(start='2020-01-01', periods=n, freq='D')
})

# Thêm missing + duplicate (FIX index)
df.loc[::10, 'area'] = np.nan
df = pd.concat([df, df.iloc[:20]], ignore_index=True)  # FIX duplicate index

print("Shape:", df.shape)

# =========================================
# 3. CLEAN TRƯỚC (QUAN TRỌNG)
# =========================================

# Fix dtype
df['area'] = pd.to_numeric(df['area'], errors='coerce')

# Missing
df['area'] = df['area'].fillna(df['area'].mean())

# Invalid
df = df[(df['price'] > 0) & (df['rooms'] > 0)]

# Drop duplicate rows
df = df.drop_duplicates()

# Reset index (FIX CRITICAL)
df = df.reset_index(drop=True)

print("After cleaning:", df.shape)

# =========================================
# 4. EDA (KHÔNG LỖI)
# =========================================
print(df.describe())

# Safe plot function
def safe_plot(series, plot_type="violin"):
    series = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(series) == 0:
        print("No data to plot")
        return
    
    if plot_type == "violin":
        sns.violinplot(x=series)
    elif plot_type == "box":
        sns.boxplot(x=series)
    
    plt.show()

# Histogram
df.hist(figsize=(10,6))
plt.show()

# Boxplot
safe_plot(df['price'], "box")

# Violin (FIXED)
safe_plot(df['area'], "violin")

# =========================================
# 5. OUTLIER (SAFE)
# =========================================
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
df = df.reset_index(drop=True)

# =========================================
# 6. FEATURE ENGINEERING
# =========================================
df['price_per_m2'] = df['price'] / df['area']

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df['desc_length'] = df['description'].apply(lambda x: len(str(x).split()))

df['area_rooms'] = df['area'] * df['rooms']

df['log_price'] = np.log1p(df['price'])

# =========================================
# 7. PIPELINE (ANTI-ERROR VERSION)
# =========================================

num_cols = ['area', 'rooms', 'price_per_m2', 'desc_length', 'area_rooms']
cat_cols = ['city']
text_col = 'description'

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_pipeline = TfidfVectorizer(max_features=50)

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
    ('text', text_pipeline, text_col)
])

# =========================================
# 8. TRAIN TEST
# =========================================
X = df[num_cols + cat_cols + [text_col]]
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================================
# 9. TRAIN MODELS
# =========================================
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoost": GradientBoostingRegressor()
}

for name, reg in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', reg)
    ])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)

# =========================================
# 10. DASHBOARD
# =========================================
fig = px.scatter(df, x='area', y='price', color='city')
fig.show()

# =========================================
# 11. TEST NEW DATA (KHÔNG BAO GIỜ LỖI)
# =========================================
sample = X.iloc[[0]].copy()

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor())
])

pipe.fit(X_train, y_train)

pred = pipe.predict(sample)
print("\nPredict:", np.expm1(pred))