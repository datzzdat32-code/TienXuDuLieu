# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline với Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Chạy 5-fold Cross-validation
cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='r2')

print(f"R2 Scores giữa các folds: {cv_scores}")
print(f"R2 trung bình: {cv_scores.mean():.4f}")

# Train model cuối cùng
rf_pipeline.fit(X_train, y_train)