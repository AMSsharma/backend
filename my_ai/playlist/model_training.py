import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# 1. Load dataset
df = pd.read_csv("data/difficulty_dataset.csv")  # columns: text, difficulty

# 2. Split into features and target
X = df["text"]
y = df["difficulty"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2))),
    ("model", XGBRegressor(objective='reg:squarederror', random_state=42))
])

# 4. Hyperparameter search space
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0]
}

# 5. Randomized Search CV
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 6. Train
search.fit(X_train, y_train)

# 7. Evaluate
y_pred = search.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Best Parameters:", search.best_params_)

# 8. Save model and vectorizer
joblib.dump(search.best_estimator_, "difficulty_model.pkl")

print("Training complete. Model saved as 'difficulty_model.pkl'")
