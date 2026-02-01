import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score

# Load data
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Pipeline for X + SVR
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

# Automatically scale y
model = TransformedTargetRegressor(
    regressor=svr_pipeline,
    transformer=StandardScaler()
)

# Hyperparameter tuning
param_grid = {
    "regressor__svr__C": [1, 10, 100],
    "regressor__svr__gamma": ["scale", 0.1, 0.01],
    "regressor__svr__epsilon": [0.01, 0.1, 0.2],
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Evaluation
print("Best parameters:", grid.best_params_)
print("R² score:", r2_score(y_test, y_pred))
