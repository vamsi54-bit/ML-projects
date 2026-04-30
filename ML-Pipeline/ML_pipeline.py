import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

param_grids = {
    "Ridge": {
        'model__alpha': [0.1, 1.0, 10.0]
    },
    "Lasso": {
        'model__alpha': [0.1, 1.0, 10.0]
    },
    "RandomForest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5]
    }
}

results = []

for name, model in models.items():
    print("............Training\n")
    pipeline = Pipeline(
        [("scaler", StandardScaler()),
         ("model", model)]
    )

    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Best Params:", grid.best_params_)
    print("MAE:", mae)
    print("R2:", r2)
    results.append((name, mae, r2))

print("Final comparison-------------------\n")

for name, mae, r2 in results:
    print(f"{name:15} MAE: {mae:.4f} | R2: {r2:.4f}")

plt.figure(figsize=(6, 5))

plt.scatter(y_test, y_pred)

# perfect prediction line
line = [min(y_test), max(y_test)]
plt.plot(line, line)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()
