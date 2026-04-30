import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# -----------------------------
# 1. Load dataset
# -----------------------------
data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame

X = df.drop('class', axis=1)
y = df['class']

# Convert target
y = y.map({'good': 1, 'bad': 0})

print(df.head())
print("\nShape:", df.shape)
print("\nTarget distribution:\n", y.value_counts())

# -----------------------------
# 2. Preprocessing
# -----------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

print("Numerical:", num_cols)
print("Categorical:", cat_cols)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# -----------------------------
# 3. Models
# -----------------------------
models = {
    "Logistic": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "Logistic": {
        "model__C": [0.1, 1, 10]
    },
    "RandomForest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5]
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7]
    }
}

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 5. Training + Evaluation + ROC
# -----------------------------
results = []

plt.figure()
best_auc = 0
best_model_final = None

for name, model in models.items():
    print(f"\n====== Model: {name} ======\n")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    # Train
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    results.append((name, acc, auc))

    # Track best model by AUC
    if auc > best_auc:
        best_auc = auc
        best_model_final = best_model

# -----------------------------
# 6. Final Comparison
# -----------------------------
print("\n====== Final Comparison ======\n")

for name, acc, auc in results:
    print(f"{name:15} Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# -----------------------------
# 7. Plot ROC Curve
# -----------------------------
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# -----------------------------
# 8. Save Best Model
# -----------------------------
joblib.dump(best_model_final, "best_model.pkl")
print("\nBest model saved as best_model.pkl")
