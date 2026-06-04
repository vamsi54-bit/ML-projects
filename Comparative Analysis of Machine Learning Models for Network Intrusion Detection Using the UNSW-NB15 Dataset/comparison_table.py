import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

train_df = pd.read_csv("UNSW_NB15_training-set.csv")

X = train_df.drop(["id", "label", "attack_cat"], axis=1)

le_proto = LabelEncoder()
le_service = LabelEncoder()
le_state = LabelEncoder()

X["proto"] = le_proto.fit_transform(X["proto"])
X["service"] = le_service.fit_transform(X["service"])
X["state"] = le_state.fit_transform(X["state"])

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df["attack_cat"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train models
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Training LightGBM...")
lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)

# Predict
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

# Comparison Table
models = {
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "LightGBM": lgbm_pred
}

results = []

for name, pred in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, average="weighted"
        ),
        "Recall": recall_score(
            y_test, pred, average="weighted"
        ),
        "F1 Score": f1_score(
            y_test, pred, average="weighted"
        )
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("              MODEL COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string(index=False))
