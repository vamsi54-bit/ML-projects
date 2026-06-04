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

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# Train
print("Training Decision Tree...")
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training LightGBM...")
lgbm_model.fit(X_train, y_train)

# Predict
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

# Evaluate
models = {
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "LightGBM": lgbm_pred
}

print("\n" + "=" * 30)
print("       MODEL EVALUATION")
print("=" * 30)

for name, pred in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(
        y_test,
        pred,
        average='weighted'
    )
    recall = recall_score(
        y_test,
        pred,
        average='weighted'
    )
    f1 = f1_score(
        y_test,
        pred,
        average='weighted'
    )
    print(f"\n{name}")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
