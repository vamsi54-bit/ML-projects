# ==========================================================
# Network Intrusion Detection System using UNSW-NB15 Dataset
# Models:
# 1. Decision Tree
# 2. Random Forest
# 3. LightGBM
# ==========================================================

# =====================
# Import Libraries
# =====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================
# Load Dataset
# =====================

train_df = pd.read_csv("UNSW_NB15_training-set.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nColumns:")
print(train_df.columns)

print("\nMissing Values:")
print(train_df.isnull().sum())

# =====================
# EDA
# =====================

print("\nAttack Categories:")
print(train_df["attack_cat"].value_counts())

plt.figure(figsize=(10, 6))

train_df["attack_cat"].value_counts().plot(
    kind="bar"
)

plt.title("Attack Category Distribution")
plt.xlabel("Attack Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =====================
# Feature Engineering
# =====================

X = train_df.drop(
    ["id", "attack_cat", "label"],
    axis=1
)

y = train_df["attack_cat"]

# =====================
# Encode Features
# =====================

le_proto = LabelEncoder()
le_service = LabelEncoder()
le_state = LabelEncoder()

X["proto"] = le_proto.fit_transform(X["proto"])
X["service"] = le_service.fit_transform(X["service"])
X["state"] = le_state.fit_transform(X["state"])

# =====================
# Encode Target
# =====================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

print("\nTarget Classes:")
print(target_encoder.classes_)

# =====================
# Train Test Split
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================
# Models
# =====================

dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

lgbm_model = LGBMClassifier(
    objective="multiclass",
    num_class=10,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

# =====================
# Training
# =====================

print("\nTraining Decision Tree...")
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training LightGBM...")
lgbm_model.fit(X_train, y_train)

# =====================
# Predictions
# =====================

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

# =====================
# Evaluation Function
# =====================

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_encoder.classes_
        )
    )

    return [
        name,
        accuracy,
        precision,
        recall,
        f1
    ]

# =====================
# Evaluate All Models
# =====================

results = []

results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        dt_pred
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred
    )
)

results.append(
    evaluate_model(
        "LightGBM",
        y_test,
        lgbm_pred
    )
)

# =====================
# Comparison Table
# =====================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# =====================
# Best Model Selection
# =====================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

print("\nBest Model:", best_model_name)

# =====================
# Confusion Matrix
# =====================

if best_model_name == "Decision Tree":
    best_pred = dt_pred

elif best_model_name == "Random Forest":
    best_pred = rf_pred

else:
    best_pred = lgbm_pred

cm = confusion_matrix(
    y_test,
    best_pred
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="rocket"
)

plt.title(
    f"{best_model_name} Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# =====================
# Feature Importance
# =====================

if best_model_name == "Random Forest":

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })

elif best_model_name == "Decision Tree":

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": dt_model.feature_importances_
    })

else:

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": lgbm_model.feature_importances_
    })

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(15))

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"].head(15),
    importance["Importance"].head(15)
)

plt.title(
    f"Top 15 Features - {best_model_name}"
)

plt.tight_layout()
plt.show()

# =====================
# Save Best Model
# =====================

if best_model_name == "Decision Tree":

    joblib.dump(
        dt_model,
        "best_intrusion_model.pkl"
    )

elif best_model_name == "Random Forest":

    joblib.dump(
        rf_model,
        "best_intrusion_model.pkl"
    )

else:

    joblib.dump(
        lgbm_model,
        "best_intrusion_model.pkl"
    )

print("\nModel Saved Successfully!")

# =====================
# End of Project
# =====================
