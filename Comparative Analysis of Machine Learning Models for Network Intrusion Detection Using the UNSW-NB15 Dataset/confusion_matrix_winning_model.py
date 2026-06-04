import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Train winning model (Random Forest)
print("Training Random Forest (Winning Model)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)

plt.title("Confusion Matrix - Random Forest (Winning Model)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nConfusion Matrix (raw):")
print(cm)
