import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
