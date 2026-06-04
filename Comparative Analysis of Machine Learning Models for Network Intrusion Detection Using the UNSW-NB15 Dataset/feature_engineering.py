import pandas as pd

train_df = pd.read_csv("UNSW_NB15_training-set.csv")

X = train_df.drop(["id", "label", "attack_cat"], axis=1)

print(X.shape)

print(X.select_dtypes(include="object").columns)
