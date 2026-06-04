import pandas as pd

train_df = pd.read_csv("UNSW_NB15_training-set.csv")

print("Shape:", train_df.shape)

print("\nColumns:")
print(train_df.columns)

print("\nInfo:")
print(train_df.info())

print("\nFirst 5 Rows:")
print(train_df.head())
