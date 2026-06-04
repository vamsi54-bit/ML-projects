import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("UNSW_NB15_training-set.csv")

train_df["attack_cat"].value_counts().plot(kind="bar")
plt.title("Attack Category Distribution")
plt.xlabel("Attack Category")
plt.ylabel("Count")
plt.show()

print(train_df["attack_cat"].value_counts())

print(train_df["attack_cat"].nunique())
