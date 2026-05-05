import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data

df = pd.DataFrame(X, columns=data.feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = model.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=df["alcohol"], y=df["malic_acid"], hue=df["Anomaly"], palette={1: "blue", -1: "red"})
plt.title("Isolation Forest on Wine Dataset")
plt.show()

print(df[df["Anomaly"] == -1].head())
