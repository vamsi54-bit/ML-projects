import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
gmm = GaussianMixture(n_components=3, random_state=42)

k_labels = kmeans.fit_predict(X_scaled)
g_labels = gmm.fit_predict(X_scaled)

df = pd.DataFrame(X_scaled[:, :2], columns=["Feature1", "Feature2"])
df["KMeans"] = k_labels
df["GMM"] = g_labels

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(data=df, x="Feature1", y="Feature2", hue="KMeans", palette="Set1", ax=axes[0])
axes[0].set_title("KMeans Clustering")

sns.scatterplot(data=df, x="Feature1", y="Feature2", hue="GMM", palette="Set2", ax=axes[1])
axes[1].set_title("GMM Clustering")

plt.tight_layout()
plt.show()

probs = gmm.predict_proba(X_scaled)
print("GMM probabilities (first 5 rows):\n", probs[:5])
