import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Variance ratio: ", pca.explained_variance_ratio_)
print("Total variance : ", pca.explained_variance_ratio_.sum())

df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["target"] = y

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2",
                hue="target", palette="deep")
plt.title("PCA using Seaborn — Iris Dataset")
plt.tight_layout()
plt.savefig('/home/claude/iris_pca.png', dpi=150, bbox_inches='tight')
plt.show()
