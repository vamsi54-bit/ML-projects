import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

data = load_iris()

X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

df_tsne = pd.DataFrame(X_tsne, columns=['dim1','dim2'])
df_tsne['target'] = y

df_umap = pd.DataFrame(X_umap, columns=['dim1','dim2'])
df_umap['target'] = y

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_tsne, x='dim1', y='dim2', hue='target', palette='deep')
plt.title("t-SNE Visualization")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_umap, x='dim1', y='dim2', hue='target', palette='deep')
plt.title("UMAP Visualization")
plt.show()
