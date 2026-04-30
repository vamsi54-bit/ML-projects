import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_iris(as_frame=True)
X = data.data   # features only

print("Features:\n", X.head())
print("\nShape:", X.shape)

# -----------------------------
# 2. Basic Visualization
# -----------------------------
plt.scatter(
    X['petal length (cm)'],
    X['petal width (cm)']
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Raw Data Visualization")
plt.tight_layout()
plt.savefig("raw_data.png")
plt.show()

# -----------------------------
# 3. Elbow Method
# -----------------------------
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# -----------------------------
# 4. Silhouette Method
# -----------------------------
sil_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)
    print(f"K={k}, Silhouette Score={score:.4f}")

plt.plot(range(2, 7), sil_scores, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.tight_layout()
plt.savefig("silhouette_method.png")
plt.show()

# -----------------------------
# 5. Final Model (choose K)
# -----------------------------
best_k = 3   # change based on your observation
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)
X = X.copy()
X['Cluster'] = labels

# -----------------------------
# 6. Final Visualization
# -----------------------------
plt.scatter(
    X['petal length (cm)'],
    X['petal width (cm)'],
    c=X['Cluster'],
    cmap='viridis'
)

# Cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 2],  # petal length index
    centers[:, 3],  # petal width index
    marker='X',
    s=200,
    c='red',
    label='Centroids'
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title(f"KMeans Clustering (K={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_clusters.png")
plt.show()

# -----------------------------
# 7. Evaluate with Silhouette
# -----------------------------
final_score = silhouette_score(X.drop('Cluster', axis=1), X['Cluster'])
print("\nFinal Silhouette Score:", final_score)
