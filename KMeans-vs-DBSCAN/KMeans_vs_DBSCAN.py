import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. Generate & Scale Data
# -----------------------------
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------
# 2. KMeans Clustering
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)

# -----------------------------
# 3. DBSCAN Clustering
# -----------------------------
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

mask = dbscan_labels != -1

if len(set(dbscan_labels)) > 1 and np.sum(mask) > 0:
    dbscan_score = silhouette_score(X_scaled[mask], dbscan_labels[mask])
else:
    dbscan_score = -1

# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.title("Raw Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title(f"KMeans\nSilhouette Score: {kmeans_score:.2f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title(f"DBSCAN\nSilhouette Score: {dbscan_score:.2f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.savefig("KMeans_vs_DBSCAN.png", dpi=150)
plt.show()

# -----------------------------
# 5. Final Comparison
# -----------------------------
print("\n===== FINAL COMPARISON =====")
print(f"KMeans Silhouette Score: {kmeans_score:.4f}")
print(f"DBSCAN Silhouette Score: {dbscan_score:.4f}")

print("\nDBSCAN unique labels (note: -1 = noise):")
print(set(dbscan_labels))
