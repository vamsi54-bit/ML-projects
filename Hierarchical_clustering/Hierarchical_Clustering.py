import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_excel(r"C:\Users\DELL\PycharmProjects\PythonProject\Mall Customers.xlsx")

# -----------------------------
# 2. Select Features
# -----------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# 3. Scale Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Plot Raw Data
# -----------------------------
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.7)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Data — Raw")
plt.tight_layout()
plt.savefig("customer_raw.png", dpi=150)
plt.show()

# -----------------------------
# 5. Dendrogram
# -----------------------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150)
plt.show()

# -----------------------------
# 6. Apply Hierarchical Clustering
# -----------------------------
model = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels = model.fit_predict(X_scaled)

# -----------------------------
# 7. Plot Clustered Data
# -----------------------------
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='rainbow', alpha=0.7)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segments (Hierarchical Clustering, K=5)")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=150)
plt.show()

# -----------------------------
# 8. Evaluate
# -----------------------------
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")
