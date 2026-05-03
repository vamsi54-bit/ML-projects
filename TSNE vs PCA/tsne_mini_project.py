import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# ── 1. Load Data ─────────────────────────────────────────────
data = load_iris()
X    = data.data        # (150, 4)
y    = data.target      # 0=setosa, 1=versicolor, 2=virginica

print("Shape   :", X.shape)
print("Classes :", data.target_names)

# ── 2. Scale ─────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Fit t-SNE ─────────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=30,
            init='pca', random_state=42, max_iter=1000)
X_2d = tsne.fit_transform(X_scaled)

print("KL Divergence :", round(tsne.kl_divergence_, 4))
print("Iterations ran:", tsne.n_iter_without_progress)

# ── 4. DataFrame for plotting ─────────────────────────────────
df = pd.DataFrame(X_2d, columns=['dim1', 'dim2'])
df['target'] = y
df['label']  = df['target'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# ── 5. t-SNE 2D scatter plot ──────────────────────────────────
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='dim1', y='dim2',
                hue='label', palette='deep', s=60, alpha=0.85)
plt.title('t-SNE 2D Projection — Iris Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('tsne_projection.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: tsne_projection.png")

# ── 6. Perplexity comparison ──────────────────────────────────
perplexities = [5, 15, 30, 50]
fig, axes    = plt.subplots(1, 4, figsize=(18, 4))

for ax, perp in zip(axes, perplexities):
    t = TSNE(n_components=2, perplexity=perp,
             init='pca', random_state=42, max_iter=1000)
    X_p = t.fit_transform(X_scaled)
    for cls in range(3):
        mask = y == cls
        ax.scatter(X_p[mask, 0], X_p[mask, 1],
                   label=data.target_names[cls], s=30, alpha=0.8)
    ax.set_title(f'perplexity = {perp}')
    ax.set_xticks([]); ax.set_yticks([])

axes[0].legend(fontsize=8)
plt.suptitle('Effect of Perplexity on t-SNE Layout', fontsize=13)
plt.tight_layout()
plt.savefig('tsne_perplexity_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved: tsne_perplexity_comparison.png")

# ── 7. Summary ───────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Dataset        : {X.shape[0]} samples, {X.shape[1]} features")
print(f"Perplexity     : 30")
print(f"KL Divergence  : {round(tsne.kl_divergence_, 4)}")
print(f"Iterations     : {tsne.n_iter_without_progress}")
