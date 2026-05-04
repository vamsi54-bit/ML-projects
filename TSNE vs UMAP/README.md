# t-SNE vs UMAP Visualization on Iris Dataset

## 📌 Overview

This project compares two popular non-linear dimensionality reduction techniques:

* t-SNE (t-distributed Stochastic Neighbor Embedding)
* UMAP (Uniform Manifold Approximation and Projection)

The goal is to visualize high-dimensional data in 2D and understand how each method represents class structure.

---

## 📊 Dataset

The Iris dataset is used, which contains:

**Features:**

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

**Classes:**

* Setosa (0)
* Versicolor (1)
* Virginica (2)

---

## ⚙️ Techniques Used

### 1. Data Preprocessing

* Standardization using `StandardScaler`
* Ensures all features contribute equally

---

### 2. t-SNE

* Focuses on preserving **local relationships**
* Produces tight and visually separated clusters
* Sensitive to parameters like `perplexity`

---

### 3. UMAP

* Preserves both **local and global structure**
* Faster than t-SNE
* Controlled using `n_neighbors` and `min_dist`

---

## 📈 Results

### 🔹 t-SNE

* Produces very tight clusters
* Strong separation between classes
* May distort global distances

### 🔹 UMAP

* Clearly separates class 0
* Shows partial overlap between class 1 and 2
* Reflects more realistic data structure

---

## 🧠 Key Observations

* Class 0 is easily separable in both methods
* Classes 1 and 2 show similarity and partial overlap
* t-SNE exaggerates separation
* UMAP provides a more balanced and interpretable layout

---

## ▶️ How to Run

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn umap-learn
```

Run the script:

```bash
python filename.py
```

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* UMAP

---

## 🎯 Learning Outcome

This project demonstrates:

* Dimensionality reduction techniques
* Differences between t-SNE and UMAP
* Importance of visualization in machine learning
* Effect of parameters on model behavior

---

## 🚀 Conclusion

t-SNE and UMAP are powerful visualization tools, but they serve different purposes:

* t-SNE is useful for exploring local cluster structure
* UMAP provides a better balance between local and global relationships

Understanding these differences is essential for choosing the right technique in real-world problems.
