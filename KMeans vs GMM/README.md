# KMeans vs Gaussian Mixture Model (GMM)

## 📌 Overview

This project compares two popular clustering algorithms:

* KMeans (Centroid-based clustering)
* Gaussian Mixture Model (Probabilistic clustering)

The goal is to understand how each algorithm groups data and how their approaches differ.

---

## 📊 Dataset

Iris dataset from Scikit-learn

**Features used:**

* All 4 features for training
* First 2 features for visualization

---

## ⚙️ Techniques Used

### 1. Data Preprocessing

* Standardization using `StandardScaler`
* Ensures equal contribution of all features

---

### 2. KMeans

* Divides data into K clusters
* Uses distance to assign points
* Produces hard cluster labels

---

### 3. Gaussian Mixture Model (GMM)

* Assumes data comes from Gaussian distributions
* Uses probabilities for cluster assignment
* Produces soft clustering

---

## 📈 Visualization

Two plots are generated:

* **Left Plot:** KMeans clustering
* **Right Plot:** GMM clustering

Each point is colored based on cluster assignment.

---

## 🧠 Key Observations

* KMeans creates rigid cluster boundaries
* GMM provides flexible clustering
* GMM can handle overlapping clusters better
* On simple datasets, both may give similar results

---

## 🔍 GMM Probabilities

GMM also outputs probabilities:

Example:

```
[0.01, 0.98, 0.01]
```

Meaning:

* 98% chance of belonging to cluster 1
* Small probability for others

---

## ▶️ How to Run

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Run:

```bash
python kmeans_vs_gmm.py
```

---

## 🛠️ Tools & Libraries

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🎯 Learning Outcome

This project demonstrates:

* Difference between hard and soft clustering
* Behavior of centroid-based vs probabilistic models
* Importance of data visualization in clustering

---

## 🚀 Conclusion

KMeans is simple and fast but limited to spherical clusters.

GMM is more flexible and can model complex data distributions, making it more suitable for real-world scenarios where clusters may overlap.
