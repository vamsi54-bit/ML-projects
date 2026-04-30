# 🔍 KMeans vs DBSCAN Clustering

A comparative analysis of two popular clustering algorithms — **KMeans** and **DBSCAN** — on a non-linear dataset.

---

## 📌 Objective

To understand how different clustering algorithms perform on complex data and identify which works better under different conditions.

---

## ⚙️ Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn

---

## 🧠 Algorithms Used

### 🔹 KMeans

* Centroid-based clustering
* Requires number of clusters (k)
* Works best for spherical clusters

### 🔹 DBSCAN

* Density-based clustering
* Detects arbitrary-shaped clusters
* Handles noise effectively

---

## 📊 Output

### KMeans Result

![KMeans](kmeans_output.png)

### DBSCAN Result

![DBSCAN](dbscan_output.png)

---

## 🧪 Performance Comparison

| Algorithm | Silhouette Score |
| --------- | ---------------- |
| KMeans    | 0.50             |
| DBSCAN    | 0.39             |

---

## 🔎 Observations

* KMeans performs well for well-separated clusters
* DBSCAN can detect complex shapes but may label noise points
* DBSCAN performance depends heavily on `eps` and `min_samples`
* No single algorithm works best for all datasets

---

## 🚀 How to Run

```bash id="z9q2pf"
pip install numpy matplotlib scikit-learn
python KMeans_vs_DBSCAN.py
```

---

## 📁 Project Structure

KMeans-vs-DBSCAN/
│── KMeans_vs_DBSCAN.py
│── README.md
│── kmeans_output.png
│── dbscan_output.png

---

## 🏁 Conclusion

Both algorithms have their strengths:

* **KMeans** is simple, fast, and efficient
* **DBSCAN** is powerful for irregular and noisy data

Choosing the right algorithm depends on the nature of the dataset.

---
