# 📊 KMeans Clustering

Implementation of the KMeans algorithm to group data into clusters based on similarity.

---

## 📌 Objective

To partition data into distinct clusters using centroid-based clustering.

---

## ⚙️ Technologies Used

* Python
* Scikit-learn
* Matplotlib

---

## 🧠 Algorithm

KMeans works by:

1. Selecting number of clusters (k)
2. Assigning data points to nearest centroid
3. Updating centroids iteratively

---

## 📊 Output

![KMeans Output](kmeans_output.png)

---

## 🚀 How to Run

```bash
pip install matplotlib scikit-learn
python kmeans_clustering.py
```

---

## 📁 Structure

KMeans-Clustering/
│── kmeans_clustering.py
│── README.md
│── kmeans_output.png

---

## 🔎 Observations

* Clusters are formed based on distance from centroids
* Works well for spherical data
* Sensitive to number of clusters (k)

---

## 🏁 Conclusion

KMeans is a simple and efficient clustering algorithm, widely used for pattern recognition and data segmentation.
