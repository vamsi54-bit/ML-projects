# PCA vs t-SNE Visualization on Iris Dataset

## Overview
This project compares two dimensionality reduction techniques:

- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)

The goal is to visualize high-dimensional data in 2D and observe how each technique represents class separation.

---

## Dataset
Iris Dataset from Scikit-learn

Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Classes:
- Setosa
- Versicolor
- Virginica

---

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Methods Applied

### 1. Data Standardization
Feature scaling using StandardScaler.

### 2. PCA
Reduces dimensions while preserving maximum variance.

### 3. t-SNE
Reduces dimensions by preserving local relationships between data points.

---

## Results

### PCA
- Preserves global structure
- Fast dimensionality reduction
- Explained variance retained is displayed

### t-SNE
- Produces tighter visual clusters
- Better local separation
- Useful for visualization

---

## Key Observation
PCA provides a linear projection of the data.

t-SNE creates clearer cluster separation by focusing on neighborhood relationships.

---

## How to Run

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
