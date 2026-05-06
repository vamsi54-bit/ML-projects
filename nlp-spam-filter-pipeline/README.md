# 📧 Spam Detection using Machine Learning

## 🚀 Overview

This project builds a spam detection system using Natural Language Processing (NLP) techniques.
It classifies SMS messages as **Spam** or **Ham (Not Spam)** using a machine learning pipeline.

---

## 🧠 Methodology

### 1. Data Preprocessing

* Dataset: SMS Spam Collection
* Cleaned dataset by selecting relevant columns
* Converted labels:

  * `ham → 0`
  * `spam → 1`

---

### 2. Text Vectorization

Used **TF-IDF (Term Frequency - Inverse Document Frequency)** to convert text into numerical features.

---

### 3. Model Training

* Algorithm: Multinomial Naive Bayes
* Train/Test Split: 80/20

---

### 4. Evaluation

* Metric: Accuracy
* Achieved accuracy: **~96%**

---

## 📊 Results

The model effectively distinguishes between spam and non-spam messages with high accuracy.

---

## 📁 Project Structure

```
spam-classifier/
│
├── main.py
├── spam.csv
└── README.md
```

---

## ⚙️ Installation

Install dependencies:

```
pip install pandas scikit-learn
```

---

## ▶️ Run the Project

```
python main.py
```

---

## 🔮 Future Improvements

* Add clustering (KMeans) for unlabeled data
* Implement semi-supervised learning
* Build a Streamlit web app for real-time prediction
* Improve model with advanced NLP techniques

---

## 💡 Key Learnings

* Text preprocessing and vectorization
* Supervised machine learning workflow
* Real-world dataset handling

---

## 📌 Conclusion

This project demonstrates how machine learning can be applied to solve real-world problems like spam detection efficiently.
