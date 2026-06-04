Network Intrusion Detection Using Machine Learning
Comparative Analysis of Decision Tree, Random Forest, and LightGBM on the UNSW-NB15 Dataset








Project Overview

Cyberattacks have become increasingly sophisticated, making traditional rule-based security systems less effective. This project develops a Network Intrusion Detection System (NIDS) using Machine Learning techniques to identify and classify malicious network traffic.

The project utilizes the UNSW-NB15 cybersecurity dataset and performs a comparative analysis of three machine learning algorithms:

Decision Tree
Random Forest
LightGBM

The objective is to determine which model provides the best performance for multiclass intrusion detection while identifying the most influential network traffic features.

Problem Statement

Modern networks generate massive amounts of traffic every second. Detecting malicious activities manually is impractical and error-prone.

This project aims to:

Detect malicious network traffic automatically.
Classify different attack categories.
Compare multiple machine learning models.
Identify key features contributing to attack detection.
Evaluate model performance using multiple metrics.
Dataset

Dataset Used:

UNSW-NB15 Network Intrusion Detection Dataset

Dataset Characteristics:

Attribute	Value
Total Samples	82,332
Features	42
Classes	10
Problem Type	Multiclass Classification
Attack Categories
Normal
Generic
Exploits
Fuzzers
DoS
Reconnaissance
Analysis
Backdoor
Shellcode
Worms
Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
LightGBM
Joblib
Project Workflow
Data Collection
       ↓
Data Understanding
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Feature Encoding
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Performance Comparison
       ↓
Feature Importance Analysis
       ↓
Model Deployment Preparation
Data Preprocessing

The following preprocessing steps were performed:

Feature Selection

Removed:

id
label

Target Variable:

attack_cat
Encoding

Categorical features:

proto
service
state

were encoded using Label Encoding.

Train-Test Split
80% Training Data
20% Testing Data

Stratified sampling was applied to preserve class distribution.

Machine Learning Models
1. Decision Tree

A tree-based supervised learning algorithm used as a baseline model.

2. Random Forest

An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

3. LightGBM

A gradient boosting framework designed for high efficiency and performance on large datasets.

Evaluation Metrics

The following metrics were used:

Accuracy
Precision
Recall
F1 Score
Classification Report
Confusion Matrix
Results
Model	Accuracy	Precision	Recall	F1 Score
Decision Tree	85.02%	84.91%	85.02%	84.86%
Random Forest	86.86%	86.42%	86.86%	86.31%
LightGBM	83.92%	84.34%	83.92%	84.06%
Best Performing Model

🏆 Random Forest

Random Forest achieved the highest overall performance across all evaluation metrics and demonstrated strong generalization capabilities on unseen network traffic data.

Feature Importance Analysis

Top features identified by the Random Forest model:

sbytes
smean
ct_dst_sport_ltm
ct_dst_src_ltm
sload
ct_srv_dst
service
rate
ct_srv_src
dur

These features were found to have the greatest influence on attack classification.

Confusion Matrix

A confusion matrix was generated for the best-performing model to visualize prediction performance across all attack categories.

The model demonstrated strong classification capability for major attack categories such as:

Normal
Generic
Exploits
Fuzzers
Reconnaissance

Performance on rare attack classes such as Worms and Shellcode was limited due to class imbalance within the dataset.

Key Findings
Random Forest outperformed Decision Tree and LightGBM.
Network traffic volume and connection behavior features were highly influential.
Class imbalance significantly affected minority attack categories.
Ensemble methods provided more robust performance than a single Decision Tree.
Machine Learning can effectively support automated intrusion detection systems.
Future Improvements

Potential enhancements include:

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Handling class imbalance using SMOTE.
Explainable AI using SHAP.
Real-time deployment using Streamlit.
Deep Learning-based intrusion detection models.
Hybrid ensemble approaches.
Project Structure
Network-Intrusion-Detection/
│
├── UNSW_NB15_training-set.csv
├── intrusion_detection.py
├── best_intrusion_model.pkl
├── README.md
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
└── requirements.txt
Installation

Clone the repository:



Install dependencies:

pip install -r requirements.txt

Run the project:

python intrusion_detection.py
Conclusion

This project demonstrates the application of machine learning techniques for cybersecurity and network intrusion detection. Through comparative analysis of Decision Tree, Random Forest, and LightGBM models, Random Forest emerged as the most effective approach for multiclass attack classification on the UNSW-NB15 dataset.

The findings highlight the potential of machine learning to enhance automated threat detection and support modern cybersecurity systems.

Author

Vamsi

Machine Learning • Data Science • Cybersecurity Enthusiast 🚀        
