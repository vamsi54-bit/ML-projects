Isolation Forest on Wine Dataset
Overview
This project applies Isolation Forest to detect anomalies in the Wine dataset from scikit-learn.
Dataset
Wine dataset containing chemical properties of wines.
Method
Standardized the data
Applied Isolation Forest
Detected anomalies
Visualization
Scatter plot of:
Alcohol vs Malic Acid
Blue: Normal points
Red: Anomalies
How to Run
Install dependencies:
pip install pandas matplotlib seaborn scikit-learn
Run:
python isolation_forest_wine.py
Output
Visualization of anomalies
Printed anomalous rows
Conclusion
Isolation Forest identifies rare and unusual data points effectively in multi-dimensional data. 
