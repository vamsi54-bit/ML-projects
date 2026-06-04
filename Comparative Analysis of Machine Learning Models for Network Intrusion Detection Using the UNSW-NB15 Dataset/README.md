Project Objective

Develop and compare multiple machine learning models for detecting cyberattacks in network traffic and determine which model performs best on the UNSW-NB15 dataset.

Models to Compare
Model 1

Decision Tree

Baseline model
Easy to understand
Model 2

Random Forest

Ensemble method
Usually much better than Decision Trees
Model 3

LightGBM

Fast
Powerful
Usually among the best performers
Input

Network traffic features:

Duration
Protocol
Service
State
Source Bytes
Destination Bytes
Source Packets
Destination Packets
...
Output
Prediction: Normal

or

Prediction: Reconnaissance

or

Prediction: Exploit

depending on whether you do binary or multiclass classification.

Workflow
UNSW-NB15 Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Encoding
        ↓
Train/Test Split
        ↓
Decision Tree Training
        ↓
Random Forest Training
        ↓
LightGBM Training
        ↓
Performance Comparison
        ↓
Best Model Selection
        ↓
Feature Importance Analysis
Evaluation Metrics

Compare:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

Example table in your report:

Model	Accuracy	Precision	Recall	F1
Decision Tree	89%	88%	87%	87%
Random Forest	94%	93%	94%	93%
LightGBM	96%	95%	96%	95%

(Example numbers only.)

Expected Conclusion

Something like:

LightGBM achieved the highest accuracy and F1-score while maintaining efficient training time. Random Forest provided strong performance, whereas Decision Tree served as a useful baseline model.
