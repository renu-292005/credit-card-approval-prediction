# Model Evaluation & Performance Comparison Report

This report summarizes the performance of the machine learning models trained on the Credit Card Approval Prediction dataset.

## Target Class Imbalance
The dataset contains a significant class imbalance:
- **Approved (Class 1)**: 25007 cases (99.52%)
- **Rejected (Class 0)**: 121 cases (0.48%)

To counter this imbalance during model training, the minority class (Rejected) was upsampled in the training set to match the majority class.

## Model Comparison

| Model | Test Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) | Macro F1-Score |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.9920 | 1.0000 | 0.9920 | 0.9960 | **0.7707** |
| **Decision Tree** | 0.9928 | 0.9998 | 0.9930 | 0.9964 | **0.7787** |
| **Random Forest** | 0.9980 | 0.9998 | 0.9982 | 0.9990 | **0.9102** |
| **XGBoost** | 0.9996 | 0.9998 | 0.9998 | 0.9998 | **0.9791** |

## Selection
Based on the **Macro F1-Score** (which averages performance across both classes to ensure minority class recall is high), the best-performing model is **XGBoost**.

The trained model has been saved as `model/model.pkl` along with its StandardScaler as `model/scaler.pkl` to process inputs for web applications.

### Best Model Confusion Matrix:
- True Negative (Correct Rejections): 23
- False Positive (Incorrect Approvals): 1
- False Negative (Incorrect Rejections): 1
- True Positive (Correct Approvals): 5001
