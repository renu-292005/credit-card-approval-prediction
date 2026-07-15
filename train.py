import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier

# Import custom preprocessing module
from preprocessing import clean_and_preprocess_data, fit_and_save_scaler

def train_and_evaluate():
    print("--- [TRAINING START] Starting Credit Card Approval Model Training & Comparison ---")
    
    # 1. Load the dataset from the new directory
    data_path = os.path.join('dataset', 'Application_Data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded. Shape: {df.shape}")
    
    # 2. Preprocess using preprocessing.py
    X, y = clean_and_preprocess_data(df)
    
    # Check target class distribution
    class_counts = y.value_counts().to_dict()
    print(f"Target Class Distribution (0=Rejected, 1=Approved): {class_counts}")
    
    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")
    
    # 4. Feature Scaling using preprocessing.py
    scaler_save_path = os.path.join('model', 'scaler.pkl')
    scaler, X_train_scaled = fit_and_save_scaler(X_train, scaler_save_path)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Address Class Imbalance via manual upsampling of the minority class in the training set
    feature_cols = list(X.columns)
    train_data = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_data['Status'] = y_train.values
    
    df_majority = train_data[train_data['Status'] == 1]
    df_minority = train_data[train_data['Status'] == 0]
    
    print(f"Before Upsampling - Training Set: Majority (1) = {len(df_majority)}, Minority (0) = {len(df_minority)}")
    
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,     # Sample with replacement
        n_samples=len(df_majority),    # Match majority class count
        random_state=42
    )
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    X_train_res = df_upsampled[feature_cols].values
    y_train_res = df_upsampled['Status'].values
    
    print(f"After Upsampling - Training Set: Balanced count = {len(X_train_res)}")
    
    # 6. Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=5, eval_metric='logloss')
    }
    
    results = {}
    
    # 7. Train and Evaluate Models
    for name, model in models.items():
        print(f"Training model: {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predict on scaled test set
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'model_object': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'confusion_matrix': cm
        }
        
        print(f"[{name}] Test Accuracy: {accuracy:.4f} | Macro F1-Score: {f1_macro:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print("-" * 50)
        
    # 8. Compare and Select the Best Model (using Macro F1-score to avoid imbalanced accuracy traps)
    best_model_name = max(results, key=lambda k: results[k]['f1_macro'])
    best_model_info = results[best_model_name]
    print(f"\n>>> Best Model Selected: {best_model_name} with Macro F1-score: {best_model_info['f1_macro']:.4f} <<<")
    
    # Save best model inside model/ directory using joblib
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'model.pkl')
    joblib.dump(best_model_info['model_object'], model_path)
    print(f"Best model saved to '{model_path}' using joblib")
    
    # Save the model name for Flask server load reference
    with open(os.path.join('model', 'best_model_name.txt'), 'w') as f:
        f.write(best_model_name)
        
    # 9. Generate Markdown Evaluation Report
    report_content = f"""# Model Evaluation & Performance Comparison Report

This report summarizes the performance of the machine learning models trained on the Credit Card Approval Prediction dataset.

## Target Class Imbalance
The dataset contains a significant class imbalance:
- **Approved (Class 1)**: {class_counts.get(1, 0)} cases ({class_counts.get(1, 0)/df.shape[0]*100:.2f}%)
- **Rejected (Class 0)**: {class_counts.get(0, 0)} cases ({class_counts.get(0, 0)/df.shape[0]*100:.2f}%)

To counter this imbalance during model training, the minority class (Rejected) was upsampled in the training set to match the majority class.

## Model Comparison

| Model | Test Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) | Macro F1-Score |
|---|---|---|---|---|---|
"""
    for name, metrics in results.items():
        report_content += f"| **{name}** | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | **{metrics['f1_macro']:.4f}** |\n"
        
    report_content += f"""
## Selection
Based on the **Macro F1-Score** (which averages performance across both classes to ensure minority class recall is high), the best-performing model is **{best_model_name}**.

The trained model has been saved as `model/model.pkl` along with its StandardScaler as `model/scaler.pkl` to process inputs for web applications.

### Best Model Confusion Matrix:
- True Negative (Correct Rejections): {best_model_info['confusion_matrix'][0][0]}
- False Positive (Incorrect Approvals): {best_model_info['confusion_matrix'][0][1]}
- False Negative (Incorrect Rejections): {best_model_info['confusion_matrix'][1][0]}
- True Positive (Correct Approvals): {best_model_info['confusion_matrix'][1][1]}
"""
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report_content)
    print("Evaluation report saved to 'evaluation_report.md'")
    print("--- [TRAINING END] Model Training Complete ---")

if __name__ == '__main__':
    train_and_evaluate()
