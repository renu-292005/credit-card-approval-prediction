import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess_data(df):
    """
    Cleans column names, handles missing values, and extracts the key 
    features and target columns.
    
    Features selected:
    - Total_Income
    - Applicant_Age
    - Total_Bad_Debt
    - Total_Good_Debt
    
    Target:
    - Status
    """
    # Clean whitespace from column headers
    df.columns = df.columns.str.strip()
    
    feature_cols = ['Total_Income', 'Applicant_Age', 'Total_Bad_Debt', 'Total_Good_Debt']
    target_col = 'Status'
    
    # Check that expected columns exist in df
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset. Available: {list(df.columns)}")
            
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Missing values imputation (with median)
    X = X.fillna(X.median())
    
    return X, y

def fit_and_save_scaler(X_train, save_path):
    """
    Fits a StandardScaler on the training data and serialises it to disk using joblib.
    Returns the fitted scaler and scaled training features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save using joblib
    joblib.dump(scaler, save_path)
        
    print(f"[PREPROCESSING] Scaler fitted and saved to '{save_path}'")
    return scaler, X_train_scaled

def load_scaler_and_transform(raw_features, load_path):
    """
    Loads the serialised StandardScaler from disk using joblib and scales a raw input list.
    Handles pandas DataFrame conversion internally to preserve feature name schemas
    and prevent sklearn UserWarnings.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Scaler file not found at {load_path}")
        
    # Load using joblib
    scaler = joblib.load(load_path)
        
    # Map input list to pandas DataFrame maintaining original column headers
    feature_cols = ['Total_Income', 'Applicant_Age', 'Total_Bad_Debt', 'Total_Good_Debt']
    df_features = pd.DataFrame([raw_features], columns=feature_cols)
    
    # Scale features
    return scaler.transform(df_features)
