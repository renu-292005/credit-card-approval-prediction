## 5. Project Development Phase

### 5.1. Data Cleansing & Scaler Construction (`preprocessing.py`)
Data cleansing and normalization is isolated into a reusable python module.
- Whitespace is stripped from DataFrame columns.
- Feature columns are mapped: `Total_Income`, `Applicant_Age`, `Total_Bad_Debt`, and `Total_Good_Debt`.
- A `StandardScaler` is fitted on train features and saved to `model/scaler.pkl` using `joblib`.
- For inference, raw feature lists are mapped to a structured pandas DataFrame containing matching feature headers. This preserves name schemas and prevents training warnings:

```python
# Code snippet from preprocessing.py
def load_scaler_and_transform(raw_features, load_path):
    scaler = joblib.load(load_path)
    feature_cols = ['Total_Income', 'Applicant_Age', 'Total_Bad_Debt', 'Total_Good_Debt']
    df_features = pd.DataFrame([raw_features], columns=feature_cols)
    return scaler.transform(df_features)
```

### 5.2. Model Benchmarking and Comparison (`train.py`)
To prevent overfitting on the majority class (Approved = 25007), we apply manual oversampling to class 0 (Rejected = 97 in training split) to balance the partitions.

Four algorithms are trained on the resampled training partition and validated against a stratified 20% test partition:
- **Logistic Regression**: Serves as a baseline model.
- **Decision Tree**: Evaluates non-linear feature splits.
- **Random Forest**: Aggregates bootstrap tree estimators to reduce variance.
- **XGBoost**: Employs sequential gradient boosting to optimize classification error.

### 5.3. Web Service Component (`app.py`)
Exposes three Flask controllers:
- `home()`: Loads `model/best_model_name.txt` and renders `home.html` landing page.
- `predict_form()`: Renders `index.html` applicant inputs page.
- `get_prediction()`: Extracts POST parameters, validates bounds, scales parameters, evaluates model prediction, calculates the debt-to-income ratio, and returns `result.html` status.

---