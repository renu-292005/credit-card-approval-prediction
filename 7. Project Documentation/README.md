## 7. Project Documentation

### 7.1. Quick Start Installation Guide
1. Clone the repository:
   ```bash
   git clone https://github.com/renu-292005/credit-card-approval-prediction.git
   cd credit-card-approval-prediction
   ```
2. Set up requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Retrain the classifier models:
   ```bash
   python train.py
   ```
4. Run validation tests:
   ```bash
   python -m unittest test_app.py
   ```
5. Launch the local application:
   ```bash
   python app.py
   ```

### 7.2. Model Metrics Comparison Summary
The benchmark comparison conducted in the planning phase resulted in the following evaluation log (saved in `evaluation_report.md`):

```
| Model               | Test Accuracy | Precision (Approved) | Recall (Approved) | F1-Score (Approved) | Macro F1-Score |
|---------------------|---------------|----------------------|-------------------|---------------------|----------------|
| Logistic Regression | 99.20%        | 100.00%              | 99.20%            | 99.60%              | 0.7707         |
| Decision Tree       | 99.28%        | 99.98%               | 99.30%            | 99.64%              | 0.7787         |
| Random Forest       | 99.80%        | 99.98%               | 99.82%            | 99.90%              | 0.9102         |
| **XGBoost**         | **99.96%**    | **99.98%**           | **99.98%**        | **99.98%**          | **0.9791**     |
```

---