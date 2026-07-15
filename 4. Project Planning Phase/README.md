## 4. Project Planning Phase

### 4.1. Implementation Roadmap & Milestones
The project execution was planned across five distinct milestones over a multi-week timeline:

```
Milestone 1: Brainstorming & Data Gathering
|---> Problem definition, requirements specification, and dataset acquisition.

Milestone 2: Modular Restructuring & Preprocessing
|---> Designing preprocessing.py, configuring directories (dataset, model).

Milestone 3: ML Model Training & Comparison
|---> Implementing train.py, training estimators, comparing Macro F1, serializing model.

Milestone 4: Web Application Integration
|---> Writing app.py Flask backend, converting Bootstrap 5 UI pages, static styling.

Milestone 5: Verification & Sync
|---> Writing test_app.py unit tests, verifying locally, committing, and pushing to GitHub.
```

### 4.2. Work Breakdown Structure (WBS)
1. **Restructure Directory Layout**
   - Create `dataset/` and `model/` folders.
   - Relocate dataset and isolate clean variables.
2. **Preprocessing Pipeline Construction**
   - Write `preprocessing.py` containing data cleansing, StandardScaler fitting, and inference list transformations.
3. **ML Estimator Comparisons**
   - Implement `train.py` importing `preprocessing.py`.
   - Implement stratified splits and minority upsampling.
   - Benchmark Logistic Regression, Decision Trees, Random Forests, and XGBoost.
   - Save artifacts as `model/model.pkl` and `model/scaler.pkl` using joblib.
4. **Flask Backend & Frontend Refactoring**
   - Configure `app.py` model loading.
   - Refactor views with Bootstrap 5 templates.
   - Implement client side validation scripts (`main.js`) and animation stylesheets (`style.css`).
5. **Testing & Deployment**
   - Implement route checks and endpoint validations.
   - Perform automated `unittest` verification.
   - Synchronize with Git remote.

---