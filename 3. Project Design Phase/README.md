## 3. Project Design Phase

### 3.1. System Architecture
The application follows a standard Three-Tier Model View Controller (MVC) architecture, optimized for lightweight ML microservices:

```
+-----------------------------------------------------------------------+
|                              CLIENT TIER                              |
|   Bootstrap 5 HTML/CSS Pages (Home, Evaluation Form, Results Card)    |
|   Client Validation (main.js) & UI Animations                         |
+-----------------------------------+-----------------------------------+
                                    | HTTP POST /get_prediction
                                    v
+-----------------------------------+-----------------------------------+
|                            APPLICATION TIER                           |
|   Flask Backend Controller (app.py)                                   |
|   Validation Filters & JSON Parsers                                   |
+-------------------+---------------+-------------------+---------------+
                    |                                   |
                    v Ingest inputs                     v Load artifacts
+-------------------+---------------+   +---------------+---------------+
|           LOGIC TIER              |   |          MODEL TIER           |
|   preprocessing.py                |   |   model/model.pkl (XGBoost)   |
|   StandardScaler (scaler.pkl)     |   |   model/scaler.pkl            |
+-------------------+---------------+   +---------------+---------------+
                    |                                   |
                    +-----------------+-----------------+
                                      | Inference Predict
                                      v
                             [Binary Classification Output]
```

### 3.2. Data Flow Diagram (DFD - Level 1)
```
  [ Applicant ] --(1. Enter Metrics)--> [ Form Validation (JS) ]
                                                   |
                                            (Valid Request)
                                                   v
[ Result Render ] <--(4. Response Page)-- [ Flask Backend Route ] 
                                                   |
                                            (Parse Inputs)
                                                   v
   [ XGBoost Model ] <--(3. Scale & Predict)-- [ preprocessing.py ]
```

### 3.3. Entity Relationship Diagram (Conceptual Schema)
```
+--------------------------+             +--------------------------+
|        APPLICANT         |             |      CREDIT_DECISION     |
+--------------------------+             +--------------------------+
| PK  Applicant_ID         |             | PK  Decision_ID          |
|     Applicant_Age        |1 -------- 1 | FK  Applicant_ID         |
|     Total_Income         |             |     Decision_Status      |
|     Total_Bad_Debt       |             |     Debt_to_Income_Ratio |
|     Total_Good_Debt      |             |     Model_Fused_F1       |
+--------------------------+             +--------------------------+
```

---