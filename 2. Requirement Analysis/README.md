## 2. Requirement Analysis

### 2.1. System Requirements

#### Hardware Requirements
- **Development/Training System**:
  - Processor: Intel Core i5/i7 (10th Generation or higher) or AMD Ryzen 5/7 (minimum 4 cores, 8 threads).
  - RAM: 8 GB DDR4 (16 GB recommended for model tuning).
  - Storage: 256 GB Solid State Drive (SSD).
- **Deployment Server (Local/Staging)**:
  - Processor: Single Core vCPU.
  - RAM: 1 GB (suitable for Flask routing).
  - Storage: 10 GB free space.

#### Software Requirements
- **Operating System**: Windows 11 / Linux (Ubuntu 22.04 LTS).
- **Environment**: Python 3.12.10.
- **Package Manager**: Pip 25.0.1.
- **Core Libraries**:
  - `Flask==3.0.3` (Web application container)
  - `pandas==2.2.2` (Data ingestion and structure manipulation)
  - `numpy==1.26.4` (Mathematical arrays processing)
  - `scikit-learn==1.5.0` (StandardScaler, train-test split, linear models, metrics)
  - `xgboost==2.0.3` (Extreme Gradient Boosting model framework)
  - `joblib==1.5.3` (Serialized model storage)

### 2.2. Functional Requirements
- **FR1: Profile Ingestion**: The system must ingest four key applicant variables: Annual Income ($), Age (Years), Outstanding Debt ($), and FICO Credit Score (300-850).
- **FR2: Dynamic Input Validation**: Validates client-side and server-side range thresholds: Age $\in [18, 100]$, Credit Score $\in [300, 850]$, Income and Debt $\ge 0$.
- **FR3: Preprocessing Integration**: Automatically maps the FICO range to a 0-30 scale representing the `Total_Good_Debt` feature, wraps raw lists in pandas DataFrames to prevent shape mismatches, and normalizes values using a serialized `StandardScaler` model.
- **FR4: Automated Inference**: Queries the serialized model (`model.pkl`) to calculate binary output status (`1` = Approved, `0` = Rejected).
- **FR5: Risk Reporting**: Renders a results screen summarizing applicant parameters and computing secondary metrics like the Debt-to-Income (DTI) ratio.

### 2.3. Non-Functional Requirements
- **NFR1: Performance & Latency**: Inference and response rendering must execute within 50 milliseconds under standard workloads.
- **NFR2: Security**: Inputs must be sanitized to protect against cross-site scripting (XSS) and code injections.
- **NFR3: UI Responsiveness**: The frontend must adapt seamlessly across desktop screens, tablets, and mobile devices using Bootstrap grid systems.
- **NFR4: Reliability**: Fallback rule-based execution must trigger if model serialization artifacts are missing to maintain server uptime.

### 2.4. Data Requirements
The model was trained on `dataset/Application_Data.csv`, consisting of 25,128 records and 21 columns. The raw dataset attributes include:
- `Total_Income`: Float, annual income of the applicant.
- `Applicant_Age`: Integer, age of the applicant.
- `Total_Bad_Debt`: Float, outstanding bad debt records.
- `Total_Good_Debt`: Float, active credit records in good standing.
- `Status`: Binary integer, target class variable where `1` represents approved applications (25,007 rows) and `0` represents rejected applications (121 rows).

---