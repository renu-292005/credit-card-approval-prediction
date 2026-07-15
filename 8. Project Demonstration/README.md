## 8. Project Demonstration

### 8.1. Model In-Action Walkthrough
The application features a dark glassmorphic user interface:
1. **Landing Dashboard**: Analysts land on `http://127.0.0.1:5000/`. The page displays system capabilities and features dynamic active model metrics (`XGBoost Classifier - Macro F1: 97.91%`).
2. **Form Entry**: Clicking **Run Evaluation** loads the applicant profile forms screen.
3. **Validation Actions**: Entering invalid data (e.g. credit score 250) triggers client-side bootstrap validation states. Entering valid metrics and clicking submit displays a loader spinner inside the button.
4. **Eligibility Card**: Renders the decision status screen:
   - *Approved Case*: Displays a green checkmark alert, summarizes input parameters, and suggests positive actions (e.g., maintaining low revolving utilization).
   - *Declined Case*: Displays a crimson risk alert, indicates risk flags, and suggests credit building strategies (DTI reductions).

### 8.2. Advantages
- **Robust Imbalance Handling**: By oversampling minority training targets, the model maintains high sensitivity towards risk markers instead of blind approval predictions.
- **Performance**: Standardized microservice design achieves fast execution latency suitable for high-throughput corporate endpoints.
- **Aesthetic UI**: Transitioning to Bootstrap 5 glassmorphic layouts delivers a premium visual experience for banking agents.

### 8.3. Limitations & Future Scope
- **Feature Sparsity**: The current classifier relies on four metrics. Future iterations will include categorical variables (e.g. `Education_Type`, `Housing_Type`) in the preprocessing script.
- **Explainable AI (XAI)**: Future milestones will integrate SHAP or LIME frameworks to provide transparency on feature weight contributions during model decision predictions.
- **REST API Endpoint**: Exposing predictive interfaces as standardized secure JSON API web endpoints to support third-party application integration.