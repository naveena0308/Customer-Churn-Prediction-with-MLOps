# Comprehensive Interview Preparation Guide
## Customer Churn Prediction with MLOps

---

## 1. Executive Summary & The "Elevator Pitch"

When an interviewer asks: **"Tell me about this Telecom Churn Prediction project on your resume."**

### The 60-Second Answer (STAR Framework)
> "In this project, I built an end-to-end, production-grade MLOps pipeline to predict customer churn on over 7,000 real-world telecom records. 
> 
> **Situation & Task:** Telecom companies suffer substantial recurring revenue loss from customer churn. Retention campaigns require early, calibrated intervention rather than blunt, reactive discounts.
> 
> **Action:** I architected a leak-free ML pipeline. I split the dataset 70/15/15 into stratified train, validation, and holdout test splits *before* fitting preprocessors. I engineered domain-specific features—such as tenure grouping, charges-to-tenure ratios, and total active service counts. I systematically evaluated Random Forest, XGBoost, and Logistic Regression with cross-validated grid search, class-weight balancing, and Precision-Recall threshold tuning. I tracked experiments and versioned models in MLflow, and deployed the winning model via a containerized FastAPI service with multi-stage Docker builds.
> 
> **Result:** Logistic Regression emerged as the production champion with an ROC AUC of 0.845, a baseline F1 of 0.61, and a calibrated F1 of 0.63+ after PR threshold optimization. The system supports sub-50ms single-record inferences and batch scoring of 7,000+ customers, fully validated by an automated GitHub Actions CI suite."

---

## 2. Deep-Dive Q&A: Technical & Architecture

### Q1: Why did you choose Logistic Regression over complex ensembles like XGBoost and Random Forest?
**What the interviewer is testing:** Do you blindly pick deep/ensemble models, or do you make disciplined, metric-driven, and cost-aware engineering decisions?

**Ideal Answer:**
> "I evaluated all three models using 3-fold cross-validated grid searches on the validation set:
> - **Logistic Regression**: Val ROC AUC `0.8471`, Holdout Test ROC AUC `0.8418`, Optimal F1 `0.6357`
> - **XGBoost**: Val ROC AUC `0.8470`, Holdout Test ROC AUC `0.8402`, Optimal F1 `0.6512`
> - **Random Forest**: Val ROC AUC `0.8376`, Holdout Test ROC AUC `0.8351`, Optimal F1 `0.6337`
> 
> While XGBoost achieved a marginal gain in peak validation F1 (+0.015), Logistic Regression matched it on ROC AUC (0.845) while offering three decisive operational advantages:
> 1. **Inference Latency & Compute Footprint:** Sub-millisecond scoring per record with negligible CPU/memory overhead.
> 2. **Interpretability & Compliance:** Telecommunications and customer success teams need clear odds ratios (e.g., month-to-month contracts and fiber optic service without tech support drive high churn odds) to justify retention interventions.
> 3. **Risk of Overfitting:** With ~7,000 records, linear models with $L_2$ regularization generalize reliably with zero tree-drift over subtle distribution variations."

---

### Q2: What was your strategy for class imbalance, and why didn't you just use SMOTE?
**What the interviewer is testing:** Imbalanced data handling and practical understanding of synthetic sampling pitfalls.

**Ideal Answer:**
> "The Telco churn dataset has a ~73% stay vs ~27% churn imbalance (roughly 3:1).
> 
> Instead of synthetic oversampling like SMOTE, which often creates synthetic artifacts in high-dimensional or mixed categorical feature spaces and alters the base rate probabilities, I handled imbalance through **algorithmic weighting and threshold calibration**:
> 1. **Cost-Sensitive Learning:** I configured `class_weight='balanced'` in Logistic Regression and Random Forest, and set `scale_pos_weight = neg / pos` (~2.76) in XGBoost. This penalizes false negatives proportionally during loss calculation.
> 2. **Post-Hoc Threshold Optimization:** Standard classification uses a naive 0.5 probability cutoff. I scanned the Precision-Recall curve on the validation set to discover the cutoff that directly maximizes the F1 score (shifting the operating threshold to ~0.59).
> 3. **Benefit:** This preserves true data distributions, avoids synthetic noise, and calibrates the operational trade-off between precision and recall based on business retention costs."

---

### Q3: How did you ensure zero data leakage between training and inference?
**What the interviewer is testing:** Real-world ML rigor, pipeline cleanliness, and avoiding train-serving skew.

**Ideal Answer:**
> "Data leakage is the #1 reason models succeed in notebooks and fail in production. I enforced three architectural boundaries:
> 1. **Split-Before-Transform:** The dataset was split into train (70%), validation (15%), and holdout test (15%) splits *before* any transformation was calculated.
> 2. **Stateful `DataPreprocessor` Object:** The preprocessor learns population parameters—specifically median values for `TotalCharges`, `MonthlyCharges`, and `tenure`, as well as categorical `LabelEncoder` mappings—strictly on `X_train`.
> 3. **Single-Sample Inference Consistency:** During real-time API scoring, when a single customer payload arrives, features like `is_high_value` (which compares against median charges and median tenure) compare against the *persisted training set medians*, rather than attempting to calculate a meaningless 1-record median.
> 4. **Handling Unseen Labels:** The encoder contains fallback logic for unseen categorical levels, mapping them to 0 rather than raising a runtime exception."

---

### Q4: Walk me through your Feature Engineering. What features did you craft and why?
**What the interviewer is testing:** Domain knowledge and ability to extract non-linear signal.

**Ideal Answer:**
> "I engineered 5 domain-informed features:
> 1. **`tenure_group`**: Binned continuous customer tenure into lifecycle buckets (`0-12m`, `12-24m`, `24-48m`, `48-72m`, `72m+`). Churn risk is non-linear; the first 12 months exhibit the highest customer vulnerability.
> 2. **`charges_per_month`**: Calculated as `TotalCharges / (tenure + 1)`. Captures whether recent pricing or plan upgrades spiked relative to historical billings.
> 3. **`total_services`**: Sum of active digital services (Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies). Customers with higher service adoption have greater switching friction and churn significantly less.
> 4. **`is_month_to_month`**: Binary flag extracting whether the contract is rolling month-to-month versus annual commitments. This proved to be one of the single highest predictors of churn.
> 5. **`is_high_value`**: Flag identifying high-tenure, high-monthly-bill customers based on training medians. These represent priority retention targets where churn yields the greatest revenue loss."

---

### Q5: How did you implement MLflow, and what role does it play in your MLOps pipeline?
**What the interviewer is testing:** Production MLOps tooling, experiment tracking, and model governance.

**Ideal Answer:**
> "I integrated MLflow across two key areas:
> 1. **Experiment Tracking:** During hyperparameter tuning in `ModelTrainer`, each candidate model run logs:
>    - Hyperparameters (e.g., regularization `C`, tree depth, learning rate)
>    - Validation metrics (ROC AUC, Precision, Recall, F1 at default 0.5, and F1 at optimal threshold)
>    - Out-of-sample holdout test scores
>    - Model artifacts serialized via cloudpickle.
> 2. **Model Registry & Versioning:** Every successful run logs the trained estimator under a registered model name (`churn_prediction_logisticregression`, `churn_prediction_xgboost`, etc.). MLflow increments versions automatically, creating an auditable lineage from data split seed to production container.
> 3. **Infrastructure:** In `docker-compose.yml`, I orchestrated a dedicated MLflow tracking server backed by SQLite (`mlflow.db`) and persistent volume mounts (`mlruns/`), exposing the UI at port `5000` alongside the FastAPI container."

---

### Q6: How is the application deployed, and how does FastAPI handle inference efficiently?
**What the interviewer is testing:** Software engineering best practices, API design, containerization, and production readiness.

**Ideal Answer:**
> "The serving architecture is designed for low latency and high reliability:
> 1. **Lifespan Context Manager:** Model artifacts (`churn_model.pkl`, `scaler.pkl`, `preprocessor.pkl`, `threshold.pkl`) are loaded into memory exactly once at application startup using FastAPI's `@asynccontextmanager(app)`. No disk I/O occurs per HTTP request.
> 2. **Strict Validation with Pydantic v2:** Input payloads are strictly validated using `CustomerInput` schemas with type bounds, allowed literals, and non-negative constraints (e.g., `MonthlyCharges >= 0`), returning clear 422 error payloads on bad input.
> 3. **Dual Serving Modes:**
>    - `POST /predict`: Real-time single customer scoring, returning predicted churn binary, calibrated probability, and categorical risk tier (`LOW`, `MEDIUM`, `HIGH`).
>    - `POST /predict/batch`: High-throughput vectorized batch scoring supporting up to 1,000 customers per request.
>    - `GET /health`: Liveness and readiness probe for container orchestrators (Kubernetes / ECS).
> 4. **Multi-Stage Dockerfile:** A builder stage installs heavy build dependencies (gcc, pip packages), and a lightweight Python slim runtime copies only the final site-packages. The container runs as a non-root `appuser` with a Docker `HEALTHCHECK` probe targeting `/health`."

---

### Q7: If you deploy this to production, how do you handle Model Drift and Data Drift?
**What the interviewer is testing:** Senior-level understanding of day-2 operations in machine learning.

**Ideal Answer:**
> "Once deployed, churn models degrade due to two primary phenomena:
> 1. **Data Drift (Covariate Shift):** The distribution of input features shifts (e.g., telecom rolls out a new 5G plan, changing the distribution of `MonthlyCharges` or payment methods).
>    - *Monitoring Solution:* Log production payloads to an event stream (e.g., Kafka or cloud storage) and compute statistical distance metrics weekly using Evidently AI or Kolmogorov-Smirnov (KS) tests for continuous features, and Population Stability Index (PSI) for categorical features. A PSI > 0.2 triggers an alert.
> 2. **Concept Drift:** The relationship between features and churn changes (e.g., a competitor launches an aggressive promo in a specific region).
>    - *Monitoring Solution:* Track rolling ground truth (as customer billing cycles close and churn outcomes solidify) and monitor degradation in ROC AUC and F1 against our baseline of 0.845 / 0.61.
> 3. **Automated Retraining Trigger:** If PSI breaches 0.2 or F1 degrades by >5%, the CI/CD pipeline triggers automated retraining, logs a new candidate version to the MLflow Model Registry, and runs shadow/canary evaluation before promotion."

---

## 3. Resume Bullet Points: Cheat Sheet & Talking Points

| Bullet Point on Resume | Key Metrics to Mention | Core Technical Terminology |
|---|---|---|
| **Pipeline & Preprocessing** | 7,043 real records, 5 engineered features, 70/15/15 stratified split | Leak-free splitting, median imputation, `tenure_group`, vectorized feature engineering |
| **Model Selection** | ROC AUC `0.845`, F1 `0.61` (baseline) → `0.635` (PR threshold) | Logistic Regression, XGBoost, Random Forest, GridSearchCV, Precision-Recall curve tuning |
| **MLOps & Deployment** | Sub-50ms latency, 1,000 batch limit, 9 passing tests, multi-stage build | MLflow Model Registry, FastAPI lifespan loading, Docker non-root user, Pydantic v2 schemas |

---

## 4. Rapid-Fire / Curveball Questions

1. **"What happens if a customer has tenure = 0?"**
   - *Answer:* "New customers who just joined have `tenure=0` and often a blank string in `TotalCharges`. Our preprocessor explicitly maps `tenure == 0` TotalCharges to `0.0` instead of letting it become NaN, preventing drops or bad imputations."

2. **"Why use F1 score instead of Accuracy?"**
   - *Answer:* "With 73% non-churners, a naive model predicting 'No Churn' for everyone achieves 73% accuracy while failing 100% of business goals. F1 balances Precision (avoiding costly unnecessary discounts to loyal customers) and Recall (catching true churners before they leave)."

3. **"How does the model output risk levels?"**
   - *Answer:* "Probability < 0.35 is `LOW`, 0.35 to 0.65 is `MEDIUM`, and > 0.65 is `HIGH`. This allows business teams to tier their interventions: automated email nudges for medium risk, versus high-touch retention specialist calls for high risk."
