# Fraud Detection Project - Adey Innovations Inc.

##  Project Overview

This project focuses on **improving fraud detection for e-commerce and bank transactions**. The goal is to develop strong, interpretable machine learning models that detect fraudulent transactions accurately while balancing false positives and false negatives.

As a data scientist at **Adey Innovations Inc.**, your mission is to:
- Detect fraudulent transactions in both e-commerce and bank credit data.
- Handle challenges like **class imbalance**, real-time monitoring, and geolocation analysis.
- Build interpretable models and provide actionable insights for business decisions.

---

## 🗂 Folder Structure

fraud-detection/
├── .vscode/
│ └── settings.json
├── .github/
│ └── workflows/
│ └── unittests.yml
├── data/ # Add to .gitignore
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned & feature-engineered datasets
├── notebooks/
│ ├── init.py
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ ├── modeling.ipynb
│ ├── shap-explainability.ipynb
│ └── README.md
├── src/
│ ├── init.py
├── tests/
│ ├── init.py
├── models/ # Saved trained models
├── scripts/
│ ├── init.py
│ └── README.md
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

##  Data & Features

### 1. E-commerce Transactions (`Fraud_Data.csv`)
- **user_id**: unique user identifier  
- **signup_time**: timestamp of signup  
- **purchase_time**: timestamp of purchase  
- **purchase_value**: transaction amount  
- **device_id**: unique device identifier  
- **source**: traffic source (SEO, Ads, etc.)  
- **browser**: browser used  
- **sex**: user gender (M/F)  
- **age**: user age  
- **ip_address**: transaction IP address  
- **class**: target variable (1 = fraud, 0 = legit)  

**Challenge:** Highly imbalanced dataset.

### 2. IP Address Mapping (`IpAddress_to_Country.csv`)
- **lower_bound_ip_address**: range start  
- **upper_bound_ip_address**: range end  
- **country**: corresponding country  

### 3. Bank Transactions (`creditcard.csv`)
- **Time**: seconds since first transaction  
- **V1-V28**: anonymized PCA features  
- **Amount**: transaction amount  
- **Class**: target variable (1 = fraud, 0 = legit)  

**Challenge:** Extremely imbalanced dataset.

---

##  Tasks

### **Task 1 - Data Analysis & Preprocessing**
- Handle missing values, duplicates, and correct datatypes.
- Conduct exploratory data analysis (univariate, bivariate, class distribution).
- Geolocation integration: convert IPs to integer, merge with country info.
- Feature engineering: transaction frequency, velocity, time-based features.
- Normalize numeric features and one-hot encode categorical features.
- Handle class imbalance with **SMOTE** or undersampling.

---

### **Task 2 - Model Building & Training**
- **Data Preparation:** Stratified train-test split, separate features and targets.  
  - Targets: `class` (Fraud_Data.csv), `Class` (creditcard.csv)
- **Baseline Model:** Logistic Regression  
  - Evaluate: F1-Score, AUC-PR, Confusion Matrix
- **Ensemble Model:** Random Forest / XGBoost / LightGBM  
  - Hyperparameter tuning (e.g., `n_estimators`, `max_depth`)  
  - Evaluate same metrics
- **Cross-Validation:** Stratified K-Fold (k=5)  
  - Report mean ± std for F1-Score
- **Model Comparison & Selection:** Compare models side-by-side, select best based on performance & interpretability.

---

### **Task 3 - Model Explainability**
- **Feature Importance Baseline:** Extract from Random Forest  
  - Visualize top 10 features
- **SHAP Analysis:**  
  - SHAP Summary Plot for global importance  
  - SHAP Force Plots for 3 predictions:  
    - True Positive  
    - False Positive  
    - False Negative
- **Interpretation:**  
  - Compare SHAP importance with built-in importance  
  - Identify top 5 drivers of fraud predictions
- **Business Recommendations:**  
  - At least 3 actionable recommendations based on SHAP insights  
  - Example: "Transactions within X hours of signup should receive additional verification."

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/MekdelawitGebre/fraud-detection.git
cd fraud-detection
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
Or in Colab:

python
Copy code
!pip install imbalanced-learn scikit-learn seaborn joblib xgboost shap --quiet
3. Load Data
Place Fraud_Data.csv, creditcard.csv, and IpAddress_to_Country.csv in data/raw/.

Processed datasets should be stored in data/processed/.

4. Run Notebooks
EDA & Feature Engineering: eda-fraud-data.ipynb, eda-creditcard.ipynb, feature-engineering.ipynb

Model Training: modeling.ipynb

SHAP Explainability: shap-explainability.ipynb

5. Access Saved Models
python
Copy code
import joblib
rf_fraud = joblib.load('models/random_forest_fraud.pkl')
rf_credit = joblib.load('models/random_forest_credit.pkl')
 Model Evaluation
Metrics: F1-Score, AUC-PR, Confusion Matrix, CV mean ± std

Business Recommendations
1.Transactions shortly after signup may require additional verification.

2.Large transactions from new IP countries may trigger alerts.

3.Transactions with unusual device/browser patterns should be flagged.

