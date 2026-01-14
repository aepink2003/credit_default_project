# Credit Card Default Prediction  
### End-to-End Data Science Project

## TL;DR
Built an end-to-end credit risk prediction pipeline using 30,000 real customer records.  
Compared Logistic Regression, Random Forest, and XGBoost, and recommended XGBoost based on recall–precision tradeoffs and business cost considerations.  
Deployed an interactive Streamlit app for model exploration and insights.

🔗 **Live App:** https://creditdefaultprojectashediga.streamlit.app/

---

## Overview
This project builds an end-to-end machine learning pipeline to predict credit card default risk using real-world customer demographics, credit limits, and payment history. The goal is to demonstrate how messy financial data can be transformed into actionable business insights to support credit risk decision-making.

The project covers:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and comparison
- Model explainability
- Business recommendations

---

## Business Problem
Credit card defaults result in significant financial losses for issuers. Accurately identifying high-risk customers enables proactive interventions such as credit limit adjustments, targeted outreach, or manual review.

### Objective
Predict whether a customer will default on their next month’s payment and recommend a model based on business tradeoffs between false positives and false negatives.

---

## Dataset
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **Records:** 30,000 credit card clients
- **Features:** Demographics, credit limits, bill statements, payment amounts, repayment history (April–September 2005)
- **Target Variable:** `default` (1 = default, 0 = no default)

---

## Approach

### Data Cleaning
- Removed non-informative identifiers
- Corrected inconsistent categorical encodings
- Encoded categorical variables for modeling

### Exploratory Data Analysis (EDA)
- Analyzed default rates across demographic groups
- Explored distributions of credit limits, bills, and payments
- Examined relationships between payment history and default behavior

### Feature Engineering
- Payment delay statistics (maximum and average delay)
- Aggregated bill and payment features
- Payment-to-bill ratios

### Modeling
- Logistic Regression (baseline)
- Random Forest
- XGBoost

### Evaluation Metrics
- Precision, Recall, F1-score
- ROC-AUC  
- Emphasis on **recall**, reflecting the higher cost of missing a defaulter compared to flagging a non-defaulter

---

## Results

| Model               | Recall (Default) | Precision (Default) | ROC-AUC |
|---------------------|------------------|---------------------|---------|
| Logistic Regression | 0.70             | 0.29                | 0.64    |
| Random Forest       | 0.34             | 0.65                | 0.76    |
| XGBoost             | 0.53             | 0.48                | 0.75    |

### Key Insights
- **Logistic Regression** captures the most defaulters but generates many false positives.
- **Random Forest** is conservative and precise but misses many high-risk customers.
- **XGBoost** provides the best balance between recall and precision, making it the recommended model for deployment.

Model selection was driven by asymmetric business costs, where missing a defaulter is more expensive than a false alarm.

---

## Explainability
Feature importance and SHAP analysis were used to validate model behavior and ensure predictions aligned with financial intuition. Recent payment delays, payment-to-bill ratios, and credit limits emerged as the strongest drivers of default risk.

---

## Business Recommendations
- Flag customers with recent multi-month payment delays for proactive intervention.
- Use XGBoost as a risk-scoring model to prioritize high-risk customers.
- Adjust decision thresholds based on tolerance for false positives versus missed defaults.

---

## Tools & Libraries
- **Language:** Python  
- **Data:** pandas, NumPy  
- **Modeling:** scikit-learn, XGBoost  
- **Explainability:** SHAP  
- **Visualization:** matplotlib, seaborn  
- **Deployment:** Streamlit  

---

## Future Improvements
- Incorporate time-aware validation to better reflect production deployment
- Explore cost-sensitive learning and custom loss functions
- Evaluate model performance stability across demographic subgroups
