# Credit Card Default Prediction — End-to-End Data Science Project

## Overview
This project builds an end-to-end machine learning pipeline to predict credit card default risk using real-world customer, credit, and payment history data. The goal is to show how messy financial data can be transformed into business insights to support credit risk decision-making.

The project covers:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Model training and comparison
- Model explainability
- Business recommendations

## Business Problem
Credit card defaults result in large financial losses for issuers. Correctly identifying high-risk customers allows for proactive intervention, like credit limit adjustments, targeted outreach, or manual review.

**Objective:**  
Predict whether a customer will default on their next month’s payment and recommend an appropriate model based on business tradeoffs between false positives and false negatives.

## Dataset
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **Records:** 30,000 credit card clients
- **Features:** Demographics, credit limits, bill statements, payment amounts, and repayment history (April–September 2005)
- **Target Variable:** `default` (1 = default, 0 = no default)

## Methodology
1. **Data Cleaning**
   - Removed non-informative identifiers
   - Corrected inconsistent categorical codes
   - Encoded categorical variables
2. **EDA**
   - Analyzed default rates by demographic groups
   - Explored distributions of credit limits, bills, and payments
   - Examined correlations between payment history and default
3. **Feature Engineering**
   - Payment delay statistics (max, average delay)
   - Aggregated bill and payment features
   - Payment-to-bill ratios
4. **Modeling**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
5. **Evaluation Metrics**
   - Precision, Recall, F1-score
   - ROC-AUC
   - Emphasis on recall due to asymmetric cost of default

## Results
| Model | Recall (Default) | Precision (Default) | ROC-AUC |
|------|------------------|---------------------|--------|
| Logistic Regression | 0.70 | 0.29 | 0.64 |
| Random Forest | 0.34 | 0.65 | 0.76 |
| XGBoost | 0.53 | 0.48 | 0.75 |

**Key Insight:**  
- Logistic Regression captures the most defaulters but produces many false positives.
- Random Forest is conservative and precise but misses many defaulters.
- **XGBoost provides the best balance**, making it the recommended model for deployment.

## Explainability
Feature importance and SHAP analysis show that **recent payment delays, payment-to-bill ratios, and credit limits** are the strongest drivers of default risk.

## Business Recommendations
- Flag customers with recent multi-month payment delays for proactive intervention.
- Use XGBoost as a risk-scoring model to prioritize high-risk customers.
- Adjust decision thresholds based on tolerance for false positives vs missed defaults.
  
## Tools & Libraries
Python, pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit
