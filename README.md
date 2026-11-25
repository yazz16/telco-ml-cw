# Telco Customer Churn Prediction 📉

Predicting which telecom customers are likely to leave (churn) using
exploratory data analysis, decision trees, and a neural network.

## Problem
Losing customers is expensive for telecom companies. If we can predict churn early,
the business can target at-risk customers with retention offers.

## Dataset
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): about 7,000 customers,
with contract, services, billing and tenure details. Target: `Churn` (Yes/No).

## Project structure
| File | What it does |
|---|---|
| `task1_eda.py` | Data cleaning, churn distribution, categorical & numerical analysis, correlation heatmap |
| `task2_decision_tree.py` | Decision Tree classifier (baseline) |
| `task2_decision_tree_smote.py` | Decision Tree with **SMOTE** to handle class imbalance |
| `task2_tensorflow_nn.py` | Neural network classifier built with TensorFlow/Keras |

## Approach
1. **EDA:** fixed missing `TotalCharges`, analysed churn by Contract, InternetService,
   PaymentMethod, TechSupport and OnlineSecurity, and studied tenure and charges vs churn
2. **Feature encoding** and a correlation analysis to rank churn drivers
3. **Modelling:** Decision Tree → Decision Tree + SMOTE → Neural Network
4. **Evaluation:** compared models on accuracy, precision, recall and F1
   (recall matters most, because missing a churner costs money)

## Key insights
- Customers on **month-to-month contracts** churn far more than those on 1–2 year contracts
- **Short-tenure** customers are the highest risk
- Lack of **tech support / online security** is linked to higher churn

## Run it
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
# download the dataset from Kaggle into this folder, then:
python task1_eda.py
python task2_decision_tree.py

## Tech
Python · Pandas · Seaborn · scikit-learn · imbalanced-learn (SMOTE) · TensorFlow
