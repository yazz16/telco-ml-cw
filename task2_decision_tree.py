# ============================
# Task 2 - Decision Tree Model
# ============================

# 1. Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print(">>> Loading data...")

# 2. Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 3. Clean TotalCharges like in Task 1
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 4. Encode target variable (Churn: No=0, Yes=1)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 5. Drop customerID (not useful for prediction)
df = df.drop(columns=["customerID"])

# 6. Split into features (X) and labels (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 7. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# 8. Baseline Decision Tree
print("\n>>> Training baseline Decision Tree...")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

print("\n=== BASELINE DECISION TREE RESULTS ===")
print(classification_report(y_test, y_pred_dt))

acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_proba_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Accuracy:", acc_dt)
print("ROC-AUC:", auc_dt)
print("Confusion matrix:\n", cm_dt)

# 9. Hyperparameter tuning with GridSearchCV (for better marks)
print("\n>>> Running GridSearchCV for Decision Tree...")

param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    scoring="f1",  # focus on churn class performance
    cv=5,
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
print("\nBest parameters:", grid_dt.best_params_)

# Evaluate tuned model
y_pred_best = best_dt.predict(X_test)
y_proba_best = best_dt.predict_proba(X_test)[:, 1]

print("\n=== TUNED DECISION TREE RESULTS ===")
print(classification_report(y_test, y_pred_best))

acc_best = accuracy_score(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, y_proba_best)
cm_best = confusion_matrix(y_test, y_pred_best)

print("Accuracy:", acc_best)
print("ROC-AUC:", auc_best)
print("Confusion matrix:\n", cm_best)

print("\n>>> Task 2 - Decision Tree complete.")