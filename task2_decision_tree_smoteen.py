# ==========================================================
# Task 2 - Decision Tree + SMOTEENN (FULL SCRIPT)
# Includes: Before/After balancing stats + evaluation metrics
# ==========================================================

import pandas as pd
import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

from sklearn.utils.class_weight import compute_class_weight

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline


print(">>> Loading dataset...")

# 1) Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2) Clean TotalCharges (string -> numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3) Encode target (Churn)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 4) Drop customerID (not predictive)
df = df.drop(columns=["customerID"])

# 5) Split into X and y
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 6) One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

print("Data shape (X):", X.shape)
print("Data shape (y):", y.shape)

# 7) Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ==========================================================
# PART A: Class distribution BEFORE SMOTEENN (TRAIN)
# ==========================================================

before_counts = Counter(y_train)
before_total = len(y_train)

before_no = before_counts[0]
before_yes = before_counts[1]

before_no_pct = (before_no / before_total) * 100
before_yes_pct = (before_yes / before_total) * 100
before_ratio = before_no / before_yes

print("\n=== Class Distribution BEFORE SMOTEENN (TRAIN) ===")
print(f"No Churn (0): {before_no} ({before_no_pct:.1f}%)")
print(f"Churn (1):    {before_yes} ({before_yes_pct:.1f}%)")
print(f"Ratio (No/Yes): {before_ratio:.2f}:1")

# ==========================================================
# PART B: Show class weights (useful for TensorFlow later)
# ==========================================================

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

print("\n=== Class Weights (TRAIN) ===")
print(class_weight)

# ==========================================================
# PART C: AFTER SMOTEENN stats (TRAIN ONLY)
# ==========================================================

smoteenn = SMOTEENN(random_state=42)
X_res, y_res = smoteenn.fit_resample(X_train, y_train)

after_counts = Counter(y_res)
after_total = len(y_res)

after_no = after_counts[0]
after_yes = after_counts[1]

after_no_pct = (after_no / after_total) * 100
after_yes_pct = (after_yes / after_total) * 100
after_ratio = after_no / after_yes

change = after_total - before_total
change_pct = (change / before_total) * 100

print("\n=== Class Distribution AFTER SMOTEENN (TRAIN) ===")
print(f"No Churn (0): {after_no} ({after_no_pct:.1f}%)")
print(f"Churn (1):    {after_yes} ({after_yes_pct:.1f}%)")
print(f"Ratio (No/Yes): {after_ratio:.2f}:1")

print("\n=== Dataset Impact (TRAIN) ===")
print(f"Training size before: {before_total}")
print(f"Training size after:  {after_total}")
print(f"Change: {change} samples ({change_pct:.1f}%)")

# ==========================================================
# PART D: Train Decision Tree with SMOTEENN pipeline
# (SMOTEENN applied ONLY inside pipeline on training folds)
# ==========================================================

print("\n>>> Training Decision Tree with SMOTEENN pipeline...")

pipeline = Pipeline([
    ("smoteenn", SMOTEENN(random_state=42)),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        max_depth=6,
        min_samples_leaf=5
    ))
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ==========================================================
# PART E: Evaluation
# ==========================================================

print("\n=== DECISION TREE + SMOTEENN RESULTS (TEST) ===")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("ROC-AUC:", auc)
print("Confusion Matrix:\n", cm)

print("\n>>> Done.")