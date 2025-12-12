# =====================================================
# Task 2 - FULL TensorFlow Neural Network + KerasTuner
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import keras_tuner as kt  # KerasTuner

print(">>> Loading data...")

# 1. Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 3. Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Drop ID column
df = df.drop(columns=["customerID"])

# 5. Prepare features & labels
X = df.drop(columns=["Churn"])
y = df["Churn"]

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# PART 1 — BASELINE TENSORFLOW MODEL
# =====================================================

print("\n>>> Building baseline TensorFlow model...")

baseline_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

baseline_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n>>> Training baseline model...")
baseline_model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Baseline evaluation
y_pred_prob = baseline_model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n=== BASELINE MODEL RESULTS ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# =====================================================
# PART 2 — HYPERPARAMETER TUNING WITH KERAS TUNER
# =====================================================

print("\n>>> Setting up KerasTuner...")

def build_model(hp):
    model = Sequential()

    # Tune number of units in first layer
    units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32)
    model.add(Dense(units_1, activation="relu", input_shape=(X_train_scaled.shape[1],)))

    # Tune dropout
    dropout_rate = hp.Choice("dropout_rate", values=[0.0, 0.2, 0.3])
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Tune second layer
    units_2 = hp.Int("units_2", min_value=16, max_value=64, step=16)
    model.add(Dense(units_2, activation="relu"))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Tune learning rate
    lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    directory="kt_dir",
    project_name="telco_churn"
)

print("\n>>> Running hyperparameter search...")
tuner.search(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\n=== BEST HYPERPARAMETERS FOUND ===")
print("units_1:", best_hp.get("units_1"))
print("units_2:", best_hp.get("units_2"))
print("dropout_rate:", best_hp.get("dropout_rate"))
print("learning_rate:", best_hp.get("learning_rate"))

# =====================================================
# PART 3 — TRAIN BEST MODEL FROM TUNER
# =====================================================

best_model = tuner.get_best_models(1)[0]

print("\n>>> Training best model on full training set...")
best_model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate tuned model
y_pred_prob_best = best_model.predict(X_test_scaled).flatten()
y_pred_best = (y_pred_prob_best > 0.5).astype(int)

print("\n=== TUNED MODEL RESULTS ===")
print(classification_report(y_test, y_pred_best))
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

print("\n>>> FULL TASK 2 (TensorFlow + KerasTuner) COMPLETE!")