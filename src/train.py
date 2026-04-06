# Import Libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn


# ================================
# STEP 2: LOAD DATASET
# ================================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\nDataset Loaded Successfully!")
print("Shape:", X.shape)


# ================================
# STEP 3: TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Done!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ================================
# STEP 5: MLflow EXPERIMENT SETUP
# ================================
mlflow.set_experiment("Breast_Cancer_Experiment")


# ================================
# RUN 1: Logistic Regression
# ================================
with mlflow.start_run(run_name="Logistic_Regression"):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("\nRun 1 Completed - Logistic Regression")
    print("Accuracy:", acc)


# ================================
# RUN 2: Random Forest (100 trees)
# ================================
with mlflow.start_run(run_name="RandomForest_100"):

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("\nRun 2 Completed - Random Forest (100)")
    print("Accuracy:", acc)


# ================================
# RUN 3: Random Forest (200 trees)
# ================================
with mlflow.start_run(run_name="RandomForest_200"):

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Logging
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc)

    # Register Best Model (Optional but recommended)
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="Best_Breast_Cancer_Model"
    )

    print("\nRun 3 Completed - Random Forest (200)")
    print("Accuracy:", acc)


print("\n ALL RUNS COMPLETED SUCCESSFULLY!")