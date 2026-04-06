# =========================================
# COMPLETE TRAINING PIPELINE (FINAL - PART 3)
# =========================================

# Imports
import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn


# =========================================
# LOAD DATA
# =========================================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("\nDataset Loaded Successfully!")
print("Shape:", X.shape)


# =========================================
# TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Done!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================================
# FEATURE SCALING (IMPORTANT FIX)
# =========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================
# MLflow EXPERIMENT
# =========================================
mlflow.set_experiment("Breast_Cancer_Experiment")

best_accuracy = 0
best_model = None


# =========================================
# RUN 1 - Logistic Regression
# =========================================
with mlflow.start_run(run_name="Logistic_Regression"):
    mlflow.set_tag("type", "retraining")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model")

    print("\nRun 1 Completed - Logistic Regression")
    print("Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model


# =========================================
# RUN 2 - Random Forest (100)
# =========================================
with mlflow.start_run(run_name="RandomForest_100"):
    mlflow.set_tag("type", "retraining")

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model")

    print("\nRun 2 Completed - Random Forest (100)")
    print("Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model


# =========================================
# RUN 3 - Random Forest (200)
# =========================================
with mlflow.start_run(run_name="RandomForest_200"):
    mlflow.set_tag("type", "retraining")

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="Best_Breast_Cancer_Model"
    )

    print("\nRun 3 Completed - Random Forest (200)")
    print("Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model


# =========================================
# MODEL SELECTION LOGIC (IMPORTANT)
# =========================================
os.makedirs("models", exist_ok=True)
model_path = "models/best_model.pkl"

print("\n🔍 MODEL COMPARISON STARTED")

# Load old model if exists
if os.path.exists(model_path):
    print("📂 Previous model found")

    old_model = joblib.load(model_path)
    old_preds = old_model.predict(X_test)
    old_accuracy = accuracy_score(y_test, old_preds)

else:
    print("⚠️ No previous model found")
    old_accuracy = 0

print(f"Old Model Accuracy: {old_accuracy:.4f}")
print(f"New Model Accuracy: {best_accuracy:.4f}")


# =========================================
# MODEL DECISION
# =========================================
if best_accuracy > old_accuracy:
    print("\n✅ New model is better → Updating production model")

    joblib.dump(best_model, model_path)
    deploy_model = True

else:
    print("\n⚠️ Old model is better → Keeping existing model")

    deploy_model = False


# =========================================
# DEPLOY TO HUGGING FACE
# =========================================
if deploy_model:
    try:
        from huggingface_hub import HfApi

        print("\n🚀 Uploading model to Hugging Face...")

        api = HfApi()

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.pkl",
            repo_id="razaasherazi/mlops-best-model",
            repo_type="model",
            token=os.getenv("HF_TOKEN")
        )

        print("✅ Model successfully uploaded to Hugging Face!")

    except Exception as e:
        print("⚠️ Hugging Face upload failed:", e)

else:
    print("\n⏭️ Deployment skipped (model not improved)")