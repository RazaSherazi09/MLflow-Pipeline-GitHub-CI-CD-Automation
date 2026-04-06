import pandas as pd
from sklearn.datasets import load_breast_cancer

def check_drift():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Simulate new data (slight change)
    new_data = X + 0.5  

    drift_score = (X.mean() - new_data.mean()).abs().mean()

    print("Drift Score:", drift_score)

    # Threshold
    if drift_score > 0.1:
        print("⚠️ Drift detected!")
        return True
    else:
        print("✅ No drift")
        return False


if __name__ == "__main__":
    result = check_drift()
    
    if result:
        exit(1)   # trigger retraining
    else:
        exit(0)