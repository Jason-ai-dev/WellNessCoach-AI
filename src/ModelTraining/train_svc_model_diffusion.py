import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform
import joblib

FILTERED_CSV_FILE = "../../AU/diffusionFER_aus_no_fear_disgust.csv"
MODEL_FILE = "../../Models/svc_model_no_fear_disgust_diffusionFER.pkl"
SCALER_FILE = "../../Models/scaler_no_fear_disgust_diffusionFER.pkl"

LABEL_MAP = {
    "angry": 0,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

def load_filtered_data(csv_file):
    df = pd.read_csv(csv_file)

    df["label"] = df["image_path"].apply(lambda path: LABEL_MAP[path.split("\\")[1]])

    X = df.filter(like="AU").values
    y = df["label"].values

    return X, y

def train_svc(X, y):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = SVC(class_weight="balanced", random_state=42)

    param_distributions = {
        "C": uniform(0.1, 10),
        "gamma": uniform(0.001, 1),
        "kernel": ["linear", "rbf", "poly"],
    }

    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=50,
        scoring="accuracy",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("Starting hyperparameter tuning for SVC...")
    search.fit(X_train, y_train)
    print("Hyperparameter tuning complete.")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Tuned SVC model saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

    return best_model, scaler

def main():
    # Load data
    print("Loading filtered data...")
    X, y = load_filtered_data(FILTERED_CSV_FILE)

    train_svc(X, y)

if __name__ == "__main__":
    main()
