import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform
import joblib

LABELS = {
    "0": 0,  # anger
    "1": 1,  # disgust
    "2": 2,  # fear
    "3": 3,  # happiness
    "4": 4,  # sadness
    "5": 5,  # surprise
    "6": 6   # neutral
}

def load_aus_and_labels(csv_file):
    df = pd.read_csv(csv_file)

    if "image_path" not in df.columns or not any(df.filter(like="AU").columns):
        raise ValueError("CSV file is invalid or missing necessary columns.")

    df["label"] = df["image_path"].apply(lambda path: int(path.split("\\")[1]))

    X = df.filter(like="AU").values
    y = df["label"].values

    return X, y

def train_model(X, y, model_file="../../Models/svc_model_affectnet.pkl", scaler_file="../../Models/scaler_affectnet.pkl"):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    param_distributions = {
        "C": uniform(0.1, 10),
        "gamma": uniform(0.001, 1),
        "kernel": ["linear", "rbf", "poly"],
    }

    svc = SVC(class_weight="balanced", random_state=42)
    search = RandomizedSearchCV(
        svc,
        param_distributions,
        n_iter=50,
        scoring="accuracy",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("Starting hyperparameter tuning...")
    search.fit(X_train, y_train)
    print("Hyperparameter tuning complete.")

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model saved to {model_file}")
    print(f"Scaler saved to {scaler_file}")

def main():
    csv_file = "../../AU/large_sample_aus.csv"

    print("Loading data...")
    X, y = load_aus_and_labels(csv_file)

    print("Training model...")
    train_model(X, y)

if __name__ == "__main__":
    main()
