import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
import joblib

# Path to the test dataset
TEST_DATASET_DIR = "archive/test"
TARGET_SIZE = (48, 48)  # Resize all images to 48x48 pixels

# Emotion labels mapping
LABELS = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6
}

# Inverse mapping for labels
INV_LABELS = {v: k for k, v in LABELS.items()}

def load_test_data(test_dataset_dir, target_size):
    """
    Load test images and their corresponding labels from the dataset directory.

    Args:
        test_dataset_dir (str): Path to the test dataset directory.
        target_size (tuple): Target size to resize images (width, height).

    Returns:
        tuple: Features (X) and labels (y) as numpy arrays.
    """
    X = []
    y = []

    for label_name, label_id in LABELS.items():
        label_dir = os.path.join(test_dataset_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, target_size)
                X.append(resized_image.flatten())
                y.append(label_id)

    return np.array(X), np.array(y)

def evaluate_model(test_dataset_dir, model_path, scaler_path):
    """
    Evaluate the trained RandomForest model on the test dataset.

    Args:
        test_dataset_dir (str): Path to the test dataset.
        model_path (str): Path to the trained model file.
        scaler_path (str): Path to the scaler file.
    """
    print("Loading test data...")
    X_test, y_test = load_test_data(test_dataset_dir, TARGET_SIZE)

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test dataset is empty or could not be loaded. Check the dataset path.")

    print(f"Loaded {len(X_test)} test samples.")

    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Scaling test data...")
    X_test_scaled = scaler.transform(X_test)

    print("Predicting...")
    y_pred = model.predict(X_test_scaled)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=[INV_LABELS[i] for i in range(len(LABELS))]))

def main():
    evaluate_model(TEST_DATASET_DIR, "rf_model.pkl", "scaler.pkl")

if __name__ == "__main__":
    main()
