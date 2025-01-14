import os
import cv2
import pandas as pd
import numpy as np
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils import resample
import joblib
import shutil  # For cleaning up temporary directories

# Define paths
CSV_PATH = "C://Users//jason//Documents//Uppsala University Courses//Intelligent Interactive Systems//Training Data//DiffusionFER//DiffusionEmotion_S//dataset_sheet.csv"
DATASET_DIR = "C://Users//jason//Documents//Uppsala University Courses//Intelligent Interactive Systems//Training Data//DiffusionFER//DiffusionEmotion_S//cropped"
COMP_DIR = "C://Users//jason//Documents//Uppsala University Courses//Intelligent Interactive Systems//Training Data//test"
TEMP_DIR = "temp_unzipped"

# TODO: Terrible at detecting angry, disgusted, sad and sometimes fear. Tune the model in order to fix this.

def extract_zips(dataset_dir, temp_dir):
    """
    Extract all ZIP files in the dataset directory to a temporary directory.
    Flatten any nested directories caused by ZIP file structures.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # Clean up any existing temp directory
    os.makedirs(temp_dir, exist_ok=True)

    for file in os.listdir(dataset_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(dataset_dir, file)
            emotion = file.replace(".zip", "")
            extract_path = os.path.join(temp_dir, emotion)

            # Extract ZIP to a temporary location
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # Flatten any nested directories
            for root, dirs, files in os.walk(extract_path):
                for filename in files:
                    src_path = os.path.join(root, filename)
                    dest_path = os.path.join(extract_path, filename)
                    if src_path != dest_path:  # Avoid overwriting files unnecessarily
                        shutil.move(src_path, dest_path)

            # Remove any leftover nested directories
            for root, dirs, _ in os.walk(extract_path, topdown=False):
                for directory in dirs:
                    shutil.rmtree(os.path.join(root, directory))

            print(f"Extracted and flattened {file} to {extract_path}")
    return temp_dir


def load_dataset_info(csv_path, temp_dir):
    data = pd.read_csv(csv_path)
    image_paths = data["subDirectory_filePath"].apply(
        lambda x: os.path.join(temp_dir, x.replace("DiffusionEmotion_S_cropped/", ""))
    )
    labels = data["expression"]  # Target labels (0-neutral, 1-happy, etc.)
    return image_paths, labels


def load_and_preprocess_images(image_paths, labels, target_size=(48, 48)):
    """
    Load images, convert to grayscale, resize, flatten, and balance classes with oversampling.
    """
    images = []
    valid_indices = []

    for idx, img_path in enumerate(image_paths):
        if os.path.exists(img_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, target_size)
                images.append(resized_image.flatten())
                valid_indices.append(idx)
            else:
                print(f"Warning: Unable to read image {img_path}") # For debugging
        else:
            print(f"Warning: File not found {img_path}") # For debugging

    X = np.array(images)
    y = labels.iloc[valid_indices].values

    # Balance classes through oversampling (found on Google and chatgpt, not sure how it works to be honest. Might be stupid code hahah)
    X_balanced, y_balanced = [], []
    unique_classes = np.unique(y)
    max_samples = max(np.bincount(y))

    for label in unique_classes:
        X_class = X[y == label]
        y_class = y[y == label]
        X_upsampled, y_upsampled = resample(X_class, y_class, replace=True, n_samples=max_samples, random_state=42)
        X_balanced.append(X_upsampled)
        y_balanced.append(y_upsampled)

    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)

    print("Class distribution after balancing:")
    print({label: count for label, count in zip(*np.unique(y_balanced, return_counts=True))})

    return X_balanced, y_balanced


# Combine Features and Train the Model
def train_model(X, y):
    """
    Train a Random Forest model on the given data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Ensure X is 2D: (samples, features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train RandomForest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return rf_model, scaler

def process_unzipped_data(comp_dir, target_size=(48, 48)):
    """
    Process unzipped images in the given directory for model testing.
    - comp_dir: Path to the directory containing unzipped image folders.
    - target_size: The size to which images will be resized.
    """
    test_images = []
    test_image_paths = []

    # Traverse the directory structure to load images
    for root, _, files in os.walk(comp_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    resized_image = cv2.resize(image, target_size)
                    test_images.append(resized_image.flatten())
                    test_image_paths.append(img_path)
                else:
                    print(f"Warning: Unable to read image {img_path}")

    if len(test_images) == 0:
        raise ValueError("No valid images found in the test dataset.")

    X_test = np.array(test_images)
    return X_test, test_image_paths

def test_model_on_unzipped_data(model_path, scaler_path, comp_dir, target_size=(48, 48)):
    """
    Test the trained model on a dataset with unzipped image folders.
    """
    # Label mapping
    label_mapping = {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "surprise",
        4: "fear",
        5: "disgust",
        6: "angry"
    }

    # Load the trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Processing unzipped data...")
    X_test, test_image_paths = process_unzipped_data(comp_dir, target_size)

    # Extract true labels from folder names
    true_labels = [os.path.basename(os.path.dirname(path)) for path in test_image_paths]

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Map numeric predictions to emotion names
    y_pred_mapped = [label_mapping[pred] for pred in y_pred]

    # Ensure true labels and predictions are of the same type
    true_labels = np.array(true_labels, dtype=str)
    y_pred_mapped = np.array(y_pred_mapped, dtype=str)

    # Generate a classification report
    report = classification_report(true_labels, y_pred_mapped, zero_division=0, output_dict=True)
    print("Classification Report:\n", report)

    # Save predictions and true labels
    predictions = pd.DataFrame({
        "FilePath": test_image_paths,
        "TrueLabel": true_labels,
        "PredictedLabel": y_pred_mapped
    })
    print("Predictions on new dataset:\n", predictions)

    predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv.")

    return predictions


def main():
    print("Extracting ZIP files...")
    temp_dir = extract_zips(DATASET_DIR, TEMP_DIR)

    print("Loading dataset information...")
    image_paths, labels = load_dataset_info(CSV_PATH, temp_dir)

    # Shuffle data to ensure randomness
    image_paths, labels = shuffle(image_paths, labels, random_state=42)

    print("Loading and preprocessing images...")
    X, y = load_and_preprocess_images(image_paths, labels)

    # Ensure there is data to train on
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid data was found. Check your file paths and dataset integrity.")

    print(f"Final dataset size: {X.shape[0]} samples")
    print("Training Random Forest model...")
    model, scaler = train_model(X, y)

    # Save model and scaler
    joblib.dump(model, "rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully.")

    # Clean up the temp directory
    shutil.rmtree(temp_dir)
    print("Temporary files cleaned up.")

    # Test the model on the new dataset
    print("Testing the trained model on a new dataset...")
    test_model_on_unzipped_data("rf_model.pkl", "scaler.pkl", COMP_DIR)

if __name__ == "__main__":
    main()
