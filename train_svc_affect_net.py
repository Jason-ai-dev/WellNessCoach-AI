import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from feat import Detector
import random

# Define paths
DATASET_DIR = "affect_net/AffectNet/train"
OUTPUT_FILE = "extracted_features.csv"

def load_dataset(dataset_dir, max_samples_per_class=1000):
    """Load image paths and labels, limiting to max_samples_per_class for each emotion."""
    image_paths = []
    labels = []

    for label in range(7):  # 0 to 6 corresponds to the emotions
        label_dir = os.path.join(dataset_dir, str(label))
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} does not exist.")
            continue

        # Get all image paths for the current label
        all_images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle and limit to max_samples_per_class
        random.shuffle(all_images)
        selected_images = all_images[:max_samples_per_class]

        image_paths.extend(selected_images)
        labels.extend([label] * len(selected_images))

    return image_paths, np.array(labels)

def extract_action_units(image_paths, output_file="output_aus.csv"):
    """Extract Action Units (AUs) from images using py-feat."""
    detector = Detector(au_model="svm")  # Use the SVM AU detection model
    print("Initialized detector:", detector)
    au_features = []
    valid_indices = []

    # Open the output file in write mode
    with open(output_file, mode='w') as f:
        # Write the header row
        header = "image_path," + ",".join([f"AU{i}" for i in range(1, 18)]) + "\n"
        f.write(header)

        for idx, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                try:
                    # Detect features from the image
                    feat = detector.detect_image(img_path)
                    if not feat.empty:
                        # Extract AU-related data
                        au_data = feat.filter(like="AU")
                        if not au_data.empty:
                            # Write to the file and collect features
                            f.write(f"{img_path}," + ",".join(map(str, au_data.iloc[0].values)) + "\n")
                            au_features.append(au_data.iloc[0].values.flatten())
                            valid_indices.append(idx)
                        else:
                            print(f"Warning: No AUs detected in {img_path}")
                    else:
                        print(f"Warning: No features detected in {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"Warning: File not found: {img_path}")

    # Convert the collected features to a NumPy array
    X = np.array(au_features)
    print(f"Extraction complete: {len(valid_indices)} images processed successfully.")
    return X, valid_indices

def train_model(X, y):
    """Train an SVC model on the given data, handling NaNs."""
    # Impute missing values with the mean of each feature
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train SVC
    svc_model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svc_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svc_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return svc_model, scaler

def main():
    print("Loading dataset...")
    image_paths, labels = load_dataset(DATASET_DIR, max_samples_per_class=100)

    print("Extracting action units...")
    X, valid_indices = extract_action_units(image_paths)
    y = labels[valid_indices]

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid data was found. Check your dataset integrity.")

    print(f"Final dataset size: {X.shape[0]} samples")
    print("Training SVC model...")
    model, scaler = train_model(X, y)

    joblib.dump(model, "svc_au_model_affectnet.pkl")
    joblib.dump(scaler, "scaler_affectnet.pkl")
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
