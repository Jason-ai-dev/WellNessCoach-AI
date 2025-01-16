import os
import random
import pandas as pd
import numpy as np
from feat import Detector
from sklearn.impute import SimpleImputer

def load_dataset(dataset_dir, max_samples_per_class):
    image_paths, labels = [], []

    for label in range(7):
        label_dir = os.path.join(dataset_dir, str(label))
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} does not exist.")
            continue

        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        selected_images = images[:max_samples_per_class]

        image_paths.extend(selected_images)
        labels.extend([label] * len(selected_images))

    return image_paths, np.array(labels)

def extract_action_units(image_paths, output_file="../../AU/large_sample_aus.csv"):
    detector = Detector(au_model="svm")
    print("Initialized detector:", detector)

    au_features = []
    valid_indices = []

    for idx, img_path in enumerate(image_paths):
        if os.path.exists(img_path):
            try:
                feat = detector.detect_image(img_path)
                if not feat.empty:
                    au_data = feat.filter(like="AU").iloc[0].values
                    au_features.append(au_data)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: No features detected in {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        else:
            print(f"Warning: File not found: {img_path}")

    X = np.array(au_features)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    df = pd.DataFrame(X_imputed, columns=[f"AU{i}" for i in range(1, X_imputed.shape[1] + 1)])
    df["image_path"] = [image_paths[i] for i in valid_indices]

    df.to_csv(output_file, index=False)
    print(f"Action Units saved to {output_file}")

def main():
    dataset_dir = "../../affect_net/AffectNet/train"
    max_samples_per_class = 5000

    print("Loading dataset...")
    image_paths, _ = load_dataset(dataset_dir, max_samples_per_class)

    print("Extracting Action Units...")
    extract_action_units(image_paths)

if __name__ == "__main__":
    main()
