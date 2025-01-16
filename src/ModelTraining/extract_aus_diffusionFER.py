import os
import zipfile
import pandas as pd
import shutil
from feat import Detector
from tqdm import tqdm

DATASET_DIR = "../../DiffusionFER/DiffusionEmotion_S/original"
TEMP_DIR = "temp_unzipped"
OUTPUT_CSV = "../../AU/diffusionFER_aus.csv"

detector = Detector(au_model="svm", device="cuda")
print("Initialized detector:", detector)

def extract_images_from_zip(zip_path, temp_dir):
    """
    Extract images from a zip file to a temporary directory.
    :param zip_path: The path to the zip file.
    :param temp_dir: The temporary directory to extract the images to.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

def process_images(image_dir, detector):
    """
    Process images in a directory and extract Action Units.
    :param image_dir: The directory containing the images.
    :param detector: The detector object to use for extracting Action Units.
    :return: A tuple containing a list of Action Units data and the columns.
    """
    au_data = []
    au_columns = None
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(root, file)
                try:
                    feat = detector.detect_image(image_path)
                    if not feat.empty:
                        au_row = feat.filter(like="AU").iloc[0].tolist()
                        if au_columns is None:
                            au_columns = list(feat.filter(like="AU").columns)
                        au_data.append([image_path] + au_row)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return au_data, au_columns

# Main function
def main():
    all_au_data = []
    final_au_columns = None

    # Process each zip file in the dataset directory (unzipping them first)
    for zip_file in tqdm(os.listdir(DATASET_DIR), desc="Processing Zipped Directories"):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(DATASET_DIR, zip_file)
            emotion_name = os.path.splitext(zip_file)[0]

            print(f"Processing {emotion_name}...")

            temp_image_dir = os.path.join(TEMP_DIR, emotion_name)
            extract_images_from_zip(zip_path, temp_image_dir)

            emotion_au_data, au_columns = process_images(temp_image_dir, detector)

            all_au_data.extend(emotion_au_data)
            if final_au_columns is None:
                final_au_columns = ["image_path"] + au_columns

            shutil.rmtree(temp_image_dir)

    df = pd.DataFrame(all_au_data, columns=final_au_columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Action Units saved to {OUTPUT_CSV}")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Temporary directory {TEMP_DIR} deleted.")

if __name__ == "__main__":
    main()
