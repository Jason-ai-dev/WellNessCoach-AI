import cv2
from feat import Detector
import joblib
from sklearn.impute import SimpleImputer
import os
import time

TEMP_IMAGE_PATH = "temp_frame.jpg"
MODEL_FILE = "Models/svc_model_no_fear_disgust_diffusionFER.pkl"
SCALER_FILE = "Models/scaler_no_fear_disgust_diffusionFER.pkl"

print("Loading model and scaler...")
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

detector = Detector(device="cpu")
print("Initialized detector:", detector)

EMOTIONS = ["Anger", "Happiness", "Neutral", "Sadness", "Surprise"]

def predict_emotion(image_path, detector, model, scaler):
    try:
        detections = detector.detect_image(image_path)
        if not detections.empty:
            au_features = detections.filter(like="AU").iloc[0].values
            au_features = au_features.reshape(1, -1)

            imputer = SimpleImputer(strategy="mean")
            au_features_imputed = imputer.fit_transform(au_features)

            au_features_scaled = scaler.transform(au_features_imputed)

            emotion_index = model.predict(au_features_scaled)[0]
            return EMOTIONS[emotion_index]
    except Exception as e:
        print(f"Error during detection: {e}")
    return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    last_snapshot_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_time = time.time()

        if current_time - last_snapshot_time >= 2:
            last_snapshot_time = current_time

            cv2.imwrite(TEMP_IMAGE_PATH, frame)

            emotion = predict_emotion(TEMP_IMAGE_PATH, detector, model, scaler)

            print(f"Emotion detected: {emotion}")

        cv2.imshow("Webcam Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)

if __name__ == "__main__":
    main()
