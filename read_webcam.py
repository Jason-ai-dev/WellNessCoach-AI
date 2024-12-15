import os
import cv2
import joblib
import numpy as np
from feat import Detector

SCALER_PATH = "scaler.pkl"
MODEL_PATH = "rf_model.pkl"
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

EMOTIONS = ["neutral", "happy", "angry", "disgust", "fear", "sad", "surprise"]

detector = Detector(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")


def preprocess_face(face, target_size=(48, 48)):
    """
    Preprocess a face image: grayscale, resize, flatten, and scale.
    """
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_face = cv2.resize(gray_face, target_size)  # Resize to (48, 48)
    flattened_face = resized_face.flatten()  # Flatten to a 1D array
    scaled_face = scaler.transform([flattened_face])  # Scale using the trained scaler
    return scaled_face


def main():
    print("Starting real-time emotion detection... Press ESC to exit.")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not access the webcam.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            face_predictions = detector.detect_faces(frame)
            print(f"Detected faces: {face_predictions}")

            # Process detected face
            if not face_predictions:
                print("No faces detected.")
            else:
                for prediction in face_predictions:
                    for face in prediction:  # Handle nested list
                        if isinstance(face, list) and len(face) >= 5:

                            print("Do we enter the if statement isinstance...? YES")
                            # Extract bounding box
                            x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                            w, h = x2 - x1, y2 - y1  # Compute width and height
                            x, y = x1, y1  # Top-left corner

                            # Extract and preprocess the face region
                            face_region = frame[y:y + h, x:x + w]
                            if face_region.size == 0:  # Ensure valid face region
                                print("Empty face region detected. Skipping...")
                                continue

                            try:
                                preprocessed_face = preprocess_face(face_region)
                                print(f"Input to model: {preprocessed_face.shape}")

                                # Predict emotion
                                # Takes for granted that model has been trained
                                emotion_index = model.predict(preprocessed_face)[0]
                                predicted_emotion = EMOTIONS[emotion_index] # TODO: Send this to furhat
                                print(f"Predicted Emotion: {predicted_emotion}")

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error during prediction: {e}")

            cv2.imshow("Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("Exiting...")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
