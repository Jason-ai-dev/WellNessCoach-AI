import numpy as np
import pandas as pd
import cv2
import joblib
from feat import Detector  # Retained for face and landmark detection
from feat.utils import FEAT_EMOTION_COLUMNS

SCALER_PATH = "../scaler.pkl"
MODEL_PATH = "../rf_model.pkl"
scaler = joblib.load(SCALER_PATH)  # Load your custom scaler
model = joblib.load(MODEL_PATH)  # Load your trained RandomForest model

EMOTIONS = ["neutral", "happy", "angry", "disgust", "fear", "sad", "surprise"]


class MyDetectFace():
    _myImg: np.ndarray
    _myLabledImg: np.ndarray
    _predDone: bool
    _au: pd.DataFrame
    _fname: str

    def __init__(self):
        self._detector = Detector(
            face_model='faceboxes',
            landmark_model="pfld",
            au_model='svm',
            device='cpu'  # Force PyFeat to use CPU
        )

        au_names = self._detector.info['au_presence_columns']
        au_names.insert(0, 'face')
        au_names.insert(0, 'file')
        self._au = pd.DataFrame(columns=au_names)

    def getImg(self) -> np.ndarray:
        return self._myImg

    def getAUs(self) -> pd.DataFrame:
        return self._au

    def setImg(self, img: np.ndarray, fname: str) -> None:
        self._myImg = img
        self._myLabledImg = img.copy()
        self._predDone = False
        self._fname = fname
        return

    def preprocess_face(self, img: np.ndarray, face_coords: tuple) -> np.ndarray:
        """
        Extract, preprocess, and resize the face region for model prediction.
        """
        x0, y0, x1, y1, _ = face_coords
        face_region = img[int(y0):int(y1), int(x0):int(x1)]  # Crop face
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_face = cv2.resize(gray_face, (48, 48))  # Resize to 48x48
        return resized_face.flatten()

    def predImg(self) -> None:
        faces = self._detector.detect_faces(self._myImg)
        landmarks = self._detector.detect_landmarks(self._myImg, faces)

        # Check for detected faces
        if len(faces) == 0:
            print("No faces detected.")
            return

        f_index = 0
        for face_coords in faces[0]:
            # Preprocess the face and predict emotion
            face_data = self.preprocess_face(self._myImg, face_coords)
            face_data_scaled = scaler.transform([face_data])  # Apply scaling
            predicted_emotion = model.predict(face_data_scaled)[0]  # Predict emotion

            # Draw bounding box and label
            (x0, y0, x1, y1, p) = face_coords
            cv2.rectangle(self._myLabledImg, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            emotion_text = EMOTIONS[predicted_emotion]
            cv2.putText(self._myLabledImg, emotion_text, (int(x0), int(y0 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            f_index += 1

        self._predDone = True
        return

    def getLableImg(self) -> np.ndarray:
        if self._predDone:
            return self._myLabledImg
        else:
            return None

    def getUserEmotion(self) -> str | tuple:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emo = ""
        try:
            ret, frame = cam.read()

            # Needed because the first cam.read() call doesnt seems to get a frame good enough that a face can be found
            # Might be because the first frame is black or something
            for i in range(5):
                ret, frame = cam.read()

            if not ret:
                print("OpenCV found an error reading the next frame.")
                return "", ()

            faces = self._detector.detect_faces(frame)
            if len(faces) == 0:
                print("No faces detected.")
                return "", ()

            for face_coords in faces[0]:
                # Preprocess face and predict emotion
                face_data = self.preprocess_face(frame, face_coords)
                face_data_scaled = scaler.transform([face_data])
                predicted_emotion = model.predict(face_data_scaled)[0]
                emo = EMOTIONS[predicted_emotion]
                break  # Process only the first detected face

        finally:
            cam.release()
            return emo
