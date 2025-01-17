import time
import cv2
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue
import joblib
from sklearn.impute import SimpleImputer

TEMP_IMAGE_PATH = "temp_frame.jpg"

MODEL_FILE = "../Models/svc_model_affectnet.pkl"
SCALER_FILE = "../Models/scaler_affectnet.pkl"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]


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
            emotion = EMOTIONS[emotion_index]
            with open("output.txt", "w") as f:
                f.write(emotion)
            return emotion
    except Exception as e:
        print(f"Error during detection: {e}")
    return "Unknown"


class SeePete():
    DEPTH_PARAM = 0.3

    def __init__(self, logger):
        # self._detector = Detector(device="auto")
        self._detector = Detector(face_model='faceboxes', au_model='svm', device='cpu')
        # self._detector = Detector(face_model='retinaface',emotion_model='svm', landmark_model="mobilenet", au_model='svm', device='auto')
        self._peteSee = FurhatRemoteAPI("localhost")
        self._log = logger
        self._log.info("Starting SeePete")

        return

    def observeUser(self, queue: Queue, pete: FurhatRemoteAPI) -> None:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emo = ""
        old_emo = ""
        while True:
            ret, frame = cam.read()
            for i in range(10):
                ret, frame = cam.read()

            if not ret:
                self._log.warning("OpenCV found an error reading the next frame.")
                break

            try:
                cv2.imwrite(TEMP_IMAGE_PATH, frame)
                emotion = predict_emotion(TEMP_IMAGE_PATH, self._detector, model, scaler)  # TODO: Use this emotion

                faces = self._detector.detect_faces(frame)

                # The functions seem to assume a collection of images or frames. We access "frame 0".
                faces = faces[0]
                temp_string = "hello"
                for (face, c) in zip(faces,temp_string):
                    (x0, y0, x1, y1, p) = face

                face_loc = ((cam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2 - (x0 + x1) / 2) / cam.get(cv2.CAP_PROP_FRAME_WIDTH),
                            (cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 - (y0 + y1) / 2) / cam.get(
                                cv2.CAP_PROP_FRAME_HEIGHT),
                            self.DEPTH_PARAM * cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / (max((abs(y1 - y0), 0.01,))))
                self._log.info(f"looking at: {face_loc}")
                pete.attend(location=f"{face_loc[0]},{face_loc[1]},{face_loc[2]}")
                if (emo != emotion) and (emo != "") and (emotion != old_emo):
                    old_emo = emo
                    queue.put(f"emotion|>{emo}->{emotion}")
                emo = emotion
                time.sleep(10)
            except AttributeError:
                continue

        cam.release()