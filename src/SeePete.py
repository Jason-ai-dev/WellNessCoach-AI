import time
import cv2
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue

class SeePete():


    def __init__(self, logger):
        # self._detector = Detector(device="auto")
        self._detector = Detector(face_model='faceboxes',emotion_model='resmasknet', landmark_model="pfld", au_model='svm', device='auto')
        # self._detector = Detector(face_model='retinaface',emotion_model='svm', landmark_model="mobilenet", au_model='svm', device='auto')
        au_names = self._detector.info['au_presence_columns']
        au_names.insert(0, 'face')
        au_names.insert(0, 'file')
        self._peteSee = FurhatRemoteAPI("localhost")
        self._log = logger
        self._log.info("Starting SeePete")
        return
    
    def observeUser(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emo = ""
        old_emo = ""
        while True:
            ret, frame = cam.read()
            if not ret:
                self._log.warning("OpenCV found an error reading the next frame.")
                break
            
            try:
                faces = self._detector.detect_faces(frame)
                landmarks = self._detector.detect_landmarks(frame, faces)
                emotions = self._detector.detect_emotions(frame, faces, landmarks)

                # The functions seem to assume a collection of images or frames. We acces "frame 0".
                faces = faces[0]
                landmarks = landmarks[0]
                emotions = emotions[0]

                strongest_emotion = emotions.argmax(axis=1)

                for (face, top_emo) in zip(faces, strongest_emotion):
                    (x0, y0, x1, y1, p) = face
                    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
                    cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                
                face_loc = ((cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2-(x0+x1)/2)/cam.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                (cam.get(cv2.CAP_PROP_FRAME_HEIGHT)/2-(y0+y1)/2)/cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                pete.attend(location=f"{face_loc[0]},{face_loc[1]},1.0")
                if((emo != FEAT_EMOTION_COLUMNS[top_emo]) and (emo != "") and (FEAT_EMOTION_COLUMNS[top_emo] != old_emo)):
                    old_emo = emo
                    queue.put(f"emotion|>{emo}->{FEAT_EMOTION_COLUMNS[top_emo]}")
                emo = FEAT_EMOTION_COLUMNS[top_emo]
                time.sleep(2)
            except AttributeError:
                continue

        cam.release()
