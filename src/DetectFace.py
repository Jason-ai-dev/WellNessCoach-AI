import numpy as np
import pandas as pd
import cv2
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS

class MyDetectFace():

    _myImg:np.ndarray 
    _myLabledImg:np.ndarray 
    _predDone:bool
    _au:pd.DataFrame
    _fname:str
    
    

    def __init__(self):
        # self._detector = Detector(device="auto")
        self._detector = Detector(face_model='faceboxes',emotion_model='resmasknet', landmark_model="pfld", au_model='svm', device='auto')
        # self._detector = Detector(face_model='retinaface',emotion_model='svm', landmark_model="mobilenet", au_model='svm', device='auto')
        au_names = self._detector.info['au_presence_columns']
        au_names.insert(0, 'face')
        au_names.insert(0, 'file')
        self._au = pd.DataFrame(columns=au_names)
        return

    def getImg(self)->np.ndarray:
        return self._myImg
    
    def getAUs(self)->pd.DataFrame:
        return self._au

    def setImg(self, img:np.ndarray, fname:str)->None:
        self._myImg = img
        self._myLabledImg = img
        self._predDone = False
        self._fname = fname
        return

    def predImg(self)->None:
        faces = self._detector.detect_faces(self._myImg)
        landmarks = self._detector.detect_landmarks(self._myImg, faces)
        emotions = self._detector.detect_emotions(self._myImg, faces, landmarks)
        au = self._detector.detect_aus(self._myImg, landmarks=landmarks)

        # The functions seem to assume a collection of images or frames. We acces "frame 0".
        faces = faces[0]
        landmarks = landmarks[0]
        emotions = emotions[0]

        strongest_emotion = emotions.argmax(axis=1)
        f_index = 0
        for (face, top_emo, iau) in zip(faces, strongest_emotion, au[0]):
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(self._myLabledImg, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            cv2.putText(self._myLabledImg, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
            iau = iau.tolist()
            iau.insert(0, f_index)
            iau.insert(0, self._fname.split('.')[0])
            f_index+=1
            self._au.loc[len(self._au)] = iau

        self._predDone = True
        return

    def getLableImg(self)->np.ndarray:
        if(self._predDone):
            return self._myLabledImg
        else:
            return None
        
    def getUserEmotion(self) -> str|tuple:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emo = ""
        face_loc = ()
        try:

            ret, frame = cam.read()
            if not ret:
                print("OpenCV found an error reading the next frame.")

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

            emo = FEAT_EMOTION_COLUMNS[top_emo]
            face_loc = (int((x0+x1)/2), int((y0+y1)/2))

        finally:
            cam.release()
            return emo, face_loc
