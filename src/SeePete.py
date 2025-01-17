import time
import cv2
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue

'''
The Seeing task
'''
class SeePete():

    # Scale factor for the depth (z value) for robot attendance
    DEPTH_PARAM = 0.3

    def __init__(self, logger):
        # Initiate the Py-Feat detection model 
        self._detector = Detector(face_model='faceboxes',emotion_model='resmasknet', landmark_model="pfld", au_model='svm', device='auto')
        au_names = self._detector.info['au_presence_columns']
        au_names.insert(0, 'face')
        au_names.insert(0, 'file')
        self._log = logger
        self._log.info("Starting SeePete")
        return
    
    '''
    This funciton aquires the camera and process the image data to get the location of the 
    user's face and emotion. It will set Pete's attention to the calculated location while
    passing the emotion change information to the ThinkPete. If the user initially has emotion
    A and change it to emotion B, the change will send through the message queue but if the use
    changed the emotion from B to A immediately after, this event will not pass.

    @param queue message passing queue for inter-process communications
    @param pete Furhat API handler for pete
    '''
    def observeUser(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        # Get the camera
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        emo = ""
        old_emo = ""
        while True:
            # read the image
            ret, frame = cam.read()
            for i in range(10):
                ret, frame = cam.read()

            if not ret:
                self._log.warning("OpenCV found an error reading the next frame.")
                break

            try:
                # Do Py-Feat emotion detection
                faces = self._detector.detect_faces(frame)

                # The functions seem to assume a collection of images or frames. We access "frame 0".
                faces = faces[0]
                landmarks = landmarks[0]
                emotions = emotions[0]

                strongest_emotion = emotions.argmax(axis=1)

                # here we assume only one user is present in the frame
                for (face, top_emo) in zip(faces, strongest_emotion):
                    (x0, y0, x1, y1, p) = face
                    
                # calculate the x,y,z values based on the frame infomations and the face box infomations
                face_loc = ((cam.get(cv2.CAP_PROP_FRAME_WIDTH)/2-(x0+x1)/2)/cam.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                (cam.get(cv2.CAP_PROP_FRAME_HEIGHT)/2-(y0+y1)/2)/cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                self.DEPTH_PARAM*cam.get(cv2.CAP_PROP_FRAME_HEIGHT)/(max((abs(y1-y0), 0.01,))))
                self._log.info(f"looking at: {face_loc}")
                
                # Send attend infomations to Pete
                pete.attend(location=f"{face_loc[0]},{face_loc[1]},{face_loc[2]}")
                
                # Send only the forware emotion changes. 
                # If the user initially has emotion
                # A and change it to emotion B, the change will send through the message queue but if the use
                # changed the emotion from B to A immediately after, this event will not pass.
                if((emo != FEAT_EMOTION_COLUMNS[top_emo]) and (emo != "") and (FEAT_EMOTION_COLUMNS[top_emo] != old_emo)):
                    old_emo = emo
                    queue.put(f"emotion|>{emo}->{FEAT_EMOTION_COLUMNS[top_emo]}")
                emo = FEAT_EMOTION_COLUMNS[top_emo]

                # This sleep is needed to avoid filling the message queue with facial emotion info
                time.sleep(2)
            except AttributeError:
                # skip if face is not detected 
                self._log.info("SeePete: face not detected.")
                continue

        cam.release()