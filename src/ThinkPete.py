from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from multiprocessing import Queue
import pandas as pd

class ThinkPete():

    """
    ThinkPete is a class that uses the Gemini API to generate responses to the user.
    """
    happy_pete = {"frames": [
        {"time":[0.32,0.64],
        "persist":False, 
        "params":{
            "BROW_UP_LEFT":1,
            "BROW_UP_RIGHT":1,
            "SMILE_OPEN":0.4,
            "SMILE_CLOSED":0.7
            }
        },
        {
        "time":[1.96],
        "persist":False, 
        "params":{
            "reset":True
            }
        }
    ], 
    "class": "furhatos.gestures.Gesture"}

    quriour_pete = {"frames": [
        {
        "time": [
            1.0, 6.0
        ],
        "params": {
            "NECK_ROLL": 10.0,
            "NECK_PAN": -8.0
        }
        },
        {
        "time":[7],
        "persist":False, 
        "params":{
            "reset":True
            }
        }
    ], 
    "class": "furhatos.gestures.Gesture"}
    
    accepting_pete = {"frames": [
        {
        "time": [
            0.3, 0.6
        ],
        "params": {
            "NECK_TILT": 10.0,
            "NECK_PAN": -5.0,
            "BLINK_LEFT": 0.3,
            "BLINK_RIGHT": 0.3
        }
        },
        {"time": [
            0.9, 1.2
        ],
        "params": {
            "NECK_TILT": 0.0,
            "NECK_PAN": -5.0,
            "BLINK_LEFT": 0.0,
            "BLINK_RIGHT": 0.0
        }
        },
        {"time": [
            1.5, 2.0
        ],
        "params": {
            "NECK_TILT": 10.0,
            "NECK_PAN": -5.0
        }
        },
        {
        "time":[3],
        "persist":False, 
        "params":{
            "reset":True
            }
        }
    ], 
    "class": "furhatos.gestures.Gesture"}


    def __init__(self, expert_prompt, api_key, logger):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
        self._chat = self._model.start_chat()
        llm_r = self._chat.send_message(expert_prompt)
        self._log = logger
        self._log.info("Starting ThinkPete")
        return

    def say(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        """
        ThinkPete listens to the user and generates a response using the Gemini API.
        :param queue: The queue to listen to the user.
        :param pete: The FurhatRemoteAPI object.
        :return: None
        """
        # Have Furhat greet the user
        pete.say(text="hi! what is your name?", blocking=True)
        userName = ""
        while(userName == ""):
            qval = queue.get()
            if('i am' in qval.lower()):
                userName = qval.split()[-1]
        user_history = self.rememberUser(userName)
        llm_r = self._chat.send_message(user_history)
        pete.say(text=llm_r.text, blocking=False)
        while not queue.empty():
            queue.get()
        self._log.info(f"Gemini response: {llm_r.text}")
        while(True):
            user_behav = f'{queue.get()}'
            self._log.info(f"User behaviour: {user_behav}")
            src, val = user_behav.split("|>")
            if(src == "emotion"):
                emo1,emo2 = val.split("->")
                user_r = f'user changes his facial expression from {emo1} to {emo2}' 
                if(emo2 == 'happiness'):
                    pete.gesture(body=self.happy_pete)
            elif(src == "tell"):
                if('bye' in val.lower()):
                    llm_r = self._chat.send_message("generate a summary of this discussion")
                    self.logUser(userName, llm_r.text)
                    llm_r = self._chat.send_message("end this discussion")
                user_r = f'user tells {val}'
                pete.gesture(body=self.accepting_pete)
                while not queue.empty():
                    queue.get()
            elif(src == "ask"):
                if(val == "silence"):
                    pete.gesture(body=self.quriour_pete)
                    user_r = 'user is being silent for a while.'
            llm_r = self._chat.send_message(user_r)
            self._log.info(f"Gemini response: {llm_r.text}")
            pete.say(text=llm_r.text, blocking=False)

    def rememberUser(self, name:str)->str:
        """
        Load the user history from a file.
        :param name: The name of the user.
        :return: The user history.
        """
        userHist = pd.read_csv('userHist.csv')
        hist = f"{name} is a new user"
        if((userHist.name == name).any()):
            hist = f'The summary of the last session with user {name} is {userHist[userHist.name == name].summary.values[0]}. Ask about any improvements from {name}'
        self._log.info(f"load user hist: {hist}")
        return hist

    def logUser(self, name:str, summary:str)->None:
        """
        Save the user history to a file.
        :param name: The name of the user.
        :param summary: The summary of the user history (what was talked about).
        :return: None
        """
        userHist = pd.read_csv('userHist.csv')
        self._log.info(f"log user hist: {summary}")
        try:
            userHist._set_value(userHist.index[userHist.name == name][0],'summary',summary)
        except IndexError:
            userHist.loc[len(userHist)] = [name, summary]
        userHist.to_csv('userHist.csv', index=False)
        return