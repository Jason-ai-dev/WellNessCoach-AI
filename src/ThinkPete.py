from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from multiprocessing import Queue
import pandas as pd
import time

'''
The thinking and talking taks.
To start a conversation say "I am " and your name.
To end a conversation say something with "bye"
'''
class ThinkPete():

    # This is how Pete looks when he smiles
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

    # This is how Pete looks when he is qurious
    qurious_pete = {"frames": [
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
    
    # This is how Pete respond when the user is talking
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
        # Configure Gemini API key and initialize the chat model.
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
        self._chat = self._model.start_chat()
        llm_r = self._chat.send_message(expert_prompt)
        self._log = logger
        self._log.info("Starting ThinkPete")
        return

    '''
    This function will use the emotion information produced by SeePete and the speech-to-text
    infomation produced by ListenPete. The text infomation will process further and activate 
    Pete for new discussion or end an ongoing discussion by processing text infomation.

    @param queue message passing queue for inter-process communications
    @param pete Furhat API handler for pete
    '''
    def say(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        """
        ThinkPete listens to the user and generates a response using the Gemini API.
        :param queue: The queue to listen to the user.
        :param pete: The FurhatRemoteAPI object.
        :return: None
        """
        # Have Furhat greet the user
        
        userName = ""
        askName = True
        getHist = False
        while(True):

            # waite for new discussion. Waite for the activation key word "i am"
            if(askName):
                pete.say(text="hi! what is your name?", blocking=True)
                askName = False
            while(userName == ""):
                qval = queue.get()
                if('i am' in qval.lower()):
                    userName = qval.split()[-1]
                    getHist = True

            # Fetch user's discussion history if available
            if(getHist):
                user_history = self.rememberUser(userName)
                llm_r = self._chat.send_message(user_history)
                pete.say(text=llm_r.text, blocking=False)
                # empty the queue before starting new discussion
                while not queue.empty():
                    queue.get()
                self._log.info(f"Gemini response: {llm_r.text}")
                getHist = False

            # fetch queue data from other processes
            user_behav = f'{queue.get()}'
            self._log.info(f"User behaviour: {user_behav}")

            # detect the source and value by using "|>"" delemiter
            src, val = user_behav.split("|>")

            # If the source is SeePete and it is an emotion change on face, process that data
            if(src == "emotion"):
                emo1,emo2 = val.split("->")
                user_r = f'user changes his facial expression from {emo1} to {emo2}' 

                # if user is happy Pete will smile
                if(emo2 == 'happiness'):
                    pete.gesture(body=self.happy_pete)

            # process data from ListenPete process
            elif(src == "tell"):
                # if user say a phrase with bye, then end the discussion
                if('bye' in val.lower()):
                    # generate the summary of the discussion
                    llm_r = self._chat.send_message("generate a summary of this discussion")
                    
                    # Save the user history against user's name
                    self.logUser(userName, llm_r.text)
                    llm_r = self._chat.send_message("end this discussion")

                    # reset loop parameters to the initial conditions
                    userName = ""
                    askName = True
                    getHist = False

                user_r = f'user tells {val}'
                # Pete reacts when user is speaking
                pete.gesture(body=self.accepting_pete)
                # empty the message queue because when user is talking the facial emotions are not
                # accurate
                while not queue.empty():
                    queue.get()

            # Process ListenPete infomations
            elif(src == "ask"):
                # react if the user is being silence for a while
                if(val == "silence"):
                    pete.gesture(body=self.qurious_pete)
                    user_r = 'user is being silent for a while.'

            # Send user behaviour to the Gemini and ge the response
            llm_r = self._chat.send_message(user_r)
            self._log.info(f"Gemini response: {llm_r.text}")
            pete.say(text=llm_r.text, blocking=False)

            # if the user used bye in a phrase end the discussion and empty the message queue
            if('bye' in val.lower()):
                time.sleep(4)
                while not queue.empty():
                    queue.get()
                while queue.empty():
                    pass

    '''
    This function reads a list of users' discussion summary and fetch the matching summary for the
    given name.

    @param name the name of the user
    '''
    def rememberUser(self, name:str)->str:
        try:
            userHist = pd.read_csv('userHist.csv')
        except FileNotFoundError:
            userHist = pd.DataFrame({'name':[],'summary':[]})
            userHist.to_csv('userHist.csv', index=False)
        hist = f"{name} is a new user"
        if((userHist.name == name).any()):
            hist = f'The summary of the last session with user {name} is {userHist[userHist.name == name].summary.values[0]}. Talk to {name} about how he is doing.'
        self._log.info(f"load user hist: {hist}")
        return hist


    '''
    This function will save the user history after ending a discussion.

    @param name name of the user.
    @param summary the summary of the discussion.
    '''
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