from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from multiprocessing import Queue

def say(queue:Queue, pete:FurhatRemoteAPI) -> None:
        while(True):
            saythis = f"this is thinking: {queue.get()}"
            pete.say(saythis)

class ThinkPete():

    def __init__(self, expert_prompt, api_key, logger):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")
        self._chat = self._model.start_chat()
        llm_r = self._chat.send_message(expert_prompt)
        self._log = logger
        self._log.info("Starting ThinkPete")
        return

    def say(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        while(True):
            user_behav = f'{queue.get()}'
            self._log.info(f"User behaviour: {user_behav}")
            src, val = user_behav.split("|>")
            if(src == "emotion"):
                emo1,emo2 = val.split("->")
                user_r = f'user changes his facial expression from {emo1} to {emo2}' 
            elif(src == "tell"):
                 user_r = f'user tells {val}'
            llm_r = self._chat.send_message(user_r)
            self._log.info(f"Gemini response: {llm_r.text}")
            pete.say(text=llm_r.text, blocking=False)
            
