from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue


class ListenPete():

    """
    ListenPete is a class that listens to the user and sends the message to the queue.
    """
    SILENCE_TIME = 7

    def __init__(self, logger):
        self._log = logger
        self._log.info("Starting ListenPete")
        return
    
    def listen(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        silence_time:int = 0
        while(True):
            response = pete.listen()

            # Check if listening was successful
            if response.success and response.message:
                pete.say_stop()
                queue.put(f"tell|>{response.message}")
                silence_time = 0
                self._log.info(f"ListenPete: {response.message}")
            elif(response.success):
                silence_time = silence_time+1
                if(silence_time > self.SILENCE_TIME):
                    queue.put(f"ask|>silence")
                    silence_time = 0