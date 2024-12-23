from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue


class ListenPete():

    def __init__(self, logger):
        self._log = logger
        self._log.info("Starting ListenPete")
        return
    
    def listen(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        while(True):
            response = pete.listen()

            # Check if listening was successful
            if response.success and response.message:
                pete.say_stop()
                queue.put(f"tell|>{response.message}")