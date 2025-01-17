from furhat_remote_api import FurhatRemoteAPI
from multiprocessing import Queue

'''
The listening task.
'''
class ListenPete():

    # number of listen API timeouts to detect as being silence
    SILENCE_TIME = 7

    def __init__(self, logger):
        self._log = logger
        self._log.info("Starting ListenPete")
        return
    
    '''
    This function will continouesly trigger listen API so that Pete is listen always.
    If a speech is detected, then interrupts Pete if he is already talking. The speech 
    to text results received from the API is put to the message queue for ThinkPete to
    use. 

    @param queue message passing queue for inter-process communications
    @param pete Furhat API handler for pete
    '''
    def listen(self, queue:Queue, pete:FurhatRemoteAPI) -> None:
        silence_time:int = 0
        while(True):
            response = pete.listen()

            # Check if listening was successful
            if response.success and response.message:
                # interrupt Pete if he is already talking
                pete.say_stop()
                queue.put(f"tell|>{response.message}")
                silence_time = 0
                self._log.info(f"ListenPete: {response.message}")
            # count silence event counts
            elif(response.success):
                silence_time = silence_time+1
                if(silence_time > self.SILENCE_TIME):
                    queue.put(f"ask|>silence")
                    silence_time = 0