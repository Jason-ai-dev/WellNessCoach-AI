from furhat_remote_api import FurhatRemoteAPI
from SeePete import SeePete
from ListenPete import ListenPete
from ThinkPete import ThinkPete
from multiprocessing import Process, Queue, log_to_stderr
import logging

'''
The activation key word for Pete is "I am " 
The discussion ends when used the key word "bye" in a phrase.
'''

# The expert prompt to narrow down the search space for Gemini
expert_prompt = (
"Your name is Pete."
"You are an experienced mental health counselor with expertise in helping individuals with emotional issues"
"Your goal is to provide thoughtful, empathetic, and professional advice to users"
"You are supposed to good at dealing with mental health challenges"
"You should base your responses on psychological principles and practical solutions. "
"Read and understand the users' questions, write responses at high levels of empathic understanding. "
"Limit each response to a minimum of 10 words and a maximum of 100 words. ")

# Initiate gloable parameters
# Connect to the Furhat Localhost robot. 
furhat = FurhatRemoteAPI("localhost")

# Add your Gemini API key here
gem_key = "AIzaSyCrUvfeeZ0pu2Vy6idPrxmm0vGZGtTLOT8"

# Logger object. Used for logging background tasks
logger = logging.getLogger(__name__)
logging.basicConfig(filename='pete.log', level=logging.INFO)

'''
This function wraps the Listening task of Pete
@param queue Queue for message passing between threads
'''
def peteListen(queue:Queue)->None:
    """
    Function to listen to the user's question
    :param queue: Queue object to store the user's question
    :return: None
    """
    mlp = ListenPete(logger=logger)
    mlp.listen(queue=queue, pete=furhat)
    return 

'''
This function wraps the Think and Talk task of Pete
@param queue Queue for message passing between threads
'''
def peteThink(queue:Queue)->None:
    """
    Function to think about the user's question and provide a response
    :param queue: Queue object to store the user's question
    :return: None
    """
    mtp = ThinkPete(expert_prompt=expert_prompt, api_key=gem_key, logger=logger)
    mtp.say(queue=queue, pete=furhat)
    return 

'''
This function wraps the Seeing task of Pete
@param queue Queue for message passing between threads
'''
def peteSee(queue:Queue)->None:
    """
    Function to observe the user's facial expressions
    :param queue: Queue object to store the user's expression
    :return: None
    """
    msp = SeePete(logger=logger)
    msp.observeUser(queue=queue, pete=furhat)
    return

    
if __name__ == '__main__':
    voices = furhat.get_voices()

    # Select a face for Pete
    furhat.set_face(character="James", mask="adult")

    # Set the voice of Pete
    furhat.set_voice(name='Matthew')

    # create message passing queue
    q = Queue()

    # Initialize the subprocessors for Listen, Think and See
    p_listen = Process(target=peteListen, args=(q,))
    p_think = Process(target=peteThink, args=(q,))
    p_see = Process(target=peteSee, args=(q,))
    p_listen.start()
    p_think.start()
    p_see.start()
    p_listen.join()
    p_think.join()
    p_see.join()