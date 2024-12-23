from furhat_remote_api import FurhatRemoteAPI
from SeePete import SeePete
from ListenPete import ListenPete
from ThinkPete import ThinkPete
from multiprocessing import Process, Queue, log_to_stderr
import logging


expert_prompt = (
"You are an experienced mental health counselor with expertise in helping individuals with emotional issues"
"Your goal is to provide thoughtful, empathetic, and professional advice to users"
"You are supposed to good at dealing with mental health challenges"
"You should base your responses on psychological principles and practical solutions. "
"Read and understand the users' questions, write responses at high levels of empathic understanding. "
"Limit each response to a minimum of 100 words and a maximum of 200 words. ")

furhat = FurhatRemoteAPI("localhost")
gem_key = ""
logger = logging.getLogger(__name__)
logging.basicConfig(filename='pete.log', level=logging.INFO)

# mdf = MyDetectFace()
def peteListen(queue:Queue)->None:
    mlp = ListenPete(logger=logger)
    mlp.listen(queue=queue, pete=furhat)
    return 

def peteThink(queue:Queue)->None:
    mtp = ThinkPete(expert_prompt=expert_prompt, api_key=gem_key, logger=logger)
    mtp.say(queue=queue, pete=furhat)
    return 

def peteSee(queue:Queue)->None:
    msp = SeePete(logger=logger)
    msp.observeUser(queue=queue, pete=furhat)
    return

    
if __name__ == '__main__':
    voices = furhat.get_voices()

    # Select a character for the virtual Furhat
    furhat.set_face(character="Isabel", mask="adult")

    # Set the voice of the robot
    furhat.set_voice(name='Joanna')

    # Have Furhat greet the user
    furhat.say(text="hi", blocking=True)

    q = Queue()
    p_listen = Process(target=peteListen, args=(q,))
    p_think = Process(target=peteThink, args=(q,))
    p_see = Process(target=peteSee, args=(q,))
    p_listen.start()
    p_think.start()
    p_see.start()
    p_listen.join()
    p_think.join()
    p_see.join()