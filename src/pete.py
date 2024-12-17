from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from DetectFace import MyDetectFace

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()

expert_prompt = (
"You are an experienced mental health counselor with expertise in helping individuals with emotional issues"
"Your goal is to provide thoughtful, empathetic, and professional advice to users"
"You are supposed to good at dealing with mental health challenges"
"You should base your responses on psychological principles and practical solutions. "
"Read and understand the users' questions, write responses at high levels of empathic understanding. "
"Limit each response to a minimum of 100 words and a maximum of 200 words. ")
llm_r = chat.send_message(expert_prompt)

mdf = MyDetectFace()

furhat = FurhatRemoteAPI("localhost")
voices = furhat.get_voices()

# Select a character for the virtual Furhat
furhat.set_face(character="Isabel", mask="adult")

# Set the voice of the robot
furhat.set_voice(name='Joanna')

# Have Furhat greet the user
furhat.say(text="hi", blocking=True)


while(True):
    # Listen to the user's response
    response = furhat.listen()

    # Check if listening was successful
    if response.success and response.message:
        emo, face = mdf.getUserEmotion()
        user_r = f'User looks {emo} and user says {response.message}' 
        print("User req:", user_r)
        llm_r = chat.send_message(user_r)
        furhat.say(text=llm_r.text, blocking=True)
