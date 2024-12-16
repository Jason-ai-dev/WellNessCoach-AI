from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from DetectFace import MyDetectFace

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()
llm_r = chat.send_message("what is ML?")

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
        user_r = f'User looks {emo} and user ask {response.message}, what would you say?' 
        print("User req:", user_r)
        llm_r = chat.send_message(user_r)
        furhat.say(text=llm_r.text, blocking=True)
