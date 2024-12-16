from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai

genai.configure(api_key="AIzaSyDW9WPa7Oe2-S_SOdi4VzajU9M1YVKzBTs")
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()
llm_r = chat.send_message("what is ML?")

furhat = FurhatRemoteAPI("localhost")
voices = furhat.get_voices()

# Select a character for the virtual Furhat
furhat.set_face(character="Isabel", mask="adult")

# Set the voice of the robot
furhat.set_voice(name='Joanna')

# Have Furhat greet the user
furhat.say(text="hi", blocking=False)


while(True):
    # Listen to the user's response
    response = furhat.listen()

    # Check if listening was successful
    if response.success and response.message:
        print("User said:", response.message)
        user_r = response.message
        llm_r = chat.send_message(user_r)
        furhat.say(text=llm_r.text, blocking=True)
