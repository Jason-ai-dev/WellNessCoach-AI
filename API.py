from furhat_remote_api import FurhatRemoteAPI
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from googleapiclient.discovery import build
import requests
import time 
import threading 


# Replace with your actual API key (don't paste it here for security reasons)
API_KEY = "AIzaSyCrUvfeeZ0pu2Vy6idPrxmm0vGZGtTLOT8"

# Define the Gemini model endpoint
MODEL_ENDPOINT = "gemini-1.5-flash-latest"

def generate_response(text_prompt):
    """Sends a request to Gemini using an API key and generateContent."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ENDPOINT}:generateContent?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": text_prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()

        # Correctly extract the text from the response
        if "candidates" in response_json and response_json["candidates"]:
            first_candidate = response_json["candidates"][0]
            if "content" in first_candidate and "parts" in first_candidate["content"]:
                parts = first_candidate["content"]["parts"]
                generated_text = "".join([part.get("text", "") for part in parts])
                return generated_text
            else:
                print(f"Unexpected 'content' or 'parts' structure: {first_candidate}")
                return "Unexpected response format from API (content or parts missing)."
        else:
            print(f"Unexpected 'candidates' structure: {response_json}")
            return "Unexpected response format from API (candidates missing)."


    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the API: {e}")
        if response is not None:
            print(f"Response content: {response.text}")  # Print the content of the response error for debugging
        return f"API request failed: {e}"

furhat = FurhatRemoteAPI("localhost")
speaking_event = threading.Event()  # Event flag to track Furhat's speech

def furhat_speak(text):  # Define the function BEFORE it's used
    """Speaks the given text and manages the speaking event."""
    speaking_event.set()
    try:
        furhat.say(text=text)
    except Exception as e:
        print(f"Error during Furhat speech: {e}")
    finally:
        speaking_event.clear()
        
if __name__ == "__main__":
    while True:
        try:
            # Wait until Furhat is not speaking
            while speaking_event.is_set():
                time.sleep(0.1)  # Check every 100ms

            # Now it's safe to listen
            result = furhat.listen()

            if result and result.message:
                text_prompt = result.message
                print(f"User said: {text_prompt}")

                response = generate_response(text_prompt)
                print(f"Gemini responded: {response}")

                # Use a separate thread for speaking to avoid blocking
                speak_thread = threading.Thread(target=furhat_speak, args=(response,))
                speak_thread.start()

                # Perform gesture (optional) - do this after starting the speak thread
                furhat.gesture(body={
                    "frames": [
                        {"time": [0.33], "params": {"BLINK_LEFT": 1.0}},
                        {"time": [0.67], "params": {"reset": True}}
                    ],
                    "class": "furhatos.gestures.Gesture"
                })
            elif result is None:
                print("No speech detected.")
            
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)
            
# Get the users detected by the robot 
#users = furhat.get_users()

# Attend the user closest to the robot
#furhat.attend(user="CLOSEST")

# Attend a user with a specific id
#furhat.attend(userid="virtual-user-1")

# Attend a specific location (x,y,z)
#furhat.attend(location="0.0,0.2,1.0")

# Set the LED lights
#furhat.set_led(red=200, green=50, blue=50)