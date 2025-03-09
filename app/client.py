import os
import streamlit as st
from langserve import RemoteRunnable
from dotenv import load_dotenv


### Chatbot  setup ###
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
chat = RemoteRunnable(f"{BASE_URL}/chat/")


def fetch_ai_response(user_input):
    """
    Fetch the response by sending a request to the backend server.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The model's response.
    """
    response = chat.invoke(
        {
            "input": user_input,
            "chat_history": st.session_state.chat_history,
        },
        {
            "configurable": {
                "session_id": st.session_state.session_id
            }
        },
    )
    
    if isinstance(response, dict) and 'answer' in response:
        return response['answer']
    
    return str(response)
    
