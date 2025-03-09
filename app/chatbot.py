import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from client import fetch_ai_response 


### APP Page configuration ###
st.set_page_config(page_title="AI Chatbot Assistant", page_icon="🤖")
st.title("AI Chatbot Assistant 🤖")


# -------------------------------------------------
# Session State Management and Helper functions
# -------------------------------------------------

def initialize_session():
    """Initialize session variables for session_id and chat_history."""
    # Generate a unique session_id for the session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    # Initialize chat history if not already in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]

def add_message_to_history(message):
    """Add a message to the chat history in Streamlit's session state."""
    st.session_state.chat_history.append(message)


def display_chat_history():
    """Render the chat history in chat format."""
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)


# -------------------------------------------------
# Interactive UI Chatbot
# -------------------------------------------------

def main():
    """Main function to manage the interactive chat."""
    
    initialize_session() # Initialize session variables
    
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add the User's message to the chat history
        add_message_to_history(HumanMessage(content=user_input))
        
        # Fetch the AI's response and add it to the chat history
        ai_response = fetch_ai_response(user_input)
        add_message_to_history(AIMessage(content=ai_response))
        
        # Force a rerun to update the chat display
        st.rerun()
    
    display_chat_history()


if __name__ == "__main__":
    main()