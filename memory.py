import streamlit as st

def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_to_memory(user_query, bot_response):
    st.session_state.chat_history.append({"user": user_query, "bot": bot_response})

def get_memory():
    return st.session_state.chat_history