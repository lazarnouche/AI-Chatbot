import streamlit as st
from streamlit_chat import message
import time

# from chat import QA_Chat

#Creating the chatbot interface
st.title(" Test ")

   
def main():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = "This is test"
     
            
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.001)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            for im in imgs: 
                st.image(im)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response,"figs": imgs})
# Run the app
if __name__ == "__main__":
    main()