import streamlit as st
from streamlit_chat import message
import time
import config
from chat import qa_chat
from logos.utils import read_logo
from st_pages import Page, show_pages
st.set_page_config(page_title="Chatbot", page_icon="🧑‍🚀")

img_b64 = read_logo(config.CONF[config.CHATBOT]["logo"])

st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{img_b64}');
                background-repeat: no-repeat;
                padding-top: 50px;
                background-position: 100px 50px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

chatbot_pages =     [

        Page("Chatbot.py", "Chatbot"),

    ]


if "modules" in config.CONF[config.CHATBOT].keys():

    chatbot_pages.extend([Page(x) for x in config.CONF[config.CHATBOT]["modules"]])

show_pages(chatbot_pages)

    
#Creating the chatbot interface
st.title(config.CONF[config.CHATBOT]["name"])

@st.cache_data()
def load_data(): 
    qa_chat.process_document_and_qa_chain()       

load_data()

def main():

    chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figs" in message:
                for im in message["figs"]: 
                    st.image(im, caption='Examples')
            if len(st.session_state.messages) >= 2:
                chat_history = [(st.session_state.messages[i]['content'], 
                        st.session_state.messages[i+1]['content']) for i in range(0,
                            len(st.session_state.messages), 2)]

                    
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
    
            assistant_response, imgs= qa_chat.answer(prompt,chat_history=chat_history)
     
            
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            for im in imgs: 
                st.image(im, caption='Examples')
                time.sleep(0.05)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response,"figs": imgs})
        

# Run the app
if __name__ == "__main__":
    main()