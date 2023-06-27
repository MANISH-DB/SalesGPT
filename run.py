import argparse
import streamlit as st
from streamlit_chat import message

import os
import json

from salesgpt.agents import SalesGPT
from langchain.chat_models import AzureChatOpenAI

st.set_page_config(
    page_title="Sleep Haven",
    page_icon=":robot:"
)
st.header("Sleep Haven")
st.subheader("Achieve better night's sleep")

llm = AzureChatOpenAI(deployment_name = os.environ["DEPLOYMENT_NAME"])

sales_agent = SalesGPT.from_llm(llm, verbose=False,
                            salesperson_name="Ted Lasso",
                            salesperson_role="Sales Representative",
                            company_name="Sleep Haven",
                            company_business='''Sleep Haven 
                            is a premium mattress company that provides
                            customers with the most comfortable and
                            supportive sleeping experience possible. 
                            We offer a range of high-quality mattresses,
                            pillows, and bedding accessories 
                            that are designed to meet the unique 
                            needs of our customers.''')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    st.session_state['history'] = []
    sales_agent.seed_agent()
    sales_agent.step()
    st.session_state.generated.append(sales_agent.conversation_history[-1][:-13])
    st.session_state.history.append(sales_agent.conversation_history[-1])

if 'past' not in st.session_state:
    st.session_state['past'] = []

#get user input
if 'words' not in st.session_state:
    st.session_state.words = ''

def submit():
    st.session_state.input = ''
    st.session_state.words = st.session_state.input

def get_text():
    input_text = st.text_input("You: ", key=input, on_change=submit)
    return input_text

input_text = get_text()
st.session_state.past.append(input_text)

if input_text:
    sales_agent.human_step(input_text)
    sales_agent.conversation_history = st.session_state.history
    st.session_state.history.append("User: " + input_text + " <END_OF_TURN>")
    print(st.session_state.history[-1])
    sales_agent.step()
    generated_message = sales_agent.conversation_history[-1]
    st.session_state.generated.append(sales_agent.conversation_history[-1][:-13])
    # st.write(st.session_state)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

