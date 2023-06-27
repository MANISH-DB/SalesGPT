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
    sales_agent.step()
    st.session_state.generated.append(sales_agent.conversation_history[-1])
    st.session_state.history.append(sales_agent.conversation_history[-1])
    message(st.session_state["generated"], key="initial")
    print("no history")

if not st.session_state.history:
    sales_agent.seed_agent()

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ")
    return input_text 

human_input = get_text()
st.session_state.past.append(human_input)

if human_input:
    sales_agent.human_step(human_input)
    sales_agent.conversation_history = st.session_state.history
    st.session_state.history.append("User: " + human_input + " <END_OF_TURN>")
    sales_agent.step()
    st.session_state.generated.append(sales_agent.conversation_history[-1])
    st.session_state.history.append(sales_agent.conversation_history[-1])
    st.write(st.session_state)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

