import json
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
import salesgpt_woo.db_retriver as data
from salesgpt.agents import SalesGPT
import streamlit as st
import os
from langchain.prompts import PromptTemplate

class csAgent:
    memory = []
    chain = None
    def __init__(self):
        #config sales agent
        file = open("agent_setup.json", "r")
        config=json.load(file)
        prompt_template = """You are a customer service officer from Woo who named Kelvin. Based on following {history} and reference from {docs_str}, reply the {input}"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["input", "history", "docs_str"])
        llm = AzureChatOpenAI(deployment_name = os.environ["DEPLOYMENT_NAME"])
        self.chain = LLMChain(llm=llm, prompt=prompt)
    def run(self, input):
        docs_str = data.documents_to_string(data.get_documents(data.get_db(), input))
        return self.chain.run(input=input, history=self.memory, docs_str=docs_str)

#initialize agent
def init():
    return csAgent()

def get_generated_message(agent, input_text, history):
    agent.memory = history
    result = agent.run(input=input_text)
    agent.memory.append("agent: " + result)
    print(result)
    return result

def get_latest_message(agent):
    if len(agent.memory) > 0:
        return agent.memory[-1]
    else:
        return ""

def get_clean_message(message):
    return message[:-13]