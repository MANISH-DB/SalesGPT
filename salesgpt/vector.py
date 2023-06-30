from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

import os
# import your OpenAI key (put in your .env file)
with open('.env','r') as f:
    env_file = f.readlines()
envs_dict = {key.strip("'") :value.strip("\n") for key, value in [(i.split('=')) for i in env_file]}
os.environ['OPENAI_API_KEY'] = envs_dict['OPENAI_API_KEY']
os.environ["OPENAI_API_TYPE"] = envs_dict["OPENAI_API_TYPE"]
os.environ["OPENAI_API_BASE"] = envs_dict["OPENAI_API_BASE"]
os.environ["OPENAI_API_VERSION"] = envs_dict["OPENAI_API_VERSION"]
os.environ["DEPLOYMENT_NAME"] = envs_dict["DEPLOYMENT_NAME"]

loader = TextLoader("logs-kelvintsim-salesgpt-main-run.py-2023-06-27T07_24_51.061Z.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(docs[-1])
print(len(docs))
embeddings = OpenAIEmbeddings(deployment="chatbot-embedding-ada-dev")

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER"),
    host=os.environ.get("PGVECTOR_HOST"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE"),
    user=os.environ.get("PGVECTOR_USER"),
    password=os.environ.get("PGVECTOR_PASSWORD"),
)   

db = PGVector.from_documents(
    embedding=embeddings,
    documents=[docs[0]],
    collection_name="state_of_the_union",
    connection_string=CONNECTION_STRING,
)
query = "What products do sleep haven offer?"
db.similarity_search_with_score