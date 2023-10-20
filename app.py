import streamlit as st
from streamlit_chat import message
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from glob import glob
import pickle
from QNA_SCRIPTS.LLM import load_llm
from QNA_SCRIPTS.DB import get_vectorstore
from QNA_SCRIPTS.data import getArxiv_docs
from QNA_SCRIPTS.promot import create_prompt_template

st.set_page_config(page_title='Llama2-Chatbot')
st.header('Custom Llama2-Powered Chatbot :robot_face:')
from typing import List
from langchain.schema import Document
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




def create_qa_chain():

    # load the llm, vector store, and the prompt

    llm = load_llm()
    DB = db
    prompt = create_prompt_template()

    # create the qa_chain
    retriever = DB.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']


def get_user_input():

    # get the user query
    input_text = st.text_input('Ask me anything about the use of computer vision in sports!', "", key='input')
    return input_text




# create the qa_chain
qa_chain = create_qa_chain()


# create empty lists for user queries and responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# get the user query
user_input = get_user_input()



if user_input:
    arxiv_documents = getArxiv_docs(user_input)
    db = get_vectorstore(arxiv_documents)
    st.session_state['arxiv_db'] = db
    
    # generate response to the user input
    response = generate_response(query=user_input, qa_chain=qa_chain)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


# display conversaion history (if there is one)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
