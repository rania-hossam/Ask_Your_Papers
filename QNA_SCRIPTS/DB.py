from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from typing import List



def get_vectorstore(documents:List[Document]=None):
    """let's create the Faiss vcstore"""

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
    # create the vector store database
    db = FAISS.from_documents(embeddings=embeddings,
                              documents=documents,
                                )

    return db

