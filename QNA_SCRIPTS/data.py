from typing import List
from langchain.schema import Document
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def getArxiv_docs(wiki_topic_query, num_docs=5) -> List[Document]:
    loader = ArxivLoader(
        wiki_topic_query,
        load_max_docs=num_docs,
        load_all_available_meta=True
    )
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    documents = text_splitter.split_documents(raw_documents)
    return documents