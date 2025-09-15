from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import os

# from langchain_huggingface import HuggingFaceEmbeddings

# Extract data from PDF files in the given directory

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    "giveen list of documents obejcts return new list of documents objects containig only source in metdata "
    minimal_docs: List[Document] = []
    for doc in docs:
        src=doc.metadata.get('source')
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs

# split the documents into smaller chunks

def text_splitting(minimal_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(minimal_documents)
    return docs

## download the sentence transformer model

def download_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

