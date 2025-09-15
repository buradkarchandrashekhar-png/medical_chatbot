from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from langchain_google_genai import ChatGoogleGenerativeAI

from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_splitting,
    download_model,
)
from src.prompt import *  # uses `system_prompt`

# ---------------------------------------------------------------------
# Flask & env
# ---------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Make available to libraries that read os.environ directly
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""

# ---------------------------------------------------------------------
# Pinecone / Vector store
# ---------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = "medical-chatbot"

embeddings = download_model()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ---------------------------------------------------------------------
# LLM & RAG chain
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg  # keeping your original variable usage
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response :", response["answer"])
    return str(response["answer"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
