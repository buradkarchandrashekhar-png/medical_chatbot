from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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

# IMPORTANT: make sure this import matches your file name
# If your file is src/prompt_template.py, import as below:
from src.prompt import * 
# If your file is actually src/prompt.py, change to:
# from src.prompt import system_prompt

# ---------------------------------------------------------------------
# Flask & env
# ---------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or ""
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or ""

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------------------------------------------------------------
# Pinecone / Vector store
# ---------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medical-chatbot"

embeddings = download_model()  # now uses langchain_huggingface

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ---------------------------------------------------------------------
# LLM, Prompt (with history), Memory, and Conversational RAG
# ---------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Include chat history via placeholder.
# ConversationalRetrievalChain expects: question, chat_history, context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("user", "{question}"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# Build conversational RAG without passing a prebuilt combine_docs_chain
# (avoids "multiple values for keyword" error).
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",  # internally creates the combine-docs chain
    combine_docs_chain_kwargs={"prompt": prompt},  # inject our prompt
    return_source_documents=True,
    verbose=False,
)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = (request.form.get("msg") or "").strip()
    if not msg:
        return "Please enter a question."

    print("User:", msg)
    # ConversationalRetrievalChain expects "question"
    result = rag_chain.invoke({"question": msg})
    answer = result.get("answer", "")
    print("Response:", answer)
    return str(answer)


@app.route("/history", methods=["GET"])
def history():
    """Return current chat history (user + assistant turns)."""
    hist = memory.load_memory_variables({})
    chat = hist.get("chat_history", [])
    return jsonify([
        {"role": getattr(m, "type", "message"), "content": getattr(m, "content", "")}
        for m in chat
    ])


@app.route("/reset", methods=["POST"])
def reset():
    """Clear conversation memory."""
    memory.clear()
    return jsonify({"ok": True, "message": "Conversation history cleared."})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
