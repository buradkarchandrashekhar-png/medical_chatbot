from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_splitting,download_model
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medical-chatbot"


extracted_documents = load_pdf_files('data')
minimal_documents = filter_to_minimal_docs(extracted_documents)

texts_chunk=text_splitting(minimal_documents)
print(f"Number of chunks: {len(texts_chunk)}")

embeddings = download_model()



# check if index exists
existing = [idx.name for idx in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(
        name=index_name,
        dimension=384,          # e.g., all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch=PineconeVectorStore.from_documents(documents=texts_chunk,
 embedding=embeddings, 
 index_name=index_name)


