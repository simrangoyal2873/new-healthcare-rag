import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("medical_docs")

def add_to_db(chunks):
    for i, chunk in enumerate(chunks):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk["text"]
        )
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[emb.data[0].embedding],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"]}]
        )

def search_db(query, top_k=3):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    results = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=top_k
    )
    return results
