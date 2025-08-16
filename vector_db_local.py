# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules['pysqlite3']

# import chromadb
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Any
# import uuid
# import json


# # --- Initialize ---
# client = chromadb.PersistentClient(path="./healthcare_db")
# COLLECTION_NAME = "medical_docs_local"

# # Load the embedding model 
# _model = SentenceTransformer("all-MiniLM-L6-v2")

# def _embed_function(texts: List[str]) -> List[List[float]]:
#     """A helper function to generate embeddings."""
#     return _model.encode(texts, normalize_embeddings=True).tolist()

# try:
#     _col = client.get_collection(COLLECTION_NAME)
# except:
#     _col = client.create_collection(
#         name=COLLECTION_NAME,
#         embedding_function=_embed_function
#     )

# # --- Add pre-split chunks ---
# def add_to_db(chunks: List[Dict[str, Any]]):
#     """Adds pre-split text chunks to the ChromaDB collection."""
#     if not chunks:
#         return
    
#     ids = [str(uuid.uuid4()) for _ in chunks]
#     docs = [c["text"] for c in chunks]

#     metas = []
#     for c in chunks:
#         clean_meta = {}
#         for k, v in c.get("meta", {}).items():
#             if isinstance(v, (str, int, float, bool)) or v is None:
#                 clean_meta[k] = v
#             else:
#                 clean_meta[k] = json.dumps(v)
#         metas.append(clean_meta)

#     _col.add(ids=ids, documents=docs, metadatas=metas)

# # --- Search the database ---
# def search_db(query: str, top_k: int = 5, meta_filter: Dict[str, Any] | None = None):
#     """
#     Returns top_k most relevant chunks for a given query.
    
#     The where parameter is now correctly passed as None when there is no filter.
#     """
#     return _col.query(
#         query_texts=[query],
#         n_results=top_k,
#         where=meta_filter
#     )



import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import uuid
import json
import streamlit as st

COLLECTION_NAME = "medical_docs_local"

@st.cache_resource
def get_collection():
    # Use /tmp because Streamlit Cloud only allows writes there
    client = chromadb.PersistentClient(path="/tmp/healthcare_db")

    # Use built-in SentenceTransformerEmbeddingFunction (stable)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        col = client.get_collection(COLLECTION_NAME)
    except:
        col = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
    return col

# --- Add pre-split chunks ---
def add_to_db(chunks: List[Dict[str, Any]]):
    if not chunks:
        return
    
    col = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    docs = [c["text"] for c in chunks]

    metas = []
    for c in chunks:
        clean_meta = {}
        for k, v in c.get("meta", {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            else:
                clean_meta[k] = json.dumps(v)
        metas.append(clean_meta)

    col.add(ids=ids, documents=docs, metadatas=metas)

# --- Search the database ---
def search_db(query: str, top_k: int = 5, meta_filter: Dict[str, Any] | None = None):
    col = get_collection()
    return col.query(
        query_texts=[query],
        n_results=top_k,
        where=meta_filter
    )
