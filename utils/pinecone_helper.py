import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid

load_dotenv()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)


def split_text(text, chunk_size=300):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_index_for_resume(first_name):
    import re

    safe_first_name = re.sub(r"[^a-z0-9-]", "", first_name.lower())
    index_name = f"{safe_first_name}-resume"
    if index_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=index_name,
                dimension=384,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
        except Exception as e:
            if "ALREADY_EXISTS" not in str(e):
                raise
    return pc.Index(index_name)


def upsert_resume(resume_text, first_name):
    index = get_index_for_resume(first_name)
    chunks = split_text(resume_text)
    vectors = [
        (str(uuid.uuid4()), embedder.encode(chunk).tolist(), {"text": chunk})
        for chunk in chunks
    ]
    index.upsert(vectors)


def query_resume_context(prompt, first_name, top_k=5):
    index = get_index_for_resume(first_name)
    query_vector = embedder.encode(prompt).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in results["matches"]])


def delete_resume_index(first_name):
    import re
    safe_first_name = re.sub(r"[^a-z0-9-]", "", first_name.lower())
    index_name = f"{safe_first_name}-resume"
    pc.delete_index(index_name)
