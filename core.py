import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ====================== CONFIG ======================
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "db/chroma"
DATA_DIR = "data"

# Ensure directories exist
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ====================== LLM & EMBEDDING ======================
llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

def get_index():
    """Persistent index loader: Loads from disk if exists, otherwise creates new."""
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection("knowledge_garden")
    
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if the collection already has documents
    if collection.count() > 0:
        # Load the existing index from the vector store
        return VectorStoreIndex.from_vector_store(
            vector_store, 
            embed_model=embed_model
        )
    else:
        # Initial setup: Index files in 'data' folder if any exist
        if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                embed_model=embed_model
            )
            return index
        else:
            # Return an empty index if no data exists yet
            return VectorStoreIndex.from_documents(
                [], 
                storage_context=storage_context, 
                embed_model=embed_model
            )

def ingest_and_index(file_bytes: bytes, filename: str, index):
    """Saves file to disk AND adds it to the live index instantly."""
    # 1. Save to disk for 'Explore' tab
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    # 2. Extract text and add to Vector DB
    # We use SimpleDirectoryReader on the single new file
    new_doc = SimpleDirectoryReader(input_files=[file_path]).load_data()
    for doc in new_doc:
        index.insert(doc)
    
    return True

def generate_reflection(index):
    """Generate insight using qwen3:8b"""
    query_engine = index.as_query_engine(llm=llm)
    prompt = "Analyze the entire knowledge garden and give one deep, actionable insight or hidden connection. Be concise but valuable."
    try:
        response = query_engine.query(prompt)
        return str(response)
    except Exception as e:
        return f"Error generating reflection: {str(e)}"