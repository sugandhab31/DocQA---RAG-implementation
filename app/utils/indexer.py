from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbeddings
import chromadb
import os

def index_text_data(doc_id: str, content: str):
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents([
        {
            "text": content
        }
    ])

    embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    db_dir = f"./vectordb/{doc_id}"
    os.makedirs(db_dir, exist_ok = True)

    chroma_client = chromadb.PersistentClient(path=db_dir)
    chroma_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("docs"))

    storage_context = StorageContext.from_defaults(vector_store=chroma_store)

    index = VectorStoreIndex(nodes,storage_context=storage_context, service_context=ServiceContext.from_defaults(embed_model=embed_model))

    return index