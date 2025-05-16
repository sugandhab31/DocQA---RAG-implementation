from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import HuggingFaceLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

documents = SimpleDirectoryReader("data").load_data()

llm = HuggingFaceLLM(
    model_name = "google/flan-t5-base",
    tokenizer_name = "google/flan-t5-base",
    context_window = 1024,
    max_new_token = 256,
    generate_kwargs = {"temprature": 0.1, "do_sample": False},
    device_map = "auto"
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("Summerise the document in 3 bullet points.")

print(response)