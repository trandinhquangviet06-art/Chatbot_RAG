import os
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
def build_vector_db(input_json: str, persist_directory: str):
    embeddings_model= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs=[]
    with open(input_json, "r", encoding="utf-8") as f:
        chunked_data= json.load(f)
 # chuyen sang dang Document chuan langchain       
    for item in chunked_data:
        doc= Document(
            page_content=item["text"],
            metadata=item["metadata"]
            )
        docs.append(doc)
    print("start Embedding.....")
#tao dirt neu chx co
    os.makedirs(persist_directory, exist_ok=True)
    vector_store= Chroma.from_documents(
        documents= docs,
        embedding= embeddings_model,
        persist_directory= persist_directory
    )
    print("successful")
if __name__=="__main__":
    input_json="data/processed/chunked_data.jsonl"
    persist_directory="data/vector_store/finance_db"
    try:
        build_vector_db(input_json, persist_directory)
    except Exception as e:
        print(f"loi: {e}")

            
