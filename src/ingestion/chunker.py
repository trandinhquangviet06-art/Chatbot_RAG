import json
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
def create_chunks(input_json: str, output_json:str, chunk_size: int =1000, chunk_overlap: int =200):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap,
        length_function= len,
        separators= ["\n\n", "\n", " ", ""]
    )
    with open(input_json, "r", encoding="utf-8") as f:
        documents= json.load(f)
    chunked_data=[]
    for doc in documents:
        text=doc["text"]
        metadata= doc["metadata"]
        chunks= text_splitter.split_text(text)
        for i,chunk_text in enumerate(chunks):
            chunk_metadata=metadata.copy()
            chunk_metadata["chunk_id"]= f"chunk_{uuid.uuid4().hex[:8]}"
            chunk_metadata["chunk_index"]= i+1
            chunked_data.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    print("dang ghi du lieu ra file ......")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    input_json="data/processed/data_test.jsonl"
    output_json="data/processed/chunked_data.jsonl"
    try:
        create_chunks(input_json, output_json)
    except Exception as e:
        print(F"loi {e}")

    