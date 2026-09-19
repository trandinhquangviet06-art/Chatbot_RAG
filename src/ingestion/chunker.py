import json
import uuid
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter
def create_chunks(input_json: str, output_json:str, chunk_size: int =2000, chunk_overlap: int =400):
    text_splitter= MarkdownTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap
    )
    with open(input_json, "r", encoding="utf-8") as f:
        documents= json.load(f)
    chunked_data=[]
    for doc in documents:
        text=doc["Text"]
        metadata= doc["Metadata"]
        chunks= text_splitter.split_text(text)
        for i,chunk_text in enumerate(chunks):
            chunk_metadata=metadata.copy()
            chunk_metadata["chunk_id"]= f"chunk_{uuid.uuid4().hex[:8]}"
            chunk_metadata["chunk_index"]= i+1
            source_doc= chunk_metadata.get("source", "unknown")
            page_doc= chunk_metadata.get("page_number", "unknown")
            text_add= f"source document: {source_doc}\npage number: {page_doc}\n\n{chunk_text}"
            chunked_data.append({
                "Text": text_add,
                "Metadata": chunk_metadata
            })
    print("dang ghi du lieu ra file ......")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    input_json="data/processed/extracted_data.jsonl"
    output_json="data/processed/chunked_data.jsonl"
    try:
        create_chunks(input_json, output_json)
    except Exception as e:
        print(F"loi {e}")

    