import json
import fitz
import uuid 
from typing import List, Dict, Any
from pathlib import Path
def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    path= Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file {path} khong ton tai")
    with fitz.open(path) as doc:
     file_name=path.name
     doc_id="doc_"+ str(uuid.uuid4().hex[:8])
     documents=[]
     total_pages=len(doc)
     for page_num in range(total_pages):
        page=doc[page_num]
        text=page.get_text("text")
        cleaned_text=text.strip()
        if not cleaned_text:
            continue
        metadata={
            "source": file_name,
            "document_id": doc_id,
            "page_number": page_num +1,
            "total_pages": total_pages
        }
        documents.append({
                "text": cleaned_text,
                "metadata": metadata
            })
     
    return documents
if __name__=="__main__":
   input_path=["data/raw_data/3M_2015_10K.pdf", "data/raw_data/3M_2016_10K.pdf", "data/raw_data/3M_2017_10K.pdf"]
   output_path="data/processed/data_test.jsonl"
   try:
      for i in range(len(input_path)):
         extracted_data=load_pdf(input_path[i])
         with open(output_path,"w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
            print("thanh cong")
   except Exception as e:
      print(f"loi: {e}")        






