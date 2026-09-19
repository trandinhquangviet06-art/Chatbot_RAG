#Chay tren kaggle kernel
import pymupdf
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber, tabulate
import pandas as pd
def pdf_extraction(path: str)-> List[Dict[str,Any]]:
    path= Path(path)
    first_skip=3
    last_skip=5
    docs=[]
    file_name= path.name
    doc_pymupdf= pymupdf.open(path)
    doc_pdfplumber= pdfplumber.open(path)
    total_page= len(doc_pymupdf)
    last_page=total_page-last_skip
    for i in range(first_skip, last_page):
        page_pymupdf= doc_pymupdf[i]
        page_pdfplumber= doc_pdfplumber.pages[i]
        tables= page_pdfplumber.find_tables()
        page_elements=[]
        for table in tables:
            data_table= table.extract()
            df= pd.DataFrame(data_table[1:], columns= data_table[0])
            df=df.replace("\n", " ", regex=True)
            content_table=df.to_markdown(index=False)
            bbox= table.bbox #(vien trai, top, vien phai, bottom)
            page_elements.append({
                "top_block": bbox[1],
                "content":content_table
            })
            rect= pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3]) #chuyen toa do sang pymupdf
            page_pymupdf.add_redact_annot(rect) #add toa do
        page_pymupdf.apply_redactions() # xoa text da trich tu khu vuc nay di
        blocks= page_pymupdf.get_text("blocks") #lay block thi no tra ve nhu bbox nhung co them 3 thuoc tinh nx(text, block_num, block_type)
        for b in blocks:
            block_type= b[6]
            if block_type==0:  #0: text, 1:image
                text= b[4].strip()
                if text:
                    page_elements.append({
                        "top_block": b[1],
                        "content": text
                    })
        page_elements.sort(key= lambda x: x["top_block"])
        final_content= "\n\n".join(element["content"] for element in page_elements if element["content"])
        doc_id= "doc_"+str(uuid.uuid4().hex[:8])
        metadata={
            "source": file_name,
            "page_number": i+1,
            "id":doc_id,
            "total_page": total_page
        }
        docs.append({
            "Text": final_content,
            "Metadata": metadata
        })
    doc_pymupdf.close()
    doc_pdfplumber.close()
    return docs
import pprint
pdf_path= list(Path("/kaggle/input/datasets/quangvietdz/train-data").glob("*.pdf"))
print(len(pdf_path))
pprint.pprint(pdf_path[:5])
output_path= "extracted_data.json"
failed_file=[]
extracted_data=[]
for path in pdf_path:
    try:
        print(f"Dang xu li file {path.name}")
        docs= pdf_extraction(path)
        extracted_data.extend(docs)
        print(f"Do dai page: {len(docs)}")
    except Exception as e:
        failed_file.append({
            "file": path.name,
            "error": f"{type(e).__name__}: {e}"
        })
        print(f"{type(e).__name__}:{e}")
print(f"tong so file loi: {len(failed_file)}")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    