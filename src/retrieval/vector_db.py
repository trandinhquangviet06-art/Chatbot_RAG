#FILE NAY CHAY TREN KAGGLE
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

persist_directory = "finance_db"
input_path = "/kaggle/input/datasets/quangvietdz/chunked-data/chunked_data.jsonl"

print("Đang tải mô hình Embedding...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"batch_size": 32}
)

docs = []
print("Đang đọc dữ liệu từ file...")

with open(input_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

for doc in documents:
    tai_lieu = Document(
        page_content=doc["Text"],
        metadata=doc["Metadata"]
    )
    docs.append(tai_lieu)

print(f"=> Hoàn tất! Tổng số chunks cần nhúng: {len(docs)}")

print("\nBẮT ĐẦU QUÁ TRÌNH EMBEDDING:")
os.makedirs(persist_directory, exist_ok=True)

step = 5000
vector_store = None

for i in range(0, len(docs), step):
    batch = docs[i : i + step]
    
    if i == 0:
        vector_store = FAISS.from_documents(
            documents=batch,
            embedding=embedding_model
        )
    else:
        vector_store.add_documents(batch)
        
    print(f"  -> Đã lưu xong mẻ từ chunk {i} đến {i + len(batch)}")

vector_store.save_local(persist_directory)
print(f"\n🎉 HOÀN TẤT! Đã lưu thành công toàn bộ Vector DB tại thư mục: {persist_directory}")