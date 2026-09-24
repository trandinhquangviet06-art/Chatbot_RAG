import json
import re
import os
import sqlite3
import threading
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.stores import ByteStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


# ======================================================================
# SQLiteByteStore — luu toan bo parent docs vao 1 file .db duy nhat
# Thay the LocalFileStore (tao hang nghin file nho -> lag khi upload Kaggle)
# ======================================================================
class SQLiteByteStore(ByteStore):
    """LangChain ByteStore backed by a single SQLite file."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv_store "
                "(key TEXT PRIMARY KEY, value BLOB NOT NULL)"
            )
            conn.commit()
            conn.close()

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        if not keys:
            return []
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            placeholders = ",".join("?" * len(keys))
            rows = conn.execute(
                f"SELECT key, value FROM kv_store WHERE key IN ({placeholders})",
                list(keys),
            ).fetchall()
            conn.close()
        mapping = {r[0]: r[1] for r in rows}
        return [mapping.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        if not key_value_pairs:
            return
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.executemany(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                key_value_pairs,
            )
            conn.commit()
            conn.close()

    def mdelete(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            placeholders = ",".join("?" * len(keys))
            conn.execute(
                f"DELETE FROM kv_store WHERE key IN ({placeholders})", list(keys)
            )
            conn.commit()
            conn.close()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            if prefix:
                rows = conn.execute(
                    "SELECT key FROM kv_store WHERE key LIKE ?", (prefix + "%",)
                ).fetchall()
            else:
                rows = conn.execute("SELECT key FROM kv_store").fetchall()
            conn.close()
        for row in rows:
            yield row[0]

    def __len__(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            count = conn.execute("SELECT COUNT(*) FROM kv_store").fetchone()[0]
            conn.close()
        return count


def create_chunks(input_json: str, vector_store_path: str, parent_db_path: str):
    """
    Tao vector store (FAISS) va parent docstore (SQLite).

    Args:
        input_json:        Duong dan toi file JSON chua du lieu dau vao.
        vector_store_path: Thu muc luu FAISS index (.faiss + .pkl).
        parent_db_path:    Duong dan file SQLite luu parent documents, vi du
                           'parent_docstore.db'. Chi la 1 FILE duy nhat —
                           de upload/download hon nhieu so voi thu muc.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunked_data = []
    for doc in documents:
        text = doc["Text"]
        metadata = doc["Metadata"]
        parts = metadata.get("source", "").split("_")
        metadata["company"] = parts[0].upper()
        year_doc = re.sub(r"\D", "", parts[1])
        metadata["year"] = year_doc
        doc_text = (
            f"source: {metadata.get('source', 'unk')}, "
            f"company: {metadata.get('company', 'unk')}, "
            f"year: {metadata.get('year', 'unk')}\n\n"
            f"document: {text}"
        )
        chunked_data.append(Document(page_content=doc_text, metadata=metadata))

    parent_splitter = MarkdownTextSplitter(chunk_size=5000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"batch_size": 32}
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vector_store = FAISS.from_texts(["init"], embedding=embedding)

    # ------------------------------------------------------------------ #
    # Dung SQLiteByteStore thay cho LocalFileStore:                        #
    #   - Luu TOAN BO parent documents vao 1 file .db duy nhat            #
    #   - Khong can tao thu muc, khong co hang nghin file nho              #
    #   - De upload len Kaggle Dataset hon nhieu                           #
    # ------------------------------------------------------------------ #
    byte_store = SQLiteByteStore(parent_db_path)
    print(f"Parent docstore: {parent_db_path} (SQLite)")

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        byte_store=byte_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    step = 500  # nho hon de thay tien do va tranh OOM
    total = len(chunked_data)
    for i in range(0, total, step):
        chunk_doc = chunked_data[i : i + step]
        retriever.add_documents(chunk_doc)
        done = i + len(chunk_doc)
        print(f"da xong chunk {done}/{total} ({done / total * 100:.1f}%)")

    # Xoa document "init" duoc them luc khoi tao FAISS
    vector_store.delete([vector_store.index_to_docstore_id[0]])
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    print(f"hoan tat! FAISS luu tai: {vector_store_path}")
    print(f"Parent DB luu tai:       {parent_db_path}")
    print(f"Tong parent docs da luu: {len(byte_store)}")


if __name__ == "__main__":
    input_json = "/kaggle/input/datasets/quangvietdz/extracted-data/extracted_data.jsonl"
    vector_store_path = "finance_db"

    # Chi la 1 FILE .db thay vi 1 thu muc voi hang nghin file nho
    parent_db_path = "parent_docstore.db"

    try:
        create_chunks(input_json, vector_store_path, parent_db_path)
    except Exception as e:
        print(f"loi: {e}")
        raise
