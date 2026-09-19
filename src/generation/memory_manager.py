import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
class Memory:
    def __init__(self, max_short_term=5, persist_directory="data/vector_store/memory_db"):
        self.max_short_term= max_short_term
        # Xác định đường dẫn file JSON lưu short_term và summary
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.memory_file = os.path.join(_project_root, "data", "vector_store", "memory_db", "short_summary.json")
        # Load short_term và summary từ file nếu đã tồn tại
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.short_term = data.get("short_term", [])
            self.summary = data.get("summary", "Chưa có thông tin được lưu")
            print("Da load short-term memory va summary tu file")
        else:
            self.short_term = []
            self.summary = "Chưa có thông tin được lưu"
# chi tao rieng vector db cho thang long-term thoi
        self.embedding_model_lt= HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.vector_store_lt= Chroma(
            collection_name ="user_long_term_memory",
            embedding_function= self.embedding_model_lt,
            persist_directory= persist_directory
        )
        self.base_retriever= self.vector_store_lt.as_retriever(search_kwargs={"k":15})
        rerank_model= HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        self.base_compressor= CrossEncoderReranker(model=rerank_model, top_n=3)
        self.compressor_retriever= ContextualCompressionRetriever(
            base_compressor= self.base_compressor,
            base_retriever= self.base_retriever
        )
        self.llm= ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0
        )
    def add_conversation(self, query, ai_answer):
# luu hoi thoai moi, neu thang short day thi pop ra push vao thang long
        conversation =f"user: {query}\n AI_respone: {ai_answer}"
        self.short_term.append(conversation)
        if len(self.short_term) > self.max_short_term:
            oldest_conver=self.short_term.pop(0)
            doc = Document(page_content=oldest_conver, metadata={"type":"memory"})
            self.vector_store_lt.add_documents([doc])
            self._update_summary(oldest_conver)
        self._save_to_file()
    def _save_to_file(self):
        """Lưu short_term và summary xuống file JSON."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump({"short_term": self.short_term, "summary": self.summary}, f, ensure_ascii=False, indent=2)

    def _update_summary(self, conversation):
        prompt= f"""Hãy cập nhật bản tóm tắt cuộc trò chuyện.
        TÓM TẮT HIỆN TẠI: {self.summary}
        HỘI THOẠI VỪA TRÔI QUA: {conversation}
        
        Viết lại một bản tóm tắt mới, ngắn gọn, giữ lại các ý chính mà người dùng đã hỏi và quan tâm:"""
        try:
            respone = self.llm.invoke(prompt)
            content = respone.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            self.summary = content.strip()
            print("Da tom tat hoi thoai moi")
            self._save_to_file()
        except Exception as e:
            print(f"loi he thong: {e}")
    def context(self, current_query):
        #lay context 3 lop
        try:
            long_term_results= self.compressor_retriever.invoke(current_query)
            long_term_txt= "\n".join([doc.page_content for doc in long_term_results]) if long_term_results else "khong tim thay doan hoi thoai nao lien quan "
        except Exception as e:
            print("loi lan dau vi ban dau trong db k co du lieu nao")
            long_term_txt= "khong tim thay doan hoi thoai nao lien quan "

        short_term_txt= "\n\n".join(self.short_term) if self.short_term else "khong tim thay doan hoi thoai nao lien quan"
        return long_term_txt, short_term_txt, self.summary




