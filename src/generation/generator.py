import os
import sys
import functools
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json
#re_rank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from memory_manager import Memory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from load_model import load_qwen_local, extract_filter
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path=os.path.join(_PROJECT_ROOT,".env")
load_dotenv(dotenv_path=env_path)
api_key= os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("khong tim thay api key trong file .env")
@functools.lru_cache(maxsize=1)
def get_resource(persist_directory: str):
    """chi loa cac model llm, embedding hay rerank 1 lan thoi chu lan nao chayj cx load lai -> lau+ ton ram"""
    print("load cac model......")
    qwen_model= load_qwen_local("models/qwen2.5_3B/qwen2.5-coder-3b-instruct-q5_k_m.gguf")
    embedding_model= HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    abs_db_path= _resolve_db_path(persist_directory)
    vector_store= FAISS.load_local(
        folder_path= abs_db_path,
        embeddings= embedding_model,
        allow_dangerous_deserialization= True

    )
    vector_search= vector_store.as_retriever(search_kwargs={"k": 40})
    chunkeddata_path= os.path.join(_PROJECT_ROOT, "data/processed/chunked_data.jsonl")
    bm25_path= os.path.join(persist_directory, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_retriever= pickle.load(f)
    else:
        with open(chunkeddata_path, "r", encoding="utf-8") as f:
            documents= json.load(f)
        docs= []
        for doc in documents:
            tai_lieu= Document(
                page_content= doc["Text"],
                metadata= doc["Metadata"]
            )
            docs.append(tai_lieu)
        bm25_retriever= BM25Retriever.from_documents(docs)
        bm25_retriever.k= 40
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    ensemble_retriever= EnsembleRetriever(retrievers=[vector_search, bm25_retriever], weights=[0.5, 0.5])

    print("rerank.....")
    cross_encoder_model= HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    # top_n=8 để giữ nhiều chunk hơn, tăng cơ hội có đủ số liệu cho tính toán
    base_compressor= CrossEncoderReranker(model= cross_encoder_model, top_n=8)
    llm= ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    return qwen_model, base_compressor, llm, ensemble_retriever
                                



def _resolve_db_path(persist_directory: str) -> str:
    """Return an absolute path: if already absolute, use as-is; otherwise join with project root."""
    if os.path.isabs(persist_directory):
        return persist_directory
    return os.path.join(_PROJECT_ROOT, persist_directory)

agent_memory= Memory()
def rewrite_query(query: str, llm)-> str:
    rewrite_prompt="""You are a senior US GAAP accountant.
    Extract the core search keywords from the user's question to search a financial database.
    CRITICAL RULES:
    1. Identify the Company Name and Year, and put them at the VERY BEGINNING of the keywords.
    2. If the user asks for a calculated metric (e.g., "Free Cash Flow Conversion", "Gross Margin", "YoY Growth"), DO NOT output the name of the metric. Instead, output the RAW accounting line items needed to calculate it (e.g., "Net Income, Net cash provided by operating activities, Capital expenditures, Purchases of property and equipment").
    3. Return ONLY a comma-separated list of keywords. Do not explain.
    Original question: {query}
    Financial keywords:"""
    response= llm.invoke(rewrite_prompt.format(query= query))
    keyword= response.content
    return f"{query}. Finance keywords:{keyword}" if keyword else query


def _build_prompt_and_retrieve(query: str, persist_directory: str):
    """Hàm nội bộ: retrieve docs + build prompt, dùng chung cho cả streaming và eval."""
    long_term_txt, short_term_txt, summary = agent_memory.context(query)
    print("Dang tim cau tra loi...")
    qwen_model, base_compressor, llm, ensemble_retriever = get_resource(persist_directory)
    metadata_filter= extract_filter(query, qwen_model)
    target_company= metadata_filter.get("company", "").upper()
    target_year= metadata_filter.get("year", "")
    print(f"Metadata filter: company={target_company}, year={target_year}")
    new_query= rewrite_query(query, llm)
    raw_doc = ensemble_retriever.invoke(new_query)
    filtered_docs = []
    for doc in raw_doc:
        source_name= str(doc.metadata.get("source", ""))
        parts= source_name.split("_")
        if len(parts)>=2:
            name_company= parts[0].upper()
            doc_year= parts[1]
        else:
            name_company= source_name.upper()
            doc_year= ""
        is_valid= True
        if target_company and name_company != target_company:
            is_valid= False
        if target_year and doc_year != target_year:
            is_valid= False
        if is_valid:
            filtered_docs.append(doc)
    print(f"đã lọc thành công giữ lại {len(filtered_docs)}/{len(raw_doc)}")
    if filtered_docs:
        results= base_compressor.compress_documents(filtered_docs, new_query)   
    context_str = "\n\n".join([
        f"source: {doc.metadata.get('source')} - page: {doc.metadata.get('page_number')}:\n{doc.page_content}"
        for doc in results
    ])
    print(f"context: {context_str[:100]}")
    context_docs = results

    prompt_template = PromptTemplate(
        input_variables=["summary", "short_term_txt", "long_term_txt", "context", "query"],
        template="""Bạn là một chuyên gia phân tích tài chính cấp cao. Hãy trả lời câu hỏi của người dùng dựa trên các dữ liệu sau:

        [TÓM TẮT LỊCH SỬ CHAT (Summary Memory)]:
        {summary}

        [KÝ ỨC DÀI HẠN CÓ LIÊN QUAN (Long-term Memory)]:
        {long_term_txt}

        [CÁC LƯỢT CHAT GẦN NHẤT (Short-term Memory)]:
        {short_term_txt}

        [TÀI LIỆU RAG TRÍCH XUẤT TỪ BÁO CÁO TÀI CHÍNH]:
        {context}

        QUY TẮC BẮT BUỘC:
        1. Ưu tiên tuyệt đối dùng [TÀI LIỆU RAG] để trả lời — KHÔNG suy đoán nếu không có trong context.
        2. Dùng [Memory] để hiểu ngữ cảnh hội thoại nếu câu hỏi là follow-up.
        3. Luôn TRÍCH DẪN rõ tên file và số trang (Ví dụ: Theo 3M_2018_10K.pdf, trang 60...).
        4. Khi câu hỏi yêu cầu TÍNH TOÁN (ratio, %, thay đổi YoY...), hãy:
           a. Xác định rõ từng số liệu cần thiết từ context
           b. Kiểm tra đơn vị (Millions/Billions) — ĐỪNG nhầm đơn vị
           c. Thực hiện tính toán từng bước (step-by-step)
           d. Trình bày công thức và kết quả
        5. Số liệu tài chính (figures): viết bằng tiếng Anh theo chuẩn quốc tế ($X.XX million/billion).
           Giải thích và phân tích: viết bằng tiếng Việt chuyên nghiệp.
        6. Nếu không tìm thấy thông tin trong context, hãy nói thẳng thay vì bịa.

        CÂU HỎI HIỆN TẠI: {query}"""
    )
    final_prompt = prompt_template.format(
        summary=summary,
        long_term_txt=long_term_txt,
        short_term_txt=short_term_txt,
        context=context_str,
        query=query
    )
    return results, context_str, context_docs, llm, final_prompt


def answer_query_eval(query: str, persist_directory: str = "data/vector_store/finance_db") -> tuple:
    """
    Dùng cho evaluation: trả về (rag_answer, context_docs, context_str) — KHÔNG phải generator.
    Memory bị TẮT hoàn toàn trong eval mode để tránh nhiễu giữa các câu hỏi độc lập.
    """
    print("Dang tim cau tra loi...")
    qwen_model, base_compressor, llm, ensemble_retriever = get_resource(persist_directory)
    new_query= rewrite_query(query, llm)
    metadata_filter= extract_filter(query, qwen_model)
    target_company= metadata_filter.get("company", "").upper()
    target_year= metadata_filter.get("year", "")
    raw_doc= ensemble_retriever.invoke(new_query)
    filtered_docs=[]
    for doc in raw_doc:
        sour= doc.metadata.get("source", "")
        parts= sour.split("_")
        if len(parts)>=2:
            doc_company= parts[0].upper()
            doc_year= parts[1]
        else:
            doc_company= sour
            doc_year= ""
        is_val= True
        if target_company and doc_company!= target_company:
            is_val=False
        if target_year and doc_year!= target_year:
            is_val= False
        if is_val:
            filtered_docs.append(doc)
    print(f"da loc va giu lai {len(filtered_docs)}/{len(raw_doc)}")
    if filtered_docs:
        results= base_compressor.compress_documents(filtered_docs, new_query)
    if not results:
        return "Không tìm thấy tài liệu liên quan đến câu hỏi này trong cơ sở dữ liệu.", [], ""

    context_str = "\n\n".join([
        f"source: {doc.metadata.get('source')} - page: {doc.metadata.get('page_number')}:\n{doc.page_content}"
        for doc in results
    ])
    print("context: {context_str}")

    # Prompt đơn giản hơn cho eval — không có memory noise
    eval_prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""You are a senior financial analyst. Answer the question based ONLY on the provided documents.

DOCUMENTS:
{context}

RULES:
1. Use ONLY information from the documents above — do NOT guess or fabricate.
2. Always cite the source file and page number (e.g., "According to 3M_2018_10K.pdf, page 60...").
3. For calculations (ratios, %, YoY change):
   a. Identify each required figure from the context
   b. Verify units carefully (Millions vs Billions)
   c. Show step-by-step calculation
   d. State the final answer clearly
4. Report financial figures in standard international format ($X.XX million or $X.XX billion).
5. If the answer cannot be found in the context, state that explicitly.

QUESTION: {query}

ANSWER:"""
    )
    final_prompt = eval_prompt_template.format(context=context_str, query=query)

    response = llm.invoke(final_prompt)
    raw = response.content
    if isinstance(raw, list):
        full_answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw
        )
    else:
        full_answer = str(raw)

    # context_docs là alias của results để nhất quán với caller
    context_docs = results
    # KHÔNG gọi agent_memory.add_conversation() trong eval mode
    return full_answer, context_docs, context_str


def answer_query(query: str, persist_directory: str = "data/vector_store/finance_db", is_eval: bool = False):
    """Generator streaming — dùng cho UI chat."""
    results, context_str, context_docs, llm, final_prompt = _build_prompt_and_retrieve(query, persist_directory)
    if not results:
        yield "Không tìm thấy tài liệu liên quan đến câu hỏi này trong cơ sở dữ liệu."
        return

    full_answer = ""
    try:
        for chunk in llm.stream(final_prompt):
            content = chunk.content
            
            # Làm sạch từng mảnh chunk nếu nó là dạng list
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
                
            full_answer += content
            yield content  # Nhả từng chữ ra màn hình
            
        agent_memory.add_conversation(query, full_answer)
        
    except Exception as e:
        yield f"\n\n⚠️ Lỗi khi gọi mô hình: {e}"


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Does Adobe have an improving Free cashflow conversion as of FY2022? ?answer in VietNamese."
    )
    for token in answer_query(query):
        print(token, end="", flush=True)
    print()
