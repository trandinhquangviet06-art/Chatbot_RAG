import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("khong thay API KEY")

# Resolve persist_directory relative to project root regardless of cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_db_path(persist_directory: str) -> str:
    """Return an absolute path: if already absolute, use as-is; otherwise join with project root."""
    if os.path.isabs(persist_directory):
        return persist_directory
    return os.path.join(_PROJECT_ROOT, persist_directory)


def answer_query(query: str, persist_directory: str = "data/vector_store/finance_db"):
    abs_db_path = _resolve_db_path(persist_directory)

    # Retrieval
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=abs_db_path,
    )
    results = vector_store.similarity_search(query, k=3)

    if not results:
        yield "Không tìm thấy tài liệu liên quan đến câu hỏi này trong cơ sở dữ liệu."
        return

    context = "\n\n".join([
        f"source: {doc.metadata.get('source')} - page: {doc.metadata.get('page_number')}:\n{doc.page_content}"
        for doc in results
    ])

    llm= ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
   

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""Bạn là một chuyên gia phân tích tài chính cấp cao. Hãy sử dụng CÁC TÀI LIỆU THAM KHẢO dưới đây để trả lời câu hỏi.
        Yêu cầu bắt buộc:
        - Chỉ lấy thông tin từ TÀI LIỆU THAM KHẢO được cung cấp, không tự suy diễn.
        - Luôn TRÍCH DẪN rõ tên file và số trang khi đưa ra số liệu (Ví dụ: Theo báo cáo abc.pdf, trang X...).
        - Trả lời bằng Tiếng Việt một cách chuyên nghiệp.

        TÀI LIỆU THAM KHẢO:
        {context}

        CÂU HỎI CỦA NGƯỜI DÙNG: {query}""",
    )

    final_prompt = prompt_template.format(context=context, query=query)

    try:
        for chunk in llm.stream(final_prompt):
            content = chunk.content
            # Các phiên bản LangChain mới có thể trả về list thay vì str
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            yield content
    except Exception as e:
        yield f"\n\n⚠️ Lỗi khi gọi mô hình: {e}"


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What was the total amount of research, development and related expenses for 3M in 2015? Please answer in Vietnamese"
    )
    for token in answer_query(query):
        print(token, end="", flush=True)
    print()
