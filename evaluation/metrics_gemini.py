"""
Đo 3 metrics chính của RAG pipeline — dùng Gemini làm judge (thay cho Qwen GGUF local):

  1. Answer Accuracy     — Câu trả lời RAG có đúng với ground truth không?
  2. Faithfulness        — Câu trả lời có bịa thêm thông tin KHÔNG có trong context không?
  3. Context Precision   — Trong các chunk được retrieve, bao nhiêu chunk thực sự liên quan?

Ưu điểm so với metrics.py (Qwen 3B local):
  - Chạy được trên Colab / Kaggle (không cần GPU mạnh)
  - Gemini hiểu ngữ nghĩa tài chính tốt hơn nhiều (số tương đương, đơn vị...)
  - Không cần tải model GGUF ~2GB
"""

import json
import os
import re
import sys
import time

# ── Thêm project root vào path ────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from src.generation.generator import answer_query_eval

# ── Khởi tạo Gemini Judge ──────────────────────────────────────────────────────
# Judge model mạnh hơn để đánh giá tài chính chính xác hơn
JUDGE_MODEL = "gemini-2.0-flash-lite"

_judge_llm = ChatGoogleGenerativeAI(
    model=JUDGE_MODEL,
    temperature=0,
    max_retries=5,
)


def _judge(system: str, user: str) -> str:
    """Gọi Gemini với system + user prompt, trả về text."""
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    resp = _judge_llm.invoke(messages)
    raw = resp.content
    if isinstance(raw, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        )
    return str(raw).strip()


# ── Metric 1: Answer Accuracy ──────────────────────────────────────────────────
def score_accuracy(question: str, rag_answer: str, truth_answer: str) -> dict:
    """
    So sánh rag_answer với truth_answer bằng Gemini.
    Gemini hiểu được các trường hợp tương đương:
      - Đơn vị: "$1,577M" = "1.577 tỷ USD" = "$1577.00"
      - Làm tròn: "$8.738B" ~ "$8.70B"
      - Ngôn ngữ: trả lời tiếng Việt so với ground truth tiếng Anh

    Trả về {"label": "ĐÚNG|SAI", "score": 1.0|0.0, "reason": str}
    """
    system = """Bạn là chuyên gia đánh giá câu trả lời RAG tài chính.

NHIỆM VỤ: So sánh "Câu trả lời RAG" với "Câu trả lời chuẩn".

QUY TẮC QUAN TRỌNG:
1. Nếu ý nghĩa/kết luận TƯƠNG ĐƯƠNG -> ĐÚNG, dù cách diễn đạt khác nhau.
2. Về SỐ LIỆU: coi là đúng nếu sai số làm tròn <= 1% (ví dụ: $8.738B ~ $8.70B).
3. Về ĐƠN VỊ: "$1,577 triệu" = "$1.577 tỷ" = "$1577.00M" -> ĐÚNG.
4. Về NGÔN NGỮ: câu trả lời tiếng Việt so với ground truth tiếng Anh vẫn được chấp nhận.
5. Nếu RAG trả lời ĐẦY ĐỦ HƠN ground truth (thêm context hữu ích) -> vẫn ĐÚNG.
6. Chỉ SAI nếu: số liệu sai, kết luận ngược chiều, hoặc thiếu thông tin CỐT LÕI.

FORMAT TRẢ LỜI (bắt buộc):
Dòng 1: ĐÚNG hoặc SAI (chỉ 1 từ)
Dòng 2: Lý do ngắn gọn (1-2 câu)"""

    user = (
        f"CÂU HỎI: {question}\n\n"
        f"CÂU TRẢ LỜI RAG:\n{rag_answer[:800]}\n\n"
        f"CÂU TRẢ LỜI CHUẨN:\n{truth_answer[:400]}"
    )

    verdict = _judge(system, user)
    # Tìm ĐÚNG/SAI ở dòng đầu tiên
    first_line = verdict.split("\n")[0].strip().upper()
    if "ĐÚNG" in first_line or "DUNG" in first_line or "CORRECT" in first_line or "TRUE" in first_line:
        label = "ĐÚNG"
    elif "SAI" in first_line or "WRONG" in first_line or "FALSE" in first_line or "INCORRECT" in first_line:
        label = "SAI"
    else:
        # fallback: scan toàn bộ text
        m = re.search(r"\b(ĐÚNG|SAI)\b", verdict, re.IGNORECASE)
        label = m.group(1).upper() if m else "KHÔNG RÕ"

    return {
        "label": label,
        "score": 1.0 if label == "ĐÚNG" else 0.0,
        "reason": verdict,
    }


# ── Metric 2: Faithfulness ─────────────────────────────────────────────────────
def score_faithfulness(question: str, rag_answer: str, context_str: str) -> dict:
    """
    Đo tỷ lệ claim trong RAG answer được hỗ trợ bởi context.
    Score = supported_claims / total_claims
    """
    system = """Bạn là chuyên gia kiểm tra tính trung thực của câu trả lời RAG.

NHIỆM VỤ:
1. Liệt kê tất cả CLAIM (mệnh đề có thể kiểm chứng) trong câu trả lời RAG.
2. Với mỗi claim, kiểm tra xem nó có được hỗ trợ bởi TÀI LIỆU CONTEXT không.
   - "Hỗ trợ" = thông tin trong context xác nhận claim này (dù diễn đạt khác nhau).
   - Claim suy luận logic từ context cũng được tính là "hỗ trợ".
   - Chỉ đánh dấu "không hỗ trợ" nếu claim KHÔNG thể suy ra từ context.

FORMAT TRẢ LỜI (bắt buộc, không thêm gì khác):
TOTAL_CLAIMS: <số nguyên>
SUPPORTED_CLAIMS: <số nguyên>
REASON: <1 câu tóm tắt>"""

    user = (
        f"CÂU HỎI: {question}\n\n"
        f"CÂU TRẢ LỜI RAG:\n{rag_answer[:900]}\n\n"
        f"TÀI LIỆU CONTEXT:\n{context_str[:1500]}"
    )

    verdict = _judge(system, user)

    total_m   = re.search(r"TOTAL_CLAIMS\s*:\s*(\d+)",     verdict, re.IGNORECASE)
    supp_m    = re.search(r"SUPPORTED_CLAIMS\s*:\s*(\d+)", verdict, re.IGNORECASE)
    reason_m  = re.search(r"REASON\s*:\s*(.+)",            verdict, re.IGNORECASE)

    total     = int(total_m.group(1))       if total_m   else 1
    supported = int(supp_m.group(1))        if supp_m    else 0
    reason    = reason_m.group(1).strip()   if reason_m  else verdict
    score     = round(supported / max(total, 1), 2)

    return {
        "score":     score,
        "supported": supported,
        "total":     total,
        "reason":    reason,
    }


# ── Metric 3: Context Precision ───────────────────────────────────────────────
def score_context_precision(question: str, context_docs: list) -> dict:
    """
    Với mỗi chunk retrieve được, hỏi Gemini: chunk này có liên quan đến câu hỏi không?
    Score = relevant_chunks / total_chunks
    """
    system = """Bạn đánh giá xem một đoạn văn bản có LIÊN QUAN đến câu hỏi không.

QUY TẮC:
- "Liên quan" = đoạn văn CÓ CHỨA thông tin giúp trả lời câu hỏi (dù gián tiếp).
- Trả lời: CÓ hoặc KHÔNG, sau đó 1 câu lý do ngắn."""

    total    = len(context_docs)
    relevant = 0
    details  = []

    for i, doc in enumerate(context_docs):
        chunk_text = (
            doc.page_content[:600]
            if hasattr(doc, "page_content")
            else str(doc)[:600]
        )
        source = doc.metadata.get("source", "?") if hasattr(doc, "metadata") else "?"
        page   = doc.metadata.get("page_number", "?") if hasattr(doc, "metadata") else "?"

        user = (
            f"Câu hỏi: {question}\n\n"
            f"Đoạn văn (từ {source}, trang {page}):\n{chunk_text}"
        )
        verdict = _judge(system, user)
        is_relevant = bool(
            re.search(r"\bCÓ\b",  verdict, re.IGNORECASE) or
            re.search(r"\bYES\b", verdict, re.IGNORECASE)
        )
        if is_relevant:
            relevant += 1
        details.append({
            "chunk":    i + 1,
            "source":   source,
            "page":     page,
            "relevant": is_relevant,
            "verdict":  verdict[:120],
        })
        time.sleep(0.5)   # tránh rate-limit khi judge nhiều chunk

    score = round(relevant / max(total, 1), 2)
    return {"score": score, "relevant": relevant, "total": total, "details": details}


# Pipeline tổng hợp
def run_evaluation(
    testset_path: str  = "evaluation/financebench_open_source.jsonl",
    output_path:  str  = "evaluation/eval_results_gemini.json",
    max_questions: int = None,          # None = chạy hết, đặt số để test nhanh
    rate_limit_delay: float = 1.5,      # giây giữa các câu (tránh Gemini 429)
):
    with open(testset_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    if max_questions:
        lines = lines[:max_questions]

    total = len(lines)
    print(f"\n{'='*60}")
    print(f"  Đánh giá {total} câu hỏi  |  Judge: Gemini ({JUDGE_MODEL})")
    print(f"{'='*60}\n")

    all_results = []
    sum_acc, sum_faith, sum_ctx = 0.0, 0.0, 0.0

    for i, data in enumerate(lines):
        question = data["question"]
        truth_an = data["answer"]

        print(f"[{i+1}/{total}] {question[:70]}...")

        # ── Gọi RAG pipeline ──
        rag_answer, context_docs, context_str = answer_query_eval(question)

        # ── 3 metrics ──
        acc   = score_accuracy(question, rag_answer, truth_an)
        time.sleep(rate_limit_delay)

        faith = score_faithfulness(question, rag_answer, context_str)
        time.sleep(rate_limit_delay)

        ctx   = score_context_precision(question, context_docs)
        time.sleep(rate_limit_delay)

        sum_acc   += acc["score"]
        sum_faith += faith["score"]
        sum_ctx   += ctx["score"]

        icon = "✅" if acc["score"] == 1.0 else "❌"
        print(
            f"  Accuracy={icon}  "
            f"Faithfulness={faith['score']:.2f}  "
            f"CtxPrecision={ctx['score']:.2f}"
        )

        all_results.append({
            "question":          question,
            "truth_answer":      truth_an,
            "rag_answer":        rag_answer,
            "accuracy":          acc,
            "faithfulness":      faith,
            "context_precision": ctx,
        })

    n = max(total, 1)
    avg_acc   = round(sum_acc   / n, 3)
    avg_faith = round(sum_faith / n, 3)
    avg_ctx   = round(sum_ctx   / n, 3)

    summary = {
        "total_questions":       total,
        "judge_model":           JUDGE_MODEL,
        "avg_accuracy":          avg_acc,
        "avg_faithfulness":      avg_faith,
        "avg_context_precision": avg_ctx,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ ({total} câu hỏi) — Judge: {JUDGE_MODEL}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Score'}")
    print(f"  {'-'*40}")
    print(f"  {'Answer Accuracy':<25} {avg_acc:.1%}  ({int(sum_acc)}/{total} đúng)")
    print(f"  {'Faithfulness':<25} {avg_faith:.1%}  (avg claims supported)")
    print(f"  {'Context Precision':<25} {avg_ctx:.1%}  (avg chunk relevant)")
    print(f"{'='*60}")
    print(f"\n  📄 Chi tiết đã lưu -> {output_path}\n")

    return summary


if __name__ == "__main__":
    # Chạy nhanh 3 câu để test: max_questions=3
    # Chạy hết tất cả: bỏ max_questions
    run_evaluation()
