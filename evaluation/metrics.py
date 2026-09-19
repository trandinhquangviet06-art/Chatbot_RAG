"""
Đo 3 metrics chính của RAG pipeline bằng LLM-as-a-judge (Llama):

  1. Answer Accuracy     — Câu trả lời RAG có đúng với ground truth không?
  2. Faithfulness        — Câu trả lời có bịa thêm thông tin KHÔNG có trong context không?
  3. Context Precision   — Trong các chunk được retrieve, bao nhiêu chunk thực sự liên quan?

Kết quả ghi ra file JSON + in bảng tóm tắt ra terminal.
"""

import json
import os
import re
import sys

from llama_cpp import Llama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.generation.generator import answer_query

JUDGE_MODEL_PATH = "models/qwen2.5_3B/qwen2.5-coder-3b-instruct-q5_k_m.gguf"

judge = Llama(
    model_path=JUDGE_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)

STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]


def _chat_prompt(system: str, user: str) -> str:
    """Tạo prompt theo chat format của Qwen2.5."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _judge(system: str, user: str, max_tokens: int = 120) -> str:
    out = judge(
        prompt=_chat_prompt(system, user),
        max_tokens=max_tokens,
        echo=False,
        stop=STOP_TOKENS,
    )
    return out["choices"][0]["text"].strip()


def _extract_score(text: str, pattern=r"(\d+(?:\.\d+)?)") -> float:
    """Lấy số thực đầu tiên tìm thấy trong text."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0.0


def _find_label(text: str) -> str:
    m = re.search(r"\b(ĐÚNG|SAI)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else "KHÔNG RÕ"

def score_accuracy(question: str, rag_answer: str, truth_answer: str) -> dict:
    """
    So sánh rag_answer với truth_answer.
    Trả về {"label": "ĐÚNG|SAI", "score": 1.0|0.0, "reason": str}
    """
    system = (
        "Bạn là chuyên gia đánh giá câu trả lời. "
        "So sánh câu trả lời RAG với câu trả lời chuẩn. "
        "Dòng đầu tiên PHẢI là đúng 1 từ: ĐÚNG hoặc SAI. "
        "Sau đó giải thích ngắn trong 1 câu."
    )
    user = (
        f"Câu hỏi: {question}\n\n"
        f"Câu trả lời RAG: {rag_answer[:600]}\n\n"
        f"Câu trả lời chuẩn: {truth_answer[:300]}"
    )
    verdict = _judge(system, user)
    label = _find_label(verdict)
    return {"label": label, "score": 1.0 if label == "ĐÚNG" else 0.0, "reason": verdict}


#  Metric 2: Faithfulness 
def score_faithfulness(question: str, rag_answer: str, context_str: str) -> dict:
    """
    Đo mức độ câu trả lời được hỗ trợ bởi context (không bịa thêm).
    Trả về {"score": 0.0-1.0, "reason": str}

    Cách tính:
      - Yêu cầu judge đếm: tổng số claim trong answer, bao nhiêu claim có trong context
      - Score = supported / total
    """
    system = (
        "Bạn là chuyên gia kiểm tra tính trung thực của câu trả lời RAG. "
        "Nhiệm vụ: đếm số CLAIM (mệnh đề/thông tin cụ thể) trong câu trả lời, "
        "rồi kiểm tra từng claim có được hỗ trợ bởi TÀI LIỆU CONTEXT không. "
        "Trả lời theo format sau (không viết gì khác):\n"
        "TOTAL_CLAIMS: <số>\n"
        "SUPPORTED_CLAIMS: <số>\n"
        "REASON: <1 câu giải thích>"
    )
    user = (
        f"CÂU HỎI: {question}\n\n"
        f"CÂU TRẢ LỜI RAG:\n{rag_answer[:700]}\n\n"
        f"TÀI LIỆU CONTEXT:\n{context_str[:1200]}"
    )
    verdict = _judge(system, user, max_tokens=200)

    # Parse TOTAL và SUPPORTED
    total_m = re.search(r"TOTAL_CLAIMS\s*:\s*(\d+)", verdict, re.IGNORECASE)
    supp_m = re.search(r"SUPPORTED_CLAIMS\s*:\s*(\d+)", verdict, re.IGNORECASE)
    reason_m = re.search(r"REASON\s*:\s*(.+)", verdict, re.IGNORECASE)

    total = int(total_m.group(1)) if total_m else 1
    supported = int(supp_m.group(1)) if supp_m else 0
    reason = reason_m.group(1).strip() if reason_m else verdict
    score = round(supported / max(total, 1), 2)

    return {"score": score, "supported": supported, "total": total, "reason": reason}


# ── Metric 3: Context Precision ────────────────────────────────────────────────
def score_context_precision(question: str, context_docs: list) -> dict:
    """
    Với mỗi chunk được retrieve, hỏi judge chunk đó có liên quan đến câu hỏi không.
    Score = số chunk relevant / tổng số chunk.
    Trả về {"score": 0.0-1.0, "relevant": int, "total": int, "details": list}
    """
    system = (
        "Bạn đánh giá xem một đoạn văn bản có LIÊN QUAN đến câu hỏi không. "
        "Trả lời chỉ bằng: CÓ hoặc KHÔNG, sau đó 1 câu lý do ngắn."
    )
    total = len(context_docs)
    relevant = 0
    details = []

    for i, doc in enumerate(context_docs):
        chunk_text = doc.page_content[:500] if hasattr(doc, "page_content") else str(doc)[:500]
        source = doc.metadata.get("source", "?") if hasattr(doc, "metadata") else "?"
        page = doc.metadata.get("page_number", "?") if hasattr(doc, "metadata") else "?"

        user = (
            f"Câu hỏi: {question}\n\n"
            f"Đoạn văn (từ {source}, trang {page}):\n{chunk_text}"
        )
        verdict = _judge(system, user, max_tokens=80)
        is_relevant = bool(re.search(r"\bCÓ\b", verdict, re.IGNORECASE))
        if is_relevant:
            relevant += 1
        details.append({
            "chunk": i + 1,
            "source": source,
            "page": page,
            "relevant": is_relevant,
            "verdict": verdict[:100],
        })

    score = round(relevant / max(total, 1), 2)
    return {"score": score, "relevant": relevant, "total": total, "details": details}


# ── Pipeline tổng hợp ───────────────────────────────────────────────────────────
def run_evaluation(testset_path: str = "evaluation/qa_dataset.jsonl",
                   output_path: str = "evaluation/eval_results.json"):
    with open(testset_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    total = len(lines)
    print(f"\n{'='*60}")
    print(f"  Bắt đầu đánh giá {total} câu hỏi với 3 metrics")
    print(f"{'='*60}\n")

    all_results = []
    sum_acc, sum_faith, sum_ctx = 0.0, 0.0, 0.0

    for i, data in enumerate(lines):
        question  = data["question"]
        truth_an  = data["answer"]

        print(f"[{i+1}/{total}] {question[:70]}...")

        # Gọi RAG pipeline — nhận về (answer, context_str, context_docs)
        rag_answer, context_str, context_docs = next(answer_query(question, is_eval=True))

        # ── 3 metrics ──
        acc   = score_accuracy(question, rag_answer, truth_an)
        faith = score_faithfulness(question, rag_answer, context_str)
        ctx   = score_context_precision(question, context_docs)

        sum_acc   += acc["score"]
        sum_faith += faith["score"]
        sum_ctx   += ctx["score"]

        icon_acc   = "✅" if acc["score"] == 1.0 else "❌"
        print(f"  Accuracy={icon_acc}  Faithfulness={faith['score']:.2f}  CtxPrecision={ctx['score']:.2f}")

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
        "total_questions":          total,
        "avg_accuracy":             avg_acc,
        "avg_faithfulness":         avg_faith,
        "avg_context_precision":    avg_ctx,
    }

    # Ghi kết quả ra file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_results}, f, ensure_ascii=False, indent=2)

    # In bảng tổng kết
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ ({total} câu hỏi)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Score'}")
    print(f"  {'-'*40}")
    print(f"  {'Answer Accuracy':<25} {avg_acc:.1%}  ({int(sum_acc)}/{total} đúng)")
    print(f"  {'Faithfulness':<25} {avg_faith:.1%}  (avg claims supported)")
    print(f"  {'Context Precision':<25} {avg_ctx:.1%}  (avg chunk relevant)")
    print(f"{'='*60}")
    print(f"\n  📄 Chi tiết đã lưu → {output_path}\n")

    return summary


if __name__ == "__main__":
    run_evaluation()