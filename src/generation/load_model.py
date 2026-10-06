import json
import re

# llama-cpp-python chi dung duoc o local (can C++ compiler de build).
# Tren Streamlit Cloud se fallback sang regex-based parser.
try:
    from langchain_community.llms import llamacpp
    _LLAMA_AVAILABLE = True
except Exception:
    _LLAMA_AVAILABLE = False


def load_qwen_local(path: str):
    """Load Qwen model local. Tra ve None neu khong co llama-cpp-python."""
    if not _LLAMA_AVAILABLE:
        print("[load_model] llama-cpp-python khong co san, bo qua Qwen model.")
        return None
    try:
        return llamacpp.LlamaCpp(
            model_path=path,
            n_ctx=2048,
            n_gpu_layers=-1,
            temperature=0.0,
            max_tokens=100,
            verbose=False,
        )
    except Exception as e:
        print(f"[load_model] Khong the load Qwen model: {e}")
        return None


def extract_filter(query: str, llm) -> dict:
    """
    Trich xuat {company, year} tu query.
    - Neu llm is None (cloud mode): dung regex don gian.
    - Neu co llm (local mode): goi LLM nhu cu.
    """
    # --- Fallback: regex-based (cloud / no LLM) ---
    if llm is None:
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = year_match.group(1) if year_match else ""
        known = ["ADOBE", "APPLE", "3M", "GENERALMILLS", "AMAZON",
                 "GOOGLE", "MICROSOFT", "META", "NETFLIX", "TESLA"]
        company = ""
        q_upper = query.upper()
        for k in known:
            if k in q_upper:
                company = k
                break
        return {"company": company, "year": year}

    # --- Full LLM path (local) ---
    prompt = f"""You are a strict data extraction parser. Extract the target company and year from the user query.

RULES:
1. Output ONLY a raw JSON object.
2. "company": UPPERCASE company name. If none, output "".
3. "year": Extract ANY year (4-digit number like 2022, 2023) from the query, even inside phrases like "Q2 of 2023", "Q1 2022", "fiscal year 2023", "as of 2023". Output ONLY the 4-digit year number. If none, output "".

EXAMPLES:
Query: "Does Adobe have improving Free cashflow in 2022?"
JSON: {{"company": "ADOBE", "year": "2022"}}

Query: "Does Adobe have an improving Free cashflow conversion as of FY2022?"
JSON: {{"company": "ADOBE", "year": "2022"}}

Query: "How many securities are registered under 3M's name as of Q2 of 2023?"
JSON: {{"company": "3M", "year": "2023"}}

Query: "What was Apple's revenue in Q1 2021?"
JSON: {{"company": "APPLE", "year": "2021"}}

Query: "{query}"
JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response if isinstance(response, str) else response[0]["generated_text"]
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            print(f"filter: {json_str}")
            result = json.loads(json_str)
            # Fallback: nếu LLM vẫn không extract được year, dùng regex trên query gốc
            if not result.get("year"):
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    result["year"] = year_match.group(1)
                    print(f"[fallback] year từ regex: {result['year']}")
            return result
        return {"company": "", "year": ""}
    except Exception as e:
        print(f"Loi extract_filter(): {e}")
        return {"company": "", "year": ""}