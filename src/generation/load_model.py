from langchain_community.llms import llamacpp
def load_qwen_local(path: str):
    llama_model= llamacpp.LlamaCpp(
        model_path= path,
        n_ctx= 2048,
        n_gpu_layers= -1,
        temperature=0.0,
        max_tokens= 100,
        verbose= False
    )
    return llama_model
import json
import re

def extract_filter(query: str, llm) -> dict:
    prompt = f"""You are a highly strict data extraction parser. Your ONLY job is to extract the target company and year from the user's query into a valid JSON object.

CRITICAL RULES:
1. Output ONLY a raw JSON object. NO markdown formatting (like ```json), NO conversational text, NO explanations.
2. "company": Extract the core company name and convert to UPPERCASE (e.g., "ADOBE", "3M", "GENERALMILLS"). If no company is mentioned, output "".
3. "year": Extract as a 4-digit string (e.g., "2022"). If no year is mentioned, output "".

EXAMPLES:
Query: "Does Adobe have an improving Free cashflow conversion in 2022?"
JSON: {{"company": "ADOBE", "year": "2022"}}

Query: "Explain how to calculate gross margin."
JSON: {{"company": "", "year": ""}}

Query: "What were 3M's capital expenditures last year?"
JSON: {{"company": "3M", "year": ""}}

Query: "{query}"
JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response if isinstance(response, str) else response[0]["generated_text"]
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return {"company": "", "year": ""}
            
    except Exception as e:
        print(f"Lỗi bóc tách JSON trong extract_filter(): {e}")
        return {"company": "", "year": ""}