from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
def search_database(query: str, persist_directory: str="data/vector_store/finance_db", k: int=3):
    embedding_model =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store= Chroma(
        persist_directory= persist_directory,
        embedding_function= embedding_model
    )
    results= vector_store.similarity_search(query, k=k)
    if not results:
        return "khong tim thay noi dung tra loi cho cau hoi"
    for i,doc in enumerate(results):
        print("top noi dung lien quan: ")
        print(f"source:{doc.metadata.get('source')}")
        print(f"page_number: {doc.metadata.get('page_number')}")
        print(f"content: {doc.page_content}")
if __name__=="__main__":
    query="What was the total amount of research, development and related expenses for 3M in 2015? Please answer in Vietnamese"
    try:
        search_database(query)
    except Exception as e:
        print(f"loi: {e}")


