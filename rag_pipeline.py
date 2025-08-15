import os
from vector_db_local import add_to_db, search_db
from process import extract_text_chunks

class RAGSystem:
    def __init__(self):
        # Placeholder for future LLM initialization if needed
        pass

    def process_document(self, file_path: str):
        """
        Processes a document, extracts chunks, and adds them to the database.
        """
        chunks = extract_text_chunks(file_path)
        add_to_db(chunks)
        return chunks

    def query(self, query_text: str, scope: str = "all", top_k: int = 6):
        """
        Searches the database and formats the results.
        """
        filt = None if scope == "all" else {"type": scope}
        results = search_db(query_text, top_k=top_k, meta_filter=filt)
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        answer = self.format_healthcare_answer(query_text, docs, metas)
        
        return answer, list(zip(docs, metas))

    def format_healthcare_answer(self, query: str, docs: list[str], metas: list[dict]) -> str:
        """
        Formats retrieved content into a coherent answer.
        This is a simple rule-based example.
        """
        table_texts = [(d, m) for d, m in zip(docs, metas) if m.get("type") == "table"]
        if table_texts:
            out = ["**Lab Results Summary**"]
            for d, m in table_texts:
                out.append(d[:500] + ("..." if len(d) > 500 else ""))
            return "\n".join(out)

        chart_texts = [(d, m) for d, m in zip(docs, metas) if m.get("type") in ["chart", "chart_semantic"]]
        if chart_texts:
            out = ["**Chart Summary**"]
            for d, m in chart_texts[:3]:
                out.append(f"- Page {m.get('page','?')}: " + d[:300] + ("..." if len(d) > 300 else ""))
            return "\n".join(out)

        text_snips = [d for d, m in zip(docs, metas) if m.get("type") == "text"]
        if text_snips:
            return "\n".join(text_snips[:2])
            
        return "No relevant content found."