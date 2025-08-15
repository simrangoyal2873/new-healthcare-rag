import streamlit as st
from process import extract_text_chunks
from vector_db_local import add_to_db, search_db
import pandas as pd
import io
import os
import uuid
from rag_pipeline import RAGSystem





def format_healthcare_answer(query: str, docs: list[str], metas: list[dict]) -> str:
    """Format results for healthcare domain."""
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






# --- Streamlit Config ---
st.set_page_config(page_title="Healthcare Visual RAG", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Visual Document RAG")


# --- File Upload ---
uploaded = st.file_uploader(
    "Upload a healthcare document (PDF, PNG, JPG)", 
    type=["pdf", "png", "jpg", "jpeg"],
    key="doc_uploader"
)

all_chunks = []

if uploaded:
    file_ext = uploaded.name.split('.')[-1].lower()
    temp_file = f"temp_file.{file_ext}"
    with open(temp_file, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Extracting text, tables, charts, images, and layout details..."):
        all_chunks = extract_text_chunks(temp_file)

    add_to_db(all_chunks)
    st.success(f"Indexed {len(all_chunks)} chunks successfully.")

    # --- Separate chunks by type ---
    tables = [c for c in all_chunks if c.get("meta", {}).get("type") == "table"]
    charts = [c for c in all_chunks if c.get("meta", {}).get("type") == "chart"]
    chart_insights = [c for c in all_chunks if c.get("meta", {}).get("type") == "chart_semantic"]
    images = [c for c in all_chunks if c.get("meta", {}).get("type") == "image_correlation"]
    layout = [c for c in all_chunks if c.get("meta", {}).get("type") == "layout_analysis"]
    mixed = [c for c in all_chunks if c.get("meta", {}).get("type") == "mixed_content"]

    # --- Display Tables ---
    if tables:
        with st.expander("📊 Extracted Tables"):
            for i, t in enumerate(tables, start=1):
                st.markdown(f"**Table {i} — Page {t['meta'].get('page','?')}**")
                try:
                    df = pd.read_csv(io.StringIO(t["text"]), sep="\t")
                    st.dataframe(df)
                except:
                    try:
                        st.dataframe(pd.read_csv(io.StringIO(t["text"].replace("|", ",")), skiprows=1))
                    except:
                        st.code(t["text"])

    # --- Display Charts ---
    if charts:
        with st.expander("📈 Extracted Charts"):
            for c in charts:
                meta = c.get("meta", {})
                if meta.get("image_path") and os.path.exists(meta["image_path"]):
                    st.image(meta["image_path"], caption=f"Page {meta.get('page')} — {meta.get('chart_type', 'unknown')}")
                elif meta.get("image_content"):
                    st.image(meta["image_content"])
                if meta.get("data"):
                    st.write("Parsed chart data:")
                    st.dataframe(pd.DataFrame(meta["data"]))
                if c.get("text"):
                    st.write("Chart OCR/Description:")
                    st.write(c["text"])

    # --- Chart Insights ---
    if chart_insights:
        with st.expander("🧠 Chart Semantic Insights"):
            for c in chart_insights:
                st.markdown(f"**Page {c['meta'].get('page','?')}**")
                st.write(c["text"])

    # --- Images & Correlation ---
    if images:
        with st.expander("🖼 Images & Text Correlation"):
            for c in images:
                meta = c.get("meta", {})
                if meta.get("image_path") and os.path.exists(meta["image_path"]):
                    st.image(meta["image_path"], caption=meta.get("context", ""))
                else:
                    st.write(meta.get("context", "(No image path)"))
                st.write(c.get("text", ""))

    # --- Layout Analysis ---
    if layout:
        with st.expander("📐 Layout Analysis"):
            for c in layout:
                st.write(f"Page {c['meta'].get('page','?')}:")
                st.code(c["text"])

    # --- Mixed Content ---
    if mixed:
        with st.expander("📝 Mixed Content Analysis"):
            for c in mixed:
                st.write(f"Page {c['meta'].get('page','?')}:")
                st.code(c["text"])


# --- Query Interface ---
st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    q = st.text_input("Ask a question about the document:", "")
with col2:
    scope = st.selectbox("Search scope", ["all", "text", "table", "chart", "chart_semantic", "image_correlation"], index=0)

if q:
   
    filt = None if scope == "all" else {"type": {"$eq": scope}}

    results = search_db(q, top_k=6, meta_filter=filt)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    st.subheader("Answer")
    answer = format_healthcare_answer(q, docs, metas)
    st.write(answer)

    st.subheader("Top Matches")
    for i, (d, m) in enumerate(zip(docs, metas), start=1):
        tag = m.get("type", "text")
        page = m.get("page", "?")
        st.markdown(f"**{i}. [{tag}] Page {page}**")
        if tag == "table":
            try:
                df = pd.read_csv(io.StringIO(d), sep="\t")
                st.dataframe(df)
            except:
                st.code(d)
        elif tag in ["chart", "chart_semantic"]:
            if m.get("image_path") and os.path.exists(m["image_path"]):
                st.image(m["image_path"], caption=f"Page {page} — {m.get('chart_type','unknown')}")
            elif m.get("image_content"):
                st.image(m["image_content"])
            if m.get("data"):
                st.write("Parsed chart data:")
                st.dataframe(pd.DataFrame(m["data"]))
            st.code(d)
        elif tag == "image_correlation":
            if m.get("image_path") and os.path.exists(m["image_path"]):
                st.image(m["image_path"], caption=f"Page {page}")
            st.write(f"**Context:** {m.get('context','')}")
        else:
            st.write(d)
