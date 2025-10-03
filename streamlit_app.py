import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# ---------- ENV ----------
load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./data/vectorstore")) / "faiss_index"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
TOP_K = int(os.getenv("TOP_K", 6))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", 20))


# ---------- UTILS ----------
def ensure_dir(p: Path):
    """Make sure 'p' exists as a directory; delete if a file blocks it."""
    if p.exists() and not p.is_dir():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


ensure_dir(UPLOAD_DIR)
ensure_dir(INDEX_DIR.parent)


@st.cache_resource(show_spinner=False)
def get_llm():
    """Lazy-load and cache the Groq LLM client once per container."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    return ChatGroq(model=GROQ_MODEL, temperature=0)


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    """Cache embeddings so we don't re-download on each run."""
    return HuggingFaceEmbeddings(model_name=model_name)


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(docs)


def load_docs(paths: List[Path]):
    docs = []
    for p in paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def load_or_create_index(chunks):
    embeddings = get_embeddings(EMBEDDING_MODEL)
    ensure_dir(INDEX_DIR.parent)

    if INDEX_DIR.exists() and INDEX_DIR.is_dir():
        vs = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        if chunks:
            new_vs = FAISS.from_documents(chunks, embeddings)
            vs.merge_from(new_vs)
            vs.save_local(str(INDEX_DIR))
        return vs
    else:
        if INDEX_DIR.exists() and not INDEX_DIR.is_dir():
            INDEX_DIR.unlink()
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(str(INDEX_DIR))
        return vs


def format_sources(docs) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = os.path.basename(d.metadata.get("source", "Unknown.pdf"))
        page = d.metadata.get("page")
        page_info = f" p.{page+1}" if isinstance(page, int) else ""
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(f"[S{i}] {src}{page_info}: {snippet}")
    return "\n".join(lines)


SYSTEM = SystemMessage(content=(
    "You are a precise assistant for Q&A over PDF documents. "
    "Use ONLY the provided context to answer. If the answer is not in the context, say you don't know. "
    "Cite sources inline like [S1], [S2]. Keep answers concise."
))
QA_PROMPT = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):")
])


# ---------- UI ----------
st.set_page_config(page_title="PDF Chatbot (Groq)", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot — LangChain + Groq (Free Tier)")
st.caption("Upload PDFs → build FAISS index → ask questions with citations. Uses Groq-hosted Llama; no local GPU needed.")

with st.sidebar:
    st.header("Settings")
    st.write("**Model:**", GROQ_MODEL)
    retrieve_k = st.slider("Retrieve-k", 5, 50, RETRIEVE_K)
    top_k = st.slider("Top-k", 2, 12, TOP_K)

    if st.button("Reset Index"):
        import shutil
        if INDEX_DIR.exists():
            if INDEX_DIR.is_dir():
                shutil.rmtree(INDEX_DIR)
            else:
                INDEX_DIR.unlink()
        st.success("Index reset. Upload again to rebuild.")

    # Simple health check for Groq creds/model
    if st.button("Test Groq connection"):
        try:
            test = get_llm().invoke(
                ChatPromptTemplate.from_messages([("human", "Say 'ok'.")]).format_messages()
            )
            st.success(f"Groq OK: {getattr(test, 'content', str(test))}")
        except Exception as e:
            st.error("Groq test failed.")
            st.exception(e)

uploads = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# ---------- STATE ----------
if "vs" not in st.session_state:
    st.session_state.vs = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- BUILD / MERGE INDEX ----------
if uploads:
    paths = []
    with st.spinner("Saving files..."):
        for f in uploads:
            dst = UPLOAD_DIR / f.name
            ensure_dir(UPLOAD_DIR)
            with open(dst, "wb") as out:
                out.write(f.getbuffer())
            paths.append(dst)
    with st.spinner("Reading & chunking PDFs..."):
        chunks = split_docs(load_docs(paths))
    with st.spinner("Building/merging FAISS index..."):
        st.session_state.vs = load_or_create_index(chunks)
        st.session_state.retriever = st.session_state.vs.as_retriever(
            search_kwargs={"k": retrieve_k}
        )
    st.success("Index ready! Ask a question below.")

# ---------- CHAT ----------
user_q = st.chat_input("Ask a question about your PDFs…")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Chat")

    # show history
    for u, a in st.session_state.history:
        with st.chat_message("user"):
            st.write(u)
        with st.chat_message("assistant"):
            st.write(a)

    if user_q:
        if st.session_state.retriever is None:
            st.warning("Upload PDFs first.")
        else:
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking…"):
                        docs = st.session_state.retriever.get_relevant_documents(user_q)

                        # Build context blocks with labeled sources [S1], [S2], ...
                        ctx_blocks = []
                        for i, d in enumerate(docs[:top_k], start=1):
                            src = os.path.basename(d.metadata.get("source", "Unknown.pdf"))
                            page = d.metadata.get("page")
                            page_info = f" (page {page+1})" if isinstance(page, int) else ""
                            ctx_blocks.append(f"[S{i}] from {src}{page_info}:\n{d.page_content}")
                        context_text = "\n\n".join(ctx_blocks)

                        # LLM call
                        messages = QA_PROMPT.format_messages(context=context_text, question=user_q)
                        llm = get_llm()
                        resp = llm.invoke(messages)
                        answer = getattr(resp, "content", str(resp))

                        st.write(answer)
                        with st.expander("Sources"):
                            st.code(format_sources(docs[:top_k]))

                        st.session_state.history.append((user_q, answer))

                except Exception as e:
                    # Show the most useful error details (helps with 400s)
                    import traceback, json
                    st.error("LLM request failed.")
                    err_txt = "".join(
                        traceback.format_exception_only(type(e), e)
                    ).strip()
                    st.code(err_txt)

                    # Try surfacing HTTP response body if present
                    try:
                        details = getattr(e, "response", None)
                        if details is not None:
                            st.write("**HTTP status:**", getattr(details, "status_code", ""))
                            try:
                                st.code(json.dumps(details.json(), indent=2))
                            except Exception:
                                st.code(getattr(details, "text", ""))
                    except Exception:
                        pass

with col2:
    st.subheader("Index Info")
    if st.session_state.vs is None:
        st.info("No index yet.")
    else:
        st.write("**Vector store:** FAISS")
        st.write("**Embeddings:**", EMBEDDING_MODEL)
        st.write("**History:**", len(st.session_state.history))
        st.write("**Retrieve-k:**", retrieve_k)
        st.write("**Top-k:**", top_k)

# ---------- ENV GUARD ----------
if not os.getenv("GROQ_API_KEY"):
    st.sidebar.error("Missing GROQ_API_KEY in Environment settings on Render.")
