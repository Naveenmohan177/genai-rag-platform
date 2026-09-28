import os
import io
import re
import datetime
import sqlite3
import streamlit as st
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI as DirectOpenAI

st.set_page_config(page_title="Cognitive Intelligence Platform", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stAppViewContainer { background-color: #0F172A; color: #F8FAFC; }
    .stSidebar { background-color: #1E293B; border-right: 1px solid #334155; }
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
    .stButton>button { background-color: #2563EB; color: #FFFFFF; border: none; border-radius: 6px; padding: 0.5rem 1rem; width: 100%; }
    .stButton>button:hover { background-color: #1D4ED8; }
    .metric-card { background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
    .metric-value { font-size: 1.875rem; font-weight: 700; color: #38BDF8; }
    .metric-label { font-size: 0.875rem; color: #94A3B8; text-transform: uppercase; }
    .context-box { background-color: #1E293B; border-left: 3px solid #38BDF8; padding: 1rem; border-radius: 4px; margin-bottom: 0.75rem; color: #CBD5E1; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

def init_db():
    conn = sqlite3.connect("rag_telemetry.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS query_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, query TEXT, response TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_query(session_id: str, query: str, response: str):
    conn = sqlite3.connect("rag_telemetry.db")
    c = conn.cursor()
    c.execute("INSERT INTO query_logs (session_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, query, response, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect("rag_telemetry.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM query_logs")
    count = c.fetchone()[0]
    conn.close()
    return count

init_db()

class SecurityGuardrails:
    @staticmethod
    def sanitize_input(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
        return text

    @staticmethod
    def check_prompt_injection(text: str) -> bool:
        suspicious_patterns = ["ignore previous instructions", "system prompt override", "unrestricted ai"]
        return any(pattern in text.lower() for pattern in suspicious_patterns)

class CloudRAGEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "input_audio.wav"
        transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text

    def index_documents(self, file_paths: List[str]) -> int:
        documents = []
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = text_splitter.split_documents(documents)
        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 4
        return len(chunks)

    def process_query(self, user_query: str) -> Dict[str, Any]:
        if SecurityGuardrails.check_prompt_injection(user_query):
            return {"answer": "Security Policy Violation: Prompt injection blocked.", "sources": []}

        sanitized_query = SecurityGuardrails.sanitize_input(user_query)
        sources = []
        context_str = ""

        if self.faiss_db and self.bm25_retriever:
            dense_results = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized_query, k=4)]
            sparse_results = self.bm25_retriever.invoke(sanitized_query)
            
            scores = {}
            doc_map = {}
            for rank, doc in enumerate(dense_results + sparse_results):
                content = doc.page_content
                scores[content] = scores.get(content, 0) + (1 / (60 + rank + 1))
                doc_map[content] = doc

            sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:3]
            reranked_docs = [doc_map[c] for c in sorted_contents]
            context_str = "\n\n".join([d.page_content for d in reranked_docs])
            sources = [d.page_content for d in reranked_docs]

        prompt_template = "You are an Enterprise System Architect. Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context_str if context_str else "N/A", question=sanitized_query)).content

        return {"answer": response, "sources": sources}

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem; border-bottom: 1px solid #334155; margin-bottom: 1.5rem;">
    <div>
        <h1 style="margin:0; font-size: 1.75rem;">Enterprise Cognitive Platform</h1>
        <p style="margin:0; color: #94A3B8; font-size: 0.85rem;">Hybrid Vector Search | Autonomous Agent Framework | Cloud Instance</p>
    </div>
    <div>
        <span style="background-color: #065F46; color: #34D399; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">STREAMLIT CLOUD ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None

with st.sidebar:
    st.markdown("<h3 style='font-size: 1.1rem;'>Platform Setup</h3>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter platform access key")
    
    if st.button("Initialize Engine"):
        if api_key:
            st.session_state.engine = CloudRAGEngine(openai_api_key=api_key)
            st.success("Engine Online.")
        else:
            st.warning("Key required.")

    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem;'>Knowledge Ingestion</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Specs (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Index Content"):
        if st.session_state.engine:
            saved_paths = []
            os.makedirs("./temp_docs", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("./temp_docs", f.name)
                with open(path, "wb") as file:
                    file.write(f.getbuffer())
                saved_paths.append(path)
            num_chunks = st.session_state.engine.index_documents(saved_paths)
            st.success(f"Indexed {num_chunks} chunks.")
        else:
            st.error("Initialize engine first.")

tab_query, tab_analytics = st.tabs(["Execution Console", "Platform Telemetry"])

with tab_query:
    col_text, col_voice = st.columns([2, 1])

    with col_text:
        text_prompt = st.text_area("Enter technical query", height=100)

    with col_voice:
        audio_file = st.audio_input("Record Voice Prompt")

    active_query = None

    if audio_file and st.session_state.engine:
        with st.spinner("Transcribing..."):
            active_query = st.session_state.engine.transcribe_audio(audio_file.getvalue())
            st.info(f"Transcribed: {active_query}")

    if text_prompt and not active_query:
        if st.button("Execute Query"):
            active_query = text_prompt

    if active_query:
        if st.session_state.engine:
            with st.spinner("Executing retrieval & inference..."):
                res = st.session_state.engine.process_query(active_query)
                answer = res["answer"]
                log_query("cloud-session-01", active_query, answer)

                st.markdown("<h3 style='font-size: 1.1rem; margin-top: 1rem;'>Assistant Output</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #1E293B; border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; font-size: 0.95rem; color: #E2E8F0;'>{answer}</div>", unsafe_allow_html=True)

                tts_script = f"<script>var msg = new SpeechSynthesisUtterance('{answer.replace("'", "")}'); window.speechSynthesis.speak(msg);</script>"
                st.components.v1.html(tts_script, height=0)

                st.markdown("<h4 style='font-size: 0.95rem; margin-top: 1rem; color: #94A3B8;'>Retrieved Context</h4>", unsafe_allow_html=True)
                for idx, src in enumerate(res["sources"]):
                    st.markdown(f"<div class='context-box'><b>Rank {idx+1}:</b><br>{src}</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter your API Key and click Initialize Engine first.")

with tab_analytics:
    st.markdown("<h3 style='font-size: 1.1rem;'>System Telemetry</h3>", unsafe_allow_html=True)
    if st.button("Refresh Telemetry"):
        total = get_analytics()
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Queries Executed</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
        c2.markdown("<div class='metric-card'><div class='metric-label'>Cloud System Status</div><div class='metric-value' style='color:#34D399;'>HEALTHY</div></div>", unsafe_allow_html=True)
