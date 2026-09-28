import os
import io
import re
import time
import datetime
import sqlite3
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI as DirectOpenAI

# --- ADVANCED ENTERPRISE CONFIG & THEME ---
st.set_page_config(
    page_title="Cognitive Knowledge Architecture Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #090D16;
        color: #E2E8F0;
    }
    .stSidebar {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Header Branding */
    .brand-header {
        border-bottom: 1px solid #1E293B;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.025em;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #64748B;
    }
    
    /* Enterprise Metric Cards */
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Source & Citation Cards */
    .citation-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-left: 3px solid #0EA5E9;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #CBD5E1;
    }
    .citation-meta {
        font-size: 0.75rem;
        color: #64748B;
        margin-bottom: 0.5rem;
        display: flex;
        gap: 1rem;
    }
    
    /* Custom Output Console */
    .console-output {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.5rem;
        color: #F1F5F9;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* Buttons & Inputs */
    .stButton>button {
        background-color: #0284C7;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0369A1;
    }
    </style>
""", unsafe_allow_html=True)

# --- ANALYTICS & TELEMETRY DATABASE ---
def init_analytics_db():
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS execution_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, 
                  query TEXT, 
                  response TEXT, 
                  latency_ms REAL,
                  chunks_retrieved INTEGER,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_execution(session_id: str, query: str, response: str, latency: float, chunks_count: int):
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute("INSERT INTO execution_logs (session_id, query, response, latency_ms, chunks_retrieved, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (session_id, query, response, latency, chunks_count, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()

def fetch_telemetry_dataframe():
    conn = sqlite3.connect("platform_telemetry.db")
    df = pd.read_sql_query("SELECT id, query, latency_ms, chunks_retrieved, timestamp FROM execution_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_analytics_db()

# --- SECURITY GUARDRAIL LAYER ---
class EnterpriseGuardrails:
    @staticmethod
    def sanitize_input(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
        return text

    @staticmethod
    def check_injection_vector(text: str) -> bool:
        forbidden_vectors = ["ignore system prompt", "override rules", "jailbreak mode", "unrestricted access"]
        return any(vec in text.lower() for vec in forbidden_vectors)

# --- ADVANCED HYBRID RAG & AGENT ENGINE ---
class EnterpriseRAGEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def transcribe_audio_stream(self, audio_bytes: bytes) -> str:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "stream_input.wav"
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
        chunks = text_splitter.split_documents(documents)
        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 4
        return len(chunks)

    def execute_hybrid_search(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if EnterpriseGuardrails.check_injection_vector(query):
            return {
                "answer": "Security Policy Enforcement: Potential prompt injection vector intercepted.",
                "sources": [],
                "latency_ms": 0.0
            }

        sanitized_query = EnterpriseGuardrails.sanitize_input(query)
        sources = []
        context_str = ""

        if self.faiss_db and self.bm25_retriever:
            dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized_query, k=4)]
            sparse_docs = self.bm25_retriever.invoke(sanitized_query)
            
            # Reciprocal Rank Fusion (RRF) Reranking
            rrf_scores = {}
            doc_mapping = {}
            for rank, doc in enumerate(dense_docs + sparse_docs):
                content = doc.page_content
                rrf_scores[content] = rrf_scores.get(content, 0) + (1 / (60 + rank + 1))
                doc_mapping[content] = doc

            sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:3]
            reranked_docs = [doc_mapping[c] for c in sorted_contents]
            context_str = "\n\n".join([d.page_content for d in reranked_docs])
            
            for rank, doc in enumerate(reranked_docs):
                sources.append({
                    "rank": rank + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Knowledge Base"),
                    "page": doc.metadata.get("page", 1)
                })

        prompt_template = """
        You are an Enterprise Systems Architect and Senior AI Assistant.
        Provide a precise, highly structured technical answer based on the provided context.

        Context:
        {context}

        User Query: {question}
        Response:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context_str if context_str else "N/A", question=sanitized_query)).content

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {"answer": response, "sources": sources, "latency_ms": elapsed_ms}

# --- HEADER BRANDING ---
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">Cognitive Knowledge Platform</div>
        <div class="brand-sub">Enterprise Vector Engine & Audio-Visual Intelligence Console</div>
    </div>
    <div>
        <span style="background-color: #0369A1; color: #E0F2FE; padding: 0.35rem 0.85rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;">SYSTEM OPERATIONAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None

# --- CONTROL PANEL SIDEBAR ---
with st.sidebar:
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F8FAFC; margin-bottom:0.75rem;'>API CONFIGURATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Token", type="password", help="Enter platform credentials")
    
    if st.button("Initialize Engine"):
        if api_key:
            st.session_state.engine = EnterpriseRAGEngine(openai_api_key=api_key)
            st.success("Cognitive Core Active")
        else:
            st.error("Access token required")

    st.markdown("<hr style='border-color: #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F8FAFC; margin-bottom:0.75rem;'>KNOWLEDGE BASE INGESTION</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Architecture Specs (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Build Vector Index"):
        if st.session_state.engine:
            saved_paths = []
            os.makedirs("./temp_docs", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("./temp_docs", f.name)
                with open(path, "wb") as file:
                    file.write(f.getbuffer())
                saved_paths.append(path)
            
            chunks_cnt = st.session_state.engine.index_documents(saved_paths)
            st.success(f"Indexed {chunks_cnt} vector embeddings.")
        else:
            st.error("Initialize engine first.")

# --- MAIN WORKSPACE ---
tab_execution, tab_analytics = st.tabs(["Execution Console", "Platform Telemetry & Audit Logs"])

with tab_execution:
    col_text, col_audio = st.columns([2, 1])

    with col_text:
        text_prompt = st.text_area("Query Interface", height=100, placeholder="Enter architecture query or document prompt...")

    with col_audio:
        audio_stream = st.audio_input("Voice Input Stream")

    active_query = None

    if audio_stream and st.session_state.engine:
        with st.spinner("Transcribing audio input..."):
            active_query = st.session_state.engine.transcribe_audio_stream(audio_stream.getvalue())
            st.info(f"Transcribed Query: {active_query}")

    if text_prompt and not active_query:
        if st.button("Submit Query"):
            active_query = text_prompt

    if active_query:
        if st.session_state.engine:
            with st.spinner("Processing hybrid retrieval & inference..."):
                res = st.session_state.engine.execute_hybrid_search(active_query)
                answer = res["answer"]
                latency = res["latency_ms"]
                sources = res["sources"]

                log_execution("session-prod-01", active_query, answer, latency, len(sources))

                st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F8FAFC; margin: 1.5rem 0 0.5rem 0;'>RESPONSE SYNTHESIS</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='console-output'>{answer}</div>", unsafe_allow_html=True)

                # Automated Speech Output
                tts_script = f"<script>var msg = new SpeechSynthesisUtterance('{answer.replace("'", "")}'); window.speechSynthesis.speak(msg);</script>"
                st.components.v1.html(tts_script, height=0)

                st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F8FAFC; margin: 1.5rem 0 0.5rem 0;'>RETRIEVED VECTOR CITATIONS</div>", unsafe_allow_html=True)
                for src in sources:
                    st.markdown(f"""
                    <div class='citation-card'>
                        <div class='citation-meta'>
                            <span>RANK: #{src['rank']}</span>
                            <span>SOURCE: {src['source']}</span>
                            <span>PAGE: {src['page']}</span>
                        </div>
                        {src['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please provide an Access Token and click Initialize Engine first.")

with tab_analytics:
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F8FAFC; margin-bottom:1rem;'>PERFORMANCE AUDIT TELEMETRY</div>", unsafe_allow_html=True)
    
    if st.button("Refresh Telemetry"):
        df = fetch_telemetry_dataframe()
        
        if not df.empty:
            avg_latency = round(df['latency_ms'].mean(), 2)
            total_queries = len(df)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Executions</div><div class='metric-value'>{total_queries}</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Latency</div><div class='metric-value'>{avg_latency} ms</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='metric-card'><div class='metric-label'>Cluster Status</div><div class='metric-value' style='color:#34D399;'>OPTIMAL</div></div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No query telemetry recorded yet.")
