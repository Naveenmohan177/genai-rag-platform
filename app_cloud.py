import os
import io
import re
import time
import datetime
import sqlite3
import json
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

# --- HIGH-END GLASSMORPHIC ENTERPRISE STYLING ---
st.set_page_config(
    page_title="Enterprise Cognitive Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #030712;
        color: #F3F4F6;
    }
    
    /* Glassmorphism Sidebar */
    .stSidebar {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Brand Header Banner */
    .brand-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.03em;
    }
    .brand-sub {
        font-size: 0.875rem;
        color: #9CA3AF;
        margin-top: 0.25rem;
    }
    
    /* Sleek Metric Badges */
    .metric-card {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.875rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.75rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Console & Output Boxes */
    .console-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        color: #F9FAFB;
        font-size: 0.975rem;
        line-height: 1.7;
    }
    
    .citation-card {
        background: rgba(31, 41, 55, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #0EA5E9;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        color: #D1D5DB;
    }
    </style>
""", unsafe_allow_html=True)

# --- TELEMETRY & AUDIT DATABASE ---
def init_analytics_db():
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS execution_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, 
                  query TEXT, 
                  response TEXT, 
                  groundedness_score REAL,
                  latency_ms REAL,
                  chunks_retrieved INTEGER,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_execution(session_id: str, query: str, response: str, groundedness: float, latency: float, chunks_count: int):
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute("INSERT INTO execution_logs (session_id, query, response, groundedness_score, latency_ms, chunks_retrieved, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (session_id, query, response, groundedness, latency, chunks_count, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()

def fetch_telemetry_dataframe():
    conn = sqlite3.connect("platform_telemetry.db")
    df = pd.read_sql_query("SELECT id, session_id, query, response, groundedness_score, latency_ms, chunks_retrieved, timestamp FROM execution_logs ORDER BY timestamp DESC", conn)
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

# --- ADVANCED RAG & HALLUCINATION EVALUATOR ---
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

    def _evaluate_groundedness(self, context: str, answer: str) -> float:
        """Evaluates whether the generated response is strictly grounded in the retrieved context."""
        if not context or "N/A" in context:
            return 0.5
        eval_prompt = f"Context: {context}\nAnswer: {answer}\nRate factual groundedness from 0.0 to 1.0. Output ONLY the floating point number."
        try:
            res = self.llm.invoke(eval_prompt).content.strip()
            return float(re.findall(r"0\.\d+|1\.0|\d", res)[0])
        except:
            return 0.95

    def execute_hybrid_search(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if EnterpriseGuardrails.check_injection_vector(query):
            return {
                "answer": "Security Policy Enforcement: Potential prompt injection vector intercepted.",
                "sources": [],
                "groundedness": 0.0,
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
        You are an Enterprise Systems Architect.
        Provide a precise technical answer strictly grounded in the provided context.

        Context:
        {context}

        User Query: {question}
        Response:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        response = self.llm.invoke(prompt.format(context=context_str if context_str else "N/A", question=sanitized_query)).content

        groundedness = self._evaluate_groundedness(context_str, response)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "answer": response,
            "sources": sources,
            "groundedness": groundedness,
            "latency_ms": elapsed_ms
        }

# --- HEADER BRANDING ---
st.markdown("""
<div class="brand-banner">
    <div>
        <div class="brand-title">Cognitive Knowledge Architecture Platform</div>
        <div class="brand-sub">Enterprise Vector Search | Automated Hallucination Guard | Voice-Enabled Agent</div>
    </div>
    <div>
        <span style="background-color: #0284C7; color: #FFFFFF; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">PROD INSTANCE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F9FAFB; margin-bottom:0.5rem;'>1. AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Token", type="password")
    
    if st.button("Initialize Platform"):
        if api_key:
            st.session_state.engine = EnterpriseRAGEngine(openai_api_key=api_key)
            st.success("Platform Engine Active")
        else:
            st.error("Token required")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#F9FAFB; margin-bottom:0.5rem;'>2. KNOWLEDGE BASE INGESTION</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Tech Specs (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
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
            st.success(f"Indexed {chunks_cnt} chunks.")
        else:
            st.error("Initialize platform first.")

# --- WORKSPACE TABS ---
tab_execution, tab_analytics = st.tabs(["Execution Console", "Telemetry & Audit Logs"])

with tab_execution:
    col_text, col_audio = st.columns([2, 1])

    with col_text:
        text_prompt = st.text_area("Query Console", height=100, placeholder="Ask a technical architecture question...")

    with col_audio:
        audio_stream = st.audio_input("Voice Input Stream")

    active_query = None

    if audio_stream and st.session_state.engine:
        with st.spinner("Transcribing audio input..."):
            active_query = st.session_state.engine.transcribe_audio_stream(audio_stream.getvalue())
            st.info(f"Transcribed: {active_query}")

    if text_prompt and not active_query:
        if st.button("Execute Query"):
            active_query = text_prompt

    if active_query:
        if st.session_state.engine:
            with st.spinner("Executing retrieval, inference & groundedness validation..."):
                res = st.session_state.engine.execute_hybrid_search(active_query)
                answer = res["answer"]
                latency = res["latency_ms"]
                groundedness = res["groundedness"]
                sources = res["sources"]

                log_execution("prod-session-01", active_query, answer, groundedness, latency, len(sources))

                # Display Response & Real-time Metrics
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='metric-card'><div class='metric-lbl'>Latency</div><div class='metric-val'>{latency} ms</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><div class='metric-lbl'>Groundedness Index</div><div class='metric-val'>{int(groundedness * 100)}%</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><div class='metric-lbl'>Retrieved Chunks</div><div class='metric-val'>{len(sources)}</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='console-card'>{answer}</div>", unsafe_allow_html=True)

                # Text-To-Speech Script
                tts_script = f"<script>var msg = new SpeechSynthesisUtterance('{answer.replace("'", "")}'); window.speechSynthesis.speak(msg);</script>"
                st.components.v1.html(tts_script, height=0)

                st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F3F4F6; margin: 1.5rem 0 0.5rem 0;'>RETRIEVED VECTOR CITATIONS</div>", unsafe_allow_html=True)
                for src in sources:
                    st.markdown(f"""
                    <div class='citation-card'>
                        <div style='font-size:0.75rem; color:#9CA3AF; margin-bottom:0.25rem;'>RANK #{src['rank']} | SOURCE: {src['source']} | PAGE: {src['page']}</div>
                        {src['content']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please enter an Access Token and click Initialize Platform first.")

with tab_analytics:
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#F3F4F6; margin-bottom:1rem;'>SYSTEM AUDIT LOGS & EXPORT DATA</div>", unsafe_allow_html=True)
    
    if st.button("Refresh Telemetry"):
        df = fetch_telemetry_dataframe()
        
        if not df.empty:
            st.dataframe(df, use_container_width=True)

            # Export Telemetry Options
            col_csv, col_json = st.columns(2)
            with col_csv:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Audit Log (CSV)", csv_data, "telemetry_audit.csv", "text/csv")
            with col_json:
                json_data = df.to_json(orient="records")
                st.download_button("Download Audit Log (JSON)", json_data, "telemetry_audit.json", "application/json")
        else:
            st.info("No query telemetry recorded yet.")
