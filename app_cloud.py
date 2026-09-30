import os
import io
import re
import time
import datetime
import sqlite3
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI as DirectOpenAI

# --- PAGE CONFIGURATION & ENTERPRISE DARK THEME ---
st.set_page_config(
    page_title="Enterprise Cognitive Core & Voice Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background-color: #030712;
        color: #F9FAFB;
    }
    .stSidebar {
        background-color: #0B0F19;
        border-right: 1px solid #1E293B;
    }
    
    .header-box {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.825rem;
        color: #64748B;
        margin-top: 0.15rem;
    }
    .status-tag {
        background-color: #0284C7;
        color: #FFFFFF;
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .agent-card {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .audit-pass { border-left: 4px solid #10B981; }
    .audit-fail { border-left: 4px solid #EF4444; }

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

# --- SECURITY GUARDRAIL LAYER ---
class SecurityGuardrails:
    @staticmethod
    def sanitize(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        return text

# --- MODULAR AGENT PLUGIN ENGINE ---
class AgentPluginEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def index_documents(self, file_paths: List[str], chunk_size: int = 450) -> int:
        documents = []
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=60)
        chunks = text_splitter.split_documents(documents)
        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 3
        return len(chunks)

    def retrieve_context(self, query: str) -> str:
        if self.faiss_db and self.bm25_retriever:
            dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(query, k=3)]
            sparse_docs = self.bm25_retriever.invoke(query)
            return "\n\n".join([d.page_content for d in dense_docs + sparse_docs])
        return "N/A"

    def run_architect_plugin(self, query: str, context: str) -> str:
        prompt = f"Role: Senior Architect Agent. Context:\n{context}\n\nQuery: {query}\nResponse:"
        return self.llm.invoke(prompt).content

    def run_auditor_plugin(self, draft: str) -> Dict[str, Any]:
        prompt = f"Role: DevSecOps Auditor Agent. Review this draft for vulnerabilities or leaks:\n{draft}\n\nOutput STATUS (PASSED/FLAGGED) and AUDIT_REPORT:"
        report = self.llm.invoke(prompt).content
        return {"passed": "FLAGGED" not in report.upper(), "report": report}

# --- HEADER BRANDING ---
st.markdown("""
<div class="header-box">
    <div>
        <div class="header-title">ENTERPRISE COGNITIVE ENGINE & AGENT HUB</div>
        <div class="header-subtitle">Voice-First AI Chat | Document Ingestion | Modular Agent Plugins | 3D WebGL Matrix</div>
    </div>
    <div>
        <span class="status-tag">SYSTEM OPERATIONAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None
if "voice_chat_log" not in st.session_state:
    st.session_state.voice_chat_log = []

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>1. SYSTEM AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    
    if st.button("Initialize Platform"):
        if api_key:
            st.session_state.engine = AgentPluginEngine(openai_api_key=api_key)
            st.success("Platform Core Active")
        else:
            st.error("Key required")

    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>2. DOCUMENT KNOWLEDGE INGESTION</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Specs (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Build Vector Index"):
        if st.session_state.engine:
            saved_paths = []
            os.makedirs("./temp_docs", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("./temp_docs", f.name)
                with open(path, "wb") as file:
                    file.write(f.getbuffer())
                saved_paths.append(path)
            
            chunks = st.session_state.engine.index_documents(saved_paths)
            st.success(f"Indexed {chunks} chunks across documents.")
        else:
            st.error("Initialize engine first.")

# --- WORKSPACE TABS ---
tab_voice, tab_plugins, tab_spatial = st.tabs(["Voice & Document Console", "Agent Plugin Governance", "3D Knowledge Spatial"])

# TAB 1: VOICE CHAT & QUERY CONSOLE
with tab_voice:
    col_input, col_display = st.columns([1, 1.2])

    with col_input:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>VOICE INTERACTION CONTROL</div>", unsafe_allow_html=True)
        
        # Native Web Speech Recognition Component
        speech_component = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .v-btn {
                    background-color: #0284C7; color: white; border: none; padding: 0.75rem;
                    font-weight: 700; border-radius: 4px; cursor: pointer; width: 100%;
                    font-family: sans-serif;
                }
                .v-btn:hover { background-color: #0369A1; }
            </style>
        </head>
        <body>
            <button class="v-btn" onclick="startRecognition()">ACTIVATE VOICE SPEECH INPUT</button>
            <p id="out" style="color: #64748B; font-size: 0.8rem; margin-top: 0.4rem;"></p>
            <script>
                function startRecognition() {
                    window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (!window.SpeechRecognition) { alert("Use Chrome browser for voice support."); return; }
                    const rec = new SpeechRecognition();
                    document.getElementById("out").innerText = "Listening to speech...";
                    rec.onresult = (e) => {
                        const text = e.results[0][0].transcript;
                        document.getElementById("out").innerText = "Recognized: " + text;
                        window.parent.postMessage({type: "streamlit:setComponentValue", value: text}, "*");
                    };
                    rec.start();
                }
            </script>
        </body>
        </html>
        """
        components.html(speech_component, height=100)

        query_text = st.text_area("Or enter document query directly:", height=80, placeholder="Ask a technical question about uploaded documents...")
        
        if st.button("Execute Document Query"):
            if query_text and st.session_state.engine:
                with st.spinner("Processing hybrid retrieval & synthesis..."):
                    context = st.session_state.engine.retrieve_context(query_text)
                    ans = st.session_state.engine.run_architect_plugin(query_text, context)
                    
                    st.session_state.voice_chat_log.append({"query": query_text, "response": ans})

                    # Automated Text-To-Speech Readout
                    clean_ans = ans.replace("'", "\\'").replace("\n", " ")
                    tts_script = f"<script>var msg = new SpeechSynthesisUtterance('{clean_ans}'); window.speechSynthesis.speak(msg);</script>"
                    components.html(tts_script, height=0)
            else:
                st.error("Initialize engine and enter query first.")

    with col_display:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>CONVERSATION LOG & AUDIO OUTPUT</div>", unsafe_allow_html=True)
        for log in reversed(st.session_state.voice_chat_log):
            st.markdown(f"<div class='agent-card'><b>Query:</b> {log['query']}<br><br><b>Response:</b><br>{log['response']}</div>", unsafe_allow_html=True)

# TAB 2: AGENT PLUGINS & HUMAN-IN-THE-LOOP
with tab_plugins:
    st.markdown("### Agent Plugins & Human Approval Intercepts")
    task_input = st.text_area("Engineering Task for Agent Chain", height=80)
    
    if st.button("Run Agent Plugin Pipeline"):
        if task_input and st.session_state.engine:
            context = st.session_state.engine.retrieve_context(task_input)
            draft = st.session_state.engine.run_architect_plugin(task_input, context)
            audit = st.session_state.engine.run_auditor_plugin(draft)
            
            st.session_state["agent_stage"] = {"draft": draft, "audit": audit}
        else:
            st.error("Initialize engine and enter task first.")

    if "agent_stage" in st.session_state:
        stage = st.session_state["agent_stage"]
        
        c_draft, c_audit = st.columns(2)
        with c_draft:
            st.markdown("<div class='agent-card'><b>ARCHITECT AGENT PLUGIN</b></div>", unsafe_allow_html=True)
            st.text_area("Generated Output", value=stage["draft"], height=180)

        with c_audit:
            audit_cls = "audit-pass" if stage["audit"]["passed"] else "audit-fail"
            st.markdown(f"<div class='agent-card {audit_cls}'><b>SECURITY AUDITOR AGENT PLUGIN</b></div>", unsafe_allow_html=True)
            st.write(stage["audit"]["report"])

        st.markdown("---")
        c_app, c_rej = st.columns(2)
        if c_app.button("APPROVE OUTPUT"):
            st.success("Human Operator Approved output.")
            del st.session_state["agent_stage"]
        if c_rej.button("REJECT OUTPUT"):
            st.warning("Human Operator Rejected output.")
            del st.session_state["agent_stage"]

# TAB 3: 3D SPATIAL KNOWLEDGE MATRIX
with tab_spatial:
    threejs_html = """
    <!DOCTYPE html><html><head><style>body { margin: 0; overflow: hidden; background: #030712; }</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head>
    <body><script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(400, 400); document.body.appendChild(renderer.domElement);
        const sphere = new THREE.Mesh(new THREE.IcosahedronGeometry(2, 2), new THREE.MeshStandardMaterial({ color: 0x0EA5E9, wireframe: true }));
        scene.add(sphere);
        const light = new THREE.PointLight(0x0EA5E9, 2, 100); light.position.set(10, 10, 10); scene.add(light);
        camera.position.z = 5.5;
        function animate() { requestAnimationFrame(animate); sphere.rotation.y += 0.003; renderer.render(scene, camera); }
        animate();
    </script></body></html>
    """
    components.html(threejs_html, height=410)
