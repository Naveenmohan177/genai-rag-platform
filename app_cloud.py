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

# --- PAGE CONFIGURATION & HIGH-CONTRAST DARK THEME ---
st.set_page_config(
    page_title="Cybernetic Cognitive Core 3D",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp {
        background-color: #020617;
        color: #F8FAFC;
    }
    .stSidebar {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    
    /* Header Container */
    .header-banner {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(3, 7, 18, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 1.25rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: -0.02em;
    }
    .header-sub {
        font-size: 0.85rem;
        color: #94A3B8;
    }
    
    /* Sleek Cards */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    
    .console-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        color: #F1F5F9;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

# --- TELEMETRY DATABASE ---
def init_db():
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS execution_logs 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, query TEXT, response TEXT, 
                  mode TEXT, latency_ms REAL, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def log_event(session_id: str, query: str, response: str, mode: str, latency: float):
    conn = sqlite3.connect("platform_telemetry.db")
    c = conn.cursor()
    c.execute("INSERT INTO execution_logs (session_id, query, response, mode, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
              (session_id, query, response, mode, latency, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()

init_db()

# --- SECURITY GUARDRAILS ---
class SecurityGuardrails:
    @staticmethod
    def sanitize_input(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        return text

# --- ADVANCED RAG & AGENT ENGINE ---
class Advanced3DAgentEngine:
    def __init__(self, openai_api_key: str, temp: float = 0.1):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio_stream.wav"
        return self.client.audio.transcriptions.create(model="whisper-1", file=audio_file).text

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
        self.bm25_retriever.k = 4
        return len(chunks)

    def process_query(self, query: str, mode: str) -> Dict[str, Any]:
        start = time.time()
        sanitized = SecurityGuardrails.sanitize_input(query)
        sources = []
        context_str = ""

        if self.faiss_db and self.bm25_retriever:
            dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized, k=4)]
            sparse_docs = self.bm25_retriever.invoke(sanitized)
            
            # Reciprocal Rank Fusion
            scores = {}
            doc_map = {}
            for rank, doc in enumerate(dense_docs + sparse_docs):
                c = doc.page_content
                scores[c] = scores.get(c, 0) + (1 / (60 + rank + 1))
                doc_map[c] = doc

            sorted_c = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:3]
            reranked = [doc_map[c] for c in sorted_c]
            context_str = "\n\n".join([d.page_content for d in reranked])
            sources = [d.page_content for d in reranked]

        if mode == "Deep Chain-of-Thought Reasoning":
            sys_prompt = "Perform step-by-step architectural reasoning using this context:\n" + context_str
        elif mode == "Executive Summary":
            sys_prompt = "Provide a high-level executive bulleted summary based on:\n" + context_str
        else:
            sys_prompt = "Answer technically and precisely based on context:\n" + context_str

        prompt = PromptTemplate(template="{sys}\n\nQuestion: {q}\nAnswer:", input_variables=["sys", "q"])
        res = self.llm.invoke(prompt.format(sys=sys_prompt, q=sanitized)).content
        latency = round((time.time() - start) * 1000, 2)

        return {"answer": res, "sources": sources, "latency_ms": latency}

# --- HEADER BRANDING ---
st.markdown("""
<div class="header-banner">
    <div>
        <div class="header-title">CYBERNETIC COGNITIVE CORE 3D</div>
        <div class="header-sub">Spatial WebGL Engine | Multi-Agent RAG | Real-Time Voice Intelligence</div>
    </div>
    <div>
        <span style="background: rgba(56, 189, 248, 0.2); color: #38BDF8; border: 1px solid #38BDF8; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">3D SPATIAL ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None

# --- SIDEBAR CONFIGURATION & TOOL MATRIX ---
with st.sidebar:
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#38BDF8; margin-bottom:0.5rem;'>1. AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    
    st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#38BDF8; margin-bottom:0.5rem;'>2. AGENT HYPERPARAMETERS</div>", unsafe_allow_html=True)
    
    reasoning_mode = st.selectbox("Reasoning Engine Mode", ["Direct Vector RAG", "Deep Chain-of-Thought Reasoning", "Executive Summary"])
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1, 0.05)
    chunk_size = st.slider("Embedding Chunk Size", 200, 1000, 450, 50)

    if st.button("Initialize Cognitive Engine"):
        if api_key:
            st.session_state.engine = Advanced3DAgentEngine(openai_api_key=api_key, temp=temperature)
            st.success("Cognitive Core Online")
        else:
            st.error("API Key required")

    st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.15); margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#38BDF8; margin-bottom:0.5rem;'>3. KNOWLEDGE INGESTION</div>", unsafe_allow_html=True)
    
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
            
            chunks = st.session_state.engine.index_documents(saved_paths, chunk_size=chunk_size)
            st.success(f"Indexed {chunks} chunks.")
        else:
            st.error("Initialize engine first.")

# --- MAIN WORKSPACE LAYOUT (3D MOON + CONSOLE) ---
col_3d, col_console = st.columns([1, 1.2])

with col_3d:
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#38BDF8; margin-bottom:0.5rem;'>SPATIAL 3D HOLOGRAPHIC CORE</div>", unsafe_allow_html=True)
    
    # THREE.JS 3D MOON / TECH SPHERE WEBGL EMBED
    threejs_moon_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; overflow: hidden; background-color: #020617; }
            #canvas-container { width: 100%; height: 380px; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container"></div>
        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            // Create Holographic Moon / Tech Sphere
            const geometry = new THREE.SphereGeometry(2, 64, 64);
            const material = new THREE.MeshStandardMaterial({
                color: 0x0EA5E9,
                wireframe: true,
                emissive: 0x0284C7,
                emissiveIntensity: 0.4,
                roughness: 0.2
            });
            const moon = new THREE.Mesh(geometry, material);
            scene.add(moon);

            // Orbital Ring
            const ringGeo = new THREE.RingGeometry(2.6, 2.7, 64);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x38BDF8, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            scene.add(ring);

            // Lights
            const pointLight = new THREE.PointLight(0x38BDF8, 2, 100);
            pointLight.position.set(10, 10, 10);
            scene.add(pointLight);
            
            const ambientLight = new THREE.AmbientLight(0x020617, 1.5);
            scene.add(ambientLight);

            camera.position.z = 6;

            // Interactive Dragging
            let isDragging = false;
            let previousMousePosition = { x: 0, y: 0 };

            document.addEventListener('mousedown', () => isDragging = true);
            document.addEventListener('mouseup', () => isDragging = false);
            document.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;
                    moon.rotation.y += deltaX * 0.005;
                    moon.rotation.x += deltaY * 0.005;
                }
                previousMousePosition = { x: e.clientX, y: e.clientY };
            });

            // Animation Loop
            function animate() {
                requestAnimationFrame(animate);
                if (!isDragging) {
                    moon.rotation.y += 0.003;
                    ring.rotation.z += 0.002;
                }
                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(threejs_moon_html, height=390)

with col_console:
    st.markdown("<div style='font-size:0.9rem; font-weight:700; color:#38BDF8; margin-bottom:0.5rem;'>MULTIMODAL INTELLIGENCE CONSOLE</div>", unsafe_allow_html=True)
    
    text_input = st.text_area("Text Query Interface", height=90, placeholder="Type technical query...")
    audio_stream = st.audio_input("Voice Input Stream")

    active_prompt = None

    if audio_stream and st.session_state.engine:
        with st.spinner("Transcribing voice via Whisper..."):
            active_prompt = st.session_state.engine.transcribe_audio(audio_stream.getvalue())
            st.info(f"Transcribed: {active_prompt}")

    if text_input and not active_prompt:
        if st.button("Execute Intelligence Query"):
            active_prompt = text_input

    if active_prompt:
        if st.session_state.engine:
            with st.spinner("Processing through 3D Spatial RAG Engine..."):
                res = st.session_state.engine.process_query(active_prompt, mode=reasoning_mode)
                answer = res["answer"]
                latency = res["latency_ms"]
                sources = res["sources"]

                log_event("prod-3d-01", active_prompt, answer, reasoning_mode, latency)

                m1, m2 = st.columns(2)
                m1.markdown(f"<div class='metric-box'><div class='metric-lbl'>Execution Latency</div><div class='metric-val'>{latency} ms</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-box'><div class='metric-lbl'>Mode Active</div><div class='metric-val' style='font-size:1rem; padding-top:0.3rem;'>{reasoning_mode}</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='console-card'>{answer}</div>", unsafe_allow_html=True)

                # Text-To-Speech Output
                tts_script = f"<script>var msg = new SpeechSynthesisUtterance('{answer.replace("'", "")}'); window.speechSynthesis.speak(msg);</script>"
                components.html(tts_script, height=0)

                with st.expander("Retrieved Context Chunks"):
                    for idx, src in enumerate(sources):
                        st.info(f"**Chunk {idx+1}:**\n{src}")
        else:
            st.error("Please enter OpenAI API Key and initialize engine first.")
