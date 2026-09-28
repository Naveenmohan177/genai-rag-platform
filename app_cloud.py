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

# --- PAGE CONFIGURATION & MINIMALIST SLATE THEME ---
st.set_page_config(
    page_title="Cognitive Intelligence Core",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Professional Styling (No Emojis/Logos)
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
    
    /* Sleek Clean Header */
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

    /* Cards & Metric Visuals */
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

    .output-card {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-left: 3px solid #0EA5E9;
        border-radius: 6px;
        padding: 1.25rem;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .citation-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 4px;
        padding: 0.85rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #94A3B8;
    }

    .stButton>button {
        background-color: #0284C7;
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECURITY GUARDRAIL LAYER ---
class SecurityGuardrails:
    @staticmethod
    def mask_pii(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text)
        return text

    @staticmethod
    def validate_safety(text: str) -> bool:
        forbidden_vectors = ["ignore previous rules", "override system prompt", "jailbreak mode"]
        return not any(vec in text.lower() for vec in forbidden_vectors)

# --- ADVANCED HYBRID RAG & COGNITIVE ENGINE ---
class EnterpriseCognitiveEngine:
    def __init__(self, openai_api_key: str, temperature: float = 0.1):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
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
        self.bm25_retriever.k = 4
        return len(chunks)

    def execute_query(self, query: str, mode: str) -> Dict[str, Any]:
        start = time.time()
        
        if not SecurityGuardrails.validate_safety(query):
            return {
                "answer": "Security Policy Enforcement: Intercepted untrusted prompt structure.",
                "sources": [],
                "latency_ms": 0.0
            }

        sanitized_query = SecurityGuardrails.mask_pii(query)
        sources = []
        context_str = ""

        if self.faiss_db and self.bm25_retriever:
            dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized_query, k=4)]
            sparse_docs = self.bm25_retriever.invoke(sanitized_query)
            
            # Reciprocal Rank Fusion (RRF)
            scores = {}
            doc_map = {}
            for rank, doc in enumerate(dense_docs + sparse_docs):
                c = doc.page_content
                scores[c] = scores.get(c, 0) + (1 / (60 + rank + 1))
                doc_map[c] = doc

            sorted_contents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:3]
            reranked = [doc_map[c] for c in sorted_contents]
            context_str = "\n\n".join([d.page_content for d in reranked])
            sources = [d.page_content for d in reranked]

        if mode == "Deep Chain-of-Thought Reasoning":
            sys_instructions = "Perform systematic step-by-step technical analysis using this context:\n" + context_str
        elif mode == "Code & Manifest Synthesizer":
            sys_instructions = "Generate production-ready code, Kubernetes manifests, or API definitions based on:\n" + context_str
        else:
            sys_instructions = "Provide a direct, high-precision technical response grounded in context:\n" + context_str

        prompt = PromptTemplate(template="{sys}\n\nUser Query: {q}\nResponse:", input_variables=["sys", "q"])
        response = self.llm.invoke(prompt.format(sys=sys_instructions, q=sanitized_query)).content
        elapsed = round((time.time() - start) * 1000, 2)

        return {"answer": response, "sources": sources, "latency_ms": elapsed}

# --- CLEAN ENTERPRISE HEADER ---
st.markdown("""
<div class="header-box">
    <div>
        <div class="header-title">COGNITIVE KNOWLEDGE ARCHITECTURE</div>
        <div class="header-subtitle">Hybrid Vector Search Engine | WebGL Spatial Graph | Automated Telemetry Audit</div>
    </div>
    <div>
        <span class="status-tag">SYSTEM ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None

# --- SIDEBAR CONTROL MATRIX ---
with st.sidebar:
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>1. SYSTEM AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    
    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>2. COGNITIVE PARAMETERS</div>", unsafe_allow_html=True)
    
    reasoning_mode = st.selectbox("Execution Mode", ["Direct Vector RAG", "Deep Chain-of-Thought Reasoning", "Code & Manifest Synthesizer"])
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.1, 0.05)
    chunk_size = st.slider("Vector Chunk Size", 200, 1000, 450, 50)

    if st.button("Initialize Engine"):
        if api_key:
            st.session_state.engine = EnterpriseCognitiveEngine(openai_api_key=api_key, temperature=temperature)
            st.success("Cognitive Core Online")
        else:
            st.error("Key required")

    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>3. KNOWLEDGE BASE INGESTION</div>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Specs (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Build Index"):
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

# --- MAIN WORKSPACE LAYOUT (3D GRAPH + CONSOLE) ---
col_graph, col_console = st.columns([1, 1.2])

with col_graph:
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>SPATIAL KNOWLEDGE MATRIX (3D)</div>", unsafe_allow_html=True)
    
    # Clean Three.js WebGL Node Graph Visualization (Zero Logos)
    threejs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style> body { margin: 0; overflow: hidden; background-color: #030712; } </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(380, 380);
            document.body.appendChild(renderer.domElement);

            // Create Spatial Wireframe Node Sphere
            const geometry = new THREE.IcosahedronGeometry(2, 2);
            const material = new THREE.MeshStandardMaterial({ color: 0x0EA5E9, wireframe: true });
            const sphere = new THREE.Mesh(geometry, material);
            scene.add(sphere);

            // Add Orbital Ring
            const ringGeo = new THREE.RingGeometry(2.6, 2.65, 64);
            const ringMat = new THREE.MeshBasicMaterial({ color: 0x38BDF8, side: THREE.DoubleSide, transparent: true, opacity: 0.5 });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            scene.add(ring);

            const light = new THREE.PointLight(0x0EA5E9, 2, 100);
            light.position.set(10, 10, 10);
            scene.add(light);
            camera.position.z = 5.5;

            function animate() {
                requestAnimationFrame(animate);
                sphere.rotation.y += 0.003;
                ring.rotation.z += 0.002;
                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(threejs_html, height=390)

with col_console:
    st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>EXECUTION CONSOLE</div>", unsafe_allow_html=True)
    
    text_input = st.text_area("Query Console", height=100, placeholder="Enter architecture query or engineering requirement...")
    
    if st.button("Execute Query"):
        if text_input and st.session_state.engine:
            with st.spinner("Processing through Cognitive Engine..."):
                res = st.session_state.engine.execute_query(text_input, mode=reasoning_mode)
                answer = res["answer"]
                latency = res["latency_ms"]
                sources = res["sources"]

                m1, m2 = st.columns(2)
                m1.markdown(f"<div class='metric-card'><div class='metric-lbl'>Execution Latency</div><div class='metric-val'>{latency} ms</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-lbl'>Retrieved Chunks</div><div class='metric-val'>{len(sources)}</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='output-card'>{answer}</div>", unsafe_allow_html=True)

                # Text-To-Speech Synthesis Output
                clean_answer = answer.replace("'", "\\'").replace("\n", " ")
                tts_script = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance('{clean_answer}');
                    window.speechSynthesis.speak(msg);
                </script>
                """
                components.html(tts_script, height=0)

                if sources:
                    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#64748B; margin-top:1rem;'>VECTOR CITATIONS</div>", unsafe_allow_html=True)
                    for idx, src in enumerate(sources):
                        st.markdown(f"<div class='citation-card'><b>Rank {idx+1}:</b><br>{src}</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter a query and initialize engine first.")
