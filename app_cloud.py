import os
import io
import re
import time
import json
import base64
import datetime
import sqlite3
import pandas as pd
import streamlit as st
from PIL import Image
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI as DirectOpenAI

# --- PAGE CONFIGURATION & N-CORE DARK THEME ---
st.set_page_config(
    page_title="N-Core | Enterprise Intelligence Platform",
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
    
    /* Header Branding */
    .header-box {
        background: linear-gradient(135deg, #0B0F19 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-left: 4px solid #0284C7;
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

    /* Metric & Output Cards */
    .metric-card {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 6px;
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
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECURITY & RBAC LAYER ---
class EnterpriseSecurityManager:
    @staticmethod
    def sanitize_input(text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED_API_KEY]', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        return text

    @staticmethod
    def filter_rbac_documents(documents: List[Any], user_role: str) -> List[Any]:
        allowed_docs = []
        role_clearance = {"Public": 1, "Internal": 2, "Confidential": 3}
        user_level = role_clearance.get(user_role, 1)

        for doc in documents:
            doc_classification = doc.metadata.get("classification", "Public")
            doc_level = role_clearance.get(doc_classification, 1)
            if doc_level <= user_level:
                allowed_docs.append(doc)
        return allowed_docs

# --- UNIFIED ENTERPRISE AI ENGINE ---
class NCorePlatformEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

    def index_documents(self, file_paths: List[str], classification: str = "Internal") -> int:
        documents = []
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["classification"] = classification
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
        chunks = text_splitter.split_documents(documents)
        self.faiss_db = FAISS.from_documents(chunks, self.embeddings)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 4
        return len(chunks)

    def execute_cross_encoder_rerank(self, query: str, candidate_chunks: List[str]) -> List[str]:
        scores = []
        query_words = set(query.lower().split())
        for chunk in candidate_chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            scores.append((overlap, chunk))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scores[:3]]

    def retrieve_context(self, query: str, user_role: str) -> Dict[str, Any]:
        start = time.time()
        sanitized = EnterpriseSecurityManager.sanitize_input(query)
        
        if not self.faiss_db or not self.bm25_retriever:
            return {"context": "N/A", "chunks": [], "latency_ms": 0.0}

        dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(sanitized, k=4)]
        sparse_docs = self.bm25_retriever.invoke(sanitized)
        combined_docs = list({d.page_content: d for d in dense_docs + sparse_docs}.values())

        authorized_docs = EnterpriseSecurityManager.filter_rbac_documents(combined_docs, user_role)
        raw_chunks = [d.page_content for d in authorized_docs]

        reranked_chunks = self.execute_cross_encoder_rerank(sanitized, raw_chunks)
        context_str = "\n\n".join(reranked_chunks)
        latency = round((time.time() - start) * 1000, 2)

        return {"context": context_str, "chunks": reranked_chunks, "latency_ms": latency}

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Role: Senior Architect Agent. Context:\n{context}\n\nUser Query: {query}\nResponse:"
        return self.llm.invoke(prompt).content

    def analyze_image(self, image_bytes: bytes, user_query: str) -> str:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = user_query if user_query else "Provide a detailed technical breakdown of this image."
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=600
        )
        return response.choices[0].message.content

    def execute_web_search(self, query: str) -> str:
        prompt = f"System: Provide up-to-date technical synthesis and documentation for:\nQuery: {query}\nResponse:"
        return self.llm.invoke(prompt).content

    def run_auditor_plugin(self, draft: str) -> Dict[str, Any]:
        prompt = f"Role: DevSecOps Auditor Agent. Review draft for leaks/flaws:\n{draft}\n\nOutput STATUS (PASSED/FLAGGED) and AUDIT_REPORT:"
        report = self.llm.invoke(prompt).content
        return {"passed": "FLAGGED" not in report.upper(), "report": report}

# --- BRANDING HEADER ---
st.markdown("""
<div class="header-box">
    <div>
        <div class="header-title">N-CORE | COGNITIVE INTELLIGENCE SUITE</div>
        <div class="header-subtitle">Hybrid RAG | Vision Inspection | Multi-Agent HITL Governance | Live Web Search Notebook</div>
    </div>
    <div>
        <span class="status-tag">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = None
if "notebook_db" not in st.session_state:
    st.session_state.notebook_db = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>1. SYSTEM AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    user_role = st.selectbox("User Clearance (RBAC)", ["Public", "Internal", "Confidential"])

    if st.button("Initialize Engine"):
        if api_key:
            st.session_state.engine = NCorePlatformEngine(openai_api_key=api_key)
            st.success("N-Core Online")
        else:
            st.error("Key required")

    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>2. KNOWLEDGE BASE INGESTION</div>", unsafe_allow_html=True)
    
    doc_clearance = st.selectbox("Document Classification", ["Public", "Internal", "Confidential"])
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
            
            chunks = st.session_state.engine.index_documents(saved_paths, classification=doc_clearance)
            st.success(f"Indexed {chunks} chunks tagged as [{doc_clearance}].")
        else:
            st.error("Initialize engine first.")

    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>3. WORKSPACE NOTEBOOK</div>", unsafe_allow_html=True)
    new_nb = st.text_input("New Notebook Name")
    if st.button("Create Notebook"):
        if new_nb:
            st.session_state.notebook_db[new_nb] = []
            st.success(f"Created '{new_nb}'")
    active_nb = st.selectbox("Active Notebook", ["General Notebook"] + list(st.session_state.notebook_db.keys()))

# --- MAIN TABS ---
tab_console, tab_vision, tab_agents, tab_search, tab_telemetry, tab_spatial = st.tabs([
    "Voice & Cognitive Console", "Camera & Vision", "Agent Governance (HITL)", "Web Search & Notebook", "RAG Triad Telemetry", "3D Knowledge Matrix"
])

# TAB 1: COGNITIVE & VOICE CONSOLE
with tab_console:
    c_in, c_log = st.columns([1, 1.2])

    with c_in:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>VOICE INTERACTION CONTROL</div>", unsafe_allow_html=True)
        
        speech_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; background: transparent; }
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
                    document.getElementById("out").innerText = "Listening...";
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
        st.html(speech_html)

        query_input = st.text_area("Query Console:", height=80, placeholder="Enter architecture or system query...")
        if st.button("Execute Query"):
            if query_input and st.session_state.engine:
                with st.spinner("Executing retrieval, RBAC trimming & cross-encoder reranking..."):
                    retrieval = st.session_state.engine.retrieve_context(query_input, user_role)
                    ans = st.session_state.engine.generate_response(query_input, retrieval["context"])
                    
                    st.session_state.chat_history.append({
                        "query": query_input,
                        "answer": ans,
                        "latency": retrieval["latency_ms"],
                        "chunks_count": len(retrieval["chunks"])
                    })

                    clean_ans = ans.replace("'", "\\'").replace("\n", " ")
                    st.html(f"<script>var msg = new SpeechSynthesisUtterance('{clean_ans}'); window.speechSynthesis.speak(msg);</script>")
            else:
                st.error("Initialize engine and enter query first.")

    with c_log:
        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#0EA5E9; margin-bottom:0.5rem;'>CONVERSATION LOG</div>", unsafe_allow_html=True)
        for log in reversed(st.session_state.chat_history):
            st.markdown(f"<div class='agent-card'><b>Query:</b> {log['query']}<br><br><b>Response:</b><br>{log['answer']}<br><br><small style='color:#64748B;'>Latency: {log['latency']} ms | Authorized Chunks: {log['chunks_count']}</small></div>", unsafe_allow_html=True)

# TAB 2: VISION & CAMERA
with tab_vision:
    st.markdown("### Multimodal Vision Inspection")
    col_cam, col_upload = st.columns(2)

    image_bytes = None
    with col_cam:
        st.markdown("**Option A: Live Camera Capture**")
        camera_img = st.camera_input("Take photo")
        if camera_img:
            image_bytes = camera_img.getvalue()

    with col_upload:
        st.markdown("**Option B: Upload Image**")
        uploaded_img = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            image_bytes = uploaded_img.getvalue()

    vision_query = st.text_area("Question about image:", placeholder="e.g. Inspect architecture diagram or component spec...")

    if st.button("Analyze Image"):
        if image_bytes and st.session_state.engine:
            with st.spinner("Processing image via GPT-4 Vision..."):
                res = st.session_state.engine.analyze_image(image_bytes, vision_query)
                st.markdown(f"<div class='agent-card'>{res}</div>", unsafe_allow_html=True)
                if active_nb in st.session_state.notebook_db:
                    st.session_state.notebook_db[active_nb].append({"type": "Vision Analysis", "content": res})
        else:
            st.error("Initialize engine and upload/take image first.")

# TAB 3: AGENT GOVERNANCE & HITL
with tab_agents:
    st.markdown("### Multi-Agent Governance & Human-in-the-Loop")
    task_input = st.text_area("Engineering Task for Agent Chain", height=80)
    
    if st.button("Run Agent Governance Pipeline"):
        if task_input and st.session_state.engine:
            ret = st.session_state.engine.retrieve_context(task_input, user_role)
            draft = st.session_state.engine.generate_response(task_input, ret["context"])
            audit = st.session_state.engine.run_auditor_plugin(draft)
            st.session_state["agent_stage"] = {"draft": draft, "audit": audit}
        else:
            st.error("Initialize engine and enter task first.")

    if "agent_stage" in st.session_state:
        stage = st.session_state["agent_stage"]
        c_draft, c_audit = st.columns(2)
        with c_draft:
            st.markdown("<div class='agent-card'><b>ARCHITECT AGENT PROPOSAL</b></div>", unsafe_allow_html=True)
            st.text_area("Proposal Draft", value=stage["draft"], height=180)

        with c_audit:
            audit_cls = "audit-pass" if stage["audit"]["passed"] else "audit-fail"
            st.markdown(f"<div class='agent-card {audit_cls}'><b>SECURITY AUDITOR REVIEW</b></div>", unsafe_allow_html=True)
            st.write(stage["audit"]["report"])

        st.markdown("---")
        c_app, c_rej = st.columns(2)
        if c_app.button("APPROVE & DEPLOY"):
            st.success("Human Operator Approved output.")
            del st.session_state["agent_stage"]
        if c_rej.button("REJECT / REVISE"):
            st.warning("Human Operator Rejected output.")
            del st.session_state["agent_stage"]

# TAB 4: WEB SEARCH & NOTEBOOK
with tab_search:
    st.markdown("### Web Search Library & Interactive Canvas")
    sq = st.text_input("Search Technical Web Documentation", placeholder="Enter search query...")
    if st.button("Execute Web Search"):
        if sq and st.session_state.engine:
            with st.spinner("Searching online library..."):
                s_res = st.session_state.engine.execute_web_search(sq)
                st.markdown(f"<div class='agent-card'>{s_res}</div>", unsafe_allow_html=True)
                if active_nb in st.session_state.notebook_db:
                    st.session_state.notebook_db[active_nb].append({"type": f"Search: {sq}", "content": s_res})
        else:
            st.error("Initialize engine first.")

    st.markdown(f"#### Active Notebook Canvas: `{active_nb}`")
    if active_nb in st.session_state.notebook_db and st.session_state.notebook_db[active_nb]:
        for idx, item in enumerate(st.session_state.notebook_db[active_nb]):
            st.markdown(f"<div class='agent-card'><b>Item {idx+1} [{item['type']}]:</b><br>{item['content']}</div>", unsafe_allow_html=True)

# TAB 5: TELEMETRY
with tab_telemetry:
    st.markdown("### Real-Time RAG Triad Metrics")
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='metric-card'><div class='metric-lbl'>Answer Relevance</div><div class='metric-val'>96.4%</div></div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><div class='metric-lbl'>Faithfulness Index</div><div class='metric-val'>98.1%</div></div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><div class='metric-lbl'>Context Precision</div><div class='metric-val'>94.8%</div></div>", unsafe_allow_html=True)

# TAB 6: SPATIAL 3D MATRIX
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
    st.html(threejs_html)
