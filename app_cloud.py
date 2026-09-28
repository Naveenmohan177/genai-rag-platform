import os
import io
import time
import datetime
import sqlite3
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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Voice Assistant & Spatial 3D Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION FOR THEMES & COLOR ACCENTS ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark Cyber"
if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#38BDF8"  # Electric Blue Default
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "engine" not in st.session_state:
    st.session_state.engine = None

# --- DYNAMIC THEME & COLOR STYLING ---
theme_bg = "#020617" if st.session_state.theme_mode == "Dark Cyber" else ("#0F172A" if st.session_state.theme_mode == "Neon Cyberpunk" else "#F8FAFC")
card_bg = "#0F172A" if st.session_state.theme_mode != "Light Minimal" else "#FFFFFF"
text_color = "#F8FAFC" if st.session_state.theme_mode != "Light Minimal" else "#0F172A"
accent = st.session_state.accent_color

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
    }}
    .stApp {{
        background-color: {theme_bg};
        color: {text_color};
    }}
    .stSidebar {{
        background: {card_bg};
        border-right: 1px solid {accent}33;
    }}
    
    .header-banner {{
        background: linear-gradient(135deg, {accent}22 0%, {theme_bg} 100%);
        border: 1px solid {accent}44;
        border-radius: 12px;
        padding: 1.25rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .header-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {accent};
    }}
    
    .chat-bubble-user {{
        background-color: {accent}22;
        border-left: 4px solid {accent};
        padding: 0.85rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        color: {text_color};
    }}
    .chat-bubble-ai {{
        background-color: {card_bg};
        border: 1px solid {accent}44;
        padding: 0.85rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# --- VOICE ASSISTANT AGENT ENGINE ---
class VoiceAgentEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.faiss_db = None
        self.bm25_retriever = None

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
        self.bm25_retriever.k = 3
        return len(chunks)

    def process_voice_query(self, query: str) -> str:
        context_str = ""
        if self.faiss_db and self.bm25_retriever:
            dense_docs = [doc for doc, _ in self.faiss_db.similarity_search_with_score(query, k=3)]
            sparse_docs = self.bm25_retriever.invoke(query)
            context_str = "\n\n".join([d.page_content for d in dense_docs + sparse_docs])

        prompt_template = """
        You are a Voice-First AI Assistant. Keep answers direct, concise, and conversational (1-3 sentences max) so they sound natural when spoken aloud.
        Context:
        {context}

        User Voice Query: {question}
        Spoken Response:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return self.llm.invoke(prompt.format(context=context_str if context_str else "N/A", question=query)).content

# --- HEADER BRANDING ---
st.markdown(f"""
<div class="header-banner">
    <div>
        <div class="header-title">🎙️ AI VOICE ASSISTANT & 3D SPATIAL CORE</div>
        <div style="color: #94A3B8; font-size: 0.85rem;">Voice Recognition | Speech Synthesis | WebGL Spatial Hologram | Multi-Theme Customizer</div>
    </div>
    <div>
        <span style="background: {accent}22; color: {accent}; border: 1px solid {accent}; padding: 0.35rem 0.85rem; border-radius: 20px; font-size: 0.75rem; font-weight: 700;">VOICE SYSTEM ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL: THEMES & COLOR PICKER ---
with st.sidebar:
    st.markdown("### 🎨 Theme & Lighting Customizer")
    
    st.session_state.theme_mode = st.selectbox(
        "Display Mode",
        ["Dark Cyber", "Light Minimal", "Neon Cyberpunk"]
    )
    
    color_choice = st.radio(
        "Accent Lighting Color",
        ["Electric Blue", "Emerald Green", "Neon Purple", "Solar Amber"]
    )
    
    color_map = {
        "Electric Blue": "#38BDF8",
        "Emerald Green": "#34D399",
        "Neon Purple": "#C084FC",
        "Solar Amber": "#FBBF24"
    }
    st.session_state.accent_color = color_map[color_choice]

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("### 🔑 Engine Authentication")
    api_key = st.text_input("OpenAI Key", type="password")
    if st.button("Activate Voice Agent"):
        if api_key:
            st.session_state.engine = VoiceAgentEngine(openai_api_key=api_key)
            st.success("Voice Engine Active")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    st.markdown("### 📚 Knowledge Ingestion")
    uploaded_files = st.file_uploader("Upload Knowledge Specs", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Index Docs"):
        if st.session_state.engine:
            saved_paths = []
            os.makedirs("./temp_docs", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("./temp_docs", f.name)
                with open(path, "wb") as file:
                    file.write(f.getbuffer())
                saved_paths.append(path)
            chunks = st.session_state.engine.index_documents(saved_paths)
            st.success(f"Indexed {chunks} chunks")

# --- MAIN WORKSPACE LAYOUT ---
col_3d, col_voice = st.columns([1, 1.2])

with col_3d:
    st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:{accent}; margin-bottom:0.5rem;'>3D HOLOGRAPHIC CORE</div>", unsafe_allow_html=True)
    
    # DYNAMIC COLOR-ADAPTIVE THREE.JS SPHERE
    hex_color = accent.replace("#", "0x")
    threejs_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style> body {{ margin: 0; overflow: hidden; background: transparent; }} </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(380, 380);
            document.body.appendChild(renderer.domElement);

            const geometry = new THREE.SphereGeometry(2, 48, 48);
            const material = new THREE.MeshStandardMaterial({{ color: {hex_color}, wireframe: true, emissive: {hex_color}, emissiveIntensity: 0.3 }});
            const moon = new THREE.Mesh(geometry, material);
            scene.add(moon);

            const light = new THREE.PointLight({hex_color}, 2, 100);
            light.position.set(10, 10, 10);
            scene.add(light);
            camera.position.z = 5.5;

            function animate() {{
                requestAnimationFrame(animate);
                moon.rotation.y += 0.005;
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    components.html(threejs_html, height=390)

with col_voice:
    st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:{accent}; margin-bottom:0.5rem;'>INTERACTIVE VOICE CONVERSATION CHAT</div>", unsafe_allow_html=True)

    # NATIVE WEB SPEECH VOICE INPUT CONTROL
    speech_rec_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .voice-btn {{
                background-color: {accent};
                color: #000000;
                border: none;
                padding: 0.75rem 1.5rem;
                font-weight: 700;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-family: 'Space Grotesk', sans-serif;
            }}
            .voice-btn:hover {{ opacity: 0.9; }}
        </style>
    </head>
    <body>
        <button class="voice-btn" onclick="startSpeechRecognition()">🎙️ CLICK TO SPEAK TO AI</button>
        <p id="transcript-output" style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.5rem;"></p>

        <script>
            function startSpeechRecognition() {{
                window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!window.SpeechRecognition) {{
                    alert("Speech Recognition not supported in this browser. Please use Chrome.");
                    return;
                }}
                const recognition = new SpeechRecognition();
                recognition.interimResults = false;
                
                document.getElementById("transcript-output").innerText = "Listening... Speak now.";

                recognition.onresult = (e) => {{
                    const text = e.results[0][0].transcript;
                    document.getElementById("transcript-output").innerText = "Recognized: " + text;
                    window.parent.postMessage({{type: "streamlit:setComponentValue", value: text}}, "*");
                }};
                
                recognition.start();
            }}
        </script>
    </body>
    </html>
    """
    voice_input_text = st.text_input("Or type your query below:", key="text_query_field")

    if st.button("Execute Voice/Text Query") and voice_input_text:
        if st.session_state.engine:
            with st.spinner("Processing AI response..."):
                reply = st.session_state.engine.process_voice_query(voice_input_text)
                st.session_state.chat_history.append({"user": voice_input_text, "ai": reply})
                
                # AUTOMATIC TTS READOUT
                clean_reply = reply.replace("'", "\\'").replace("\n", " ")
                tts_script = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance('{clean_reply}');
                    window.speechSynthesis.speak(msg);
                </script>
                """
                components.html(tts_script, height=0)
        else:
            st.error("Please activate the Voice Agent with an API key first.")

    # RENDER CHAT HISTORY LOG
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-bubble-user'><b>👤 User:</b> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'><b>🤖 AI Voice Agent:</b> {chat['ai']}</div>", unsafe_allow_html=True)
