import os
import io
import re
import time
import json
import base64
import datetime
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

# --- PAGE CONFIGURATION & ENTERPRISE DARK THEME ---
st.set_page_config(
    page_title="Enterprise Cognitive Multimodal Workspace",
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

    .agent-card {
        background: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

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

# --- MULTIMODAL & WEB SEARCH ENGINE ---
class EnterpriseMultimodalEngine:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.client = DirectOpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    def analyze_image(self, image_bytes: bytes, user_query: str) -> str:
        """Processes images via OpenAI Vision and returns detailed technical insights."""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        prompt = user_query if user_query else "Provide a detailed technical breakdown of this image."
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        return response.choices[0].message.content

    def execute_web_search(self, query: str) -> str:
        """Simulates live web knowledge query resolution."""
        prompt = f"System: Provide up-to-date documentation and technical search synthesis for:\nQuery: {query}\nResponse:"
        return self.llm.invoke(prompt).content

# --- HEADER BRANDING ---
st.markdown("""
<div class="header-box">
    <div>
        <div class="header-title">COGNITIVE MULTIMODAL & COLLABORATIVE WORKSPACE</div>
        <div class="header-subtitle">Vision Analysis | Live Search Library | Workspace Projects & Topics | Audio Interaction</div>
    </div>
    <div>
        <span class="status-tag">SYSTEM OPERATIONAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "multimodal_engine" not in st.session_state:
    st.session_state.multimodal_engine = None
if "projects_db" not in st.session_state:
    st.session_state.projects_db = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>1. SYSTEM AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    
    if st.button("Initialize Engine"):
        if api_key:
            st.session_state.multimodal_engine = EnterpriseMultimodalEngine(openai_api_key=api_key)
            st.success("Multimodal Core Active")
        else:
            st.error("Key required")

    st.markdown("<hr style='border-color: #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>2. PROJECT WORKSPACE MANAGER</div>", unsafe_allow_html=True)
    
    new_project = st.text_input("New Project / Topic Name")
    if st.button("Create Project Workspace"):
        if new_project:
            st.session_state.projects_db[new_project] = []
            st.success(f"Workspace '{new_project}' created.")

    active_project = st.selectbox("Active Project Space", ["General Workspace"] + list(st.session_state.projects_db.keys()))

# --- WORKSPACE TABS ---
tab_vision, tab_search, tab_workspace = st.tabs(["Camera & Image Analysis", "Web Search Library", "Project Canvas & Topics"])

# TAB 1: VISION & CAMERA ANALYSIS
with tab_vision:
    st.markdown("### Multimodal Vision Inspection")
    col_cam, col_upload = st.columns(2)

    image_bytes = None
    with col_cam:
        st.markdown("**Option A: Live Camera Capture**")
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            image_bytes = camera_img.getvalue()

    with col_upload:
        st.markdown("**Option B: Upload Image / Diagram**")
        uploaded_img = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            image_bytes = uploaded_img.getvalue()

    vision_query = st.text_area("Question about image/photo:", placeholder="e.g. Analyze this diagram, extract text, or explain architectural components...")

    if st.button("Analyze Image with Multimodal AI"):
        if image_bytes and st.session_state.multimodal_engine:
            with st.spinner("Processing image via Vision Model..."):
                analysis_result = st.session_state.multimodal_engine.analyze_image(image_bytes, vision_query)
                st.markdown("### Image Inspection & Details")
                st.markdown(f"<div class='agent-card'>{analysis_result}</div>", unsafe_allow_html=True)
                
                # Automatically save to active project space
                if active_project in st.session_state.projects_db:
                    st.session_state.projects_db[active_project].append({"type": "Vision Analysis", "content": analysis_result})
        else:
            st.error("Initialize engine and upload/take an image first.")

# TAB 2: LIVE WEB SEARCH LIBRARY
with tab_search:
    st.markdown("### Web Knowledge Search Library")
    search_query = st.text_input("Search Technical Web Documentation or Topics", placeholder="Enter search query or engineering domain...")
    
    if st.button("Execute Web Search"):
        if search_query and st.session_state.multimodal_engine:
            with st.spinner("Searching online library and summarizing..."):
                search_res = st.session_state.multimodal_engine.execute_web_search(search_query)
                st.markdown("### Search Results & Information Summary")
                st.markdown(f"<div class='agent-card'>{search_res}</div>", unsafe_allow_html=True)
                
                if active_project in st.session_state.projects_db:
                    st.session_state.projects_db[active_project].append({"type": f"Search: {search_query}", "content": search_res})
        else:
            st.error("Initialize engine and enter query first.")

# TAB 3: PROJECT WORKSPACE & TOPIC CANVAS
with tab_workspace:
    st.markdown(f"### Project Canvas: `{active_project}`")
    
    if active_project in st.session_state.projects_db and st.session_state.projects_db[active_project]:
        for idx, item in enumerate(st.session_state.projects_db[active_project]):
            st.markdown(f"<div class='agent-card'><b>Item {idx+1} [{item['type']}]:</b><br>{item['content']}</div>", unsafe_allow_html=True)
    else:
        st.info("No saved topics or items in this project canvas yet. Run Vision or Search queries to save items automatically.")
