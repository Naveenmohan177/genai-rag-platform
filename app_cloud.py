import os
import io
import re
import time
import json
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

from backend.data_pipeline import AdvancedDataPipeline

st.set_page_config(page_title="Enterprise Cognitive & Data Hub", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #030712; color: #F9FAFB; }
    .stSidebar { background-color: #0B0F19; border-right: 1px solid #1E293B; }
    .header-box { background: #0B0F19; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem 1.75rem; margin-bottom: 1.5rem; }
    .header-title { font-size: 1.5rem; font-weight: 700; color: #F8FAFC; }
    .metric-card { background: #0B0F19; border: 1px solid #1E293B; border-radius: 6px; padding: 1rem; text-align: center; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #38BDF8; }
    .metric-lbl { font-size: 0.7rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div class="header-title">ENTERPRISE DATA PIPELINE & COGNITIVE HUB</div>
    <div style="font-size:0.825rem; color:#64748B;">Synthetic Fine-Tuning JSONL Generator | Data Sanitization | 3D Knowledge Core</div>
</div>
""", unsafe_allow_html=True)

tab_console, tab_data = st.tabs(["3D Spatial Console", "Data Engineering & Fine-Tuning"])

with tab_console:
    col_graph, col_input = st.columns([1, 1.2])
    with col_graph:
        threejs_html = """
        <!DOCTYPE html><html><head><style>body { margin: 0; overflow: hidden; background: #030712; }</style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head>
        <body><script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(380, 380); document.body.appendChild(renderer.domElement);
            const sphere = new THREE.Mesh(new THREE.IcosahedronGeometry(2, 2), new THREE.MeshStandardMaterial({ color: 0x0EA5E9, wireframe: true }));
            scene.add(sphere);
            const light = new THREE.PointLight(0x0EA5E9, 2, 100); light.position.set(10, 10, 10); scene.add(light);
            camera.position.z = 5.5;
            function animate() { requestAnimationFrame(animate); sphere.rotation.y += 0.003; renderer.render(scene, camera); }
            animate();
        </script></body></html>
        """
        components.html(threejs_html, height=390)

    with col_input:
        st.markdown("**Execution Console**")
        q = st.text_area("Query Console", height=100)
        if st.button("Execute Query"):
            st.info("Query processing initiated.")

with tab_data:
    st.markdown("### Advanced Data Tools & Fine-Tuning Dataset Generator")
    
    uploaded_data_files = st.file_uploader("Upload raw documents to build fine-tuning dataset", accept_multiple_files=True, type=["pdf", "txt"])
    
    if uploaded_data_files and st.button("Run Data Pipeline & Generate JSONL"):
        pipeline = AdvancedDataPipeline()
        saved_paths = []
        os.makedirs("./temp_docs", exist_ok=True)
        for f in uploaded_data_files:
            path = os.path.join("./temp_docs", f.name)
            with open(path, "wb") as file:
                file.write(f.getbuffer())
            saved_paths.append(path)

        processed_json = pipeline.process_raw_documents(saved_paths)
        jsonl_output = pipeline.generate_openai_finetuning_jsonl(processed_json)

        st.success(f"Pipeline executed successfully!")
        
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='metric-card'><div class='metric-lbl'>Processed JSON</div><div class='metric-val'>{processed_json}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-lbl'>Fine-Tuning JSONL</div><div class='metric-val'>{jsonl_output}</div></div>", unsafe_allow_html=True)

        with open(jsonl_output, "r", encoding="utf-8") as f:
            st.download_button("Download Fine-Tuning Dataset (.jsonl)", f.read(), file_name="openai_finetune_data.jsonl", mime="application/json")
