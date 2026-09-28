import os
import io
import re
import time
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
from backend.app.multi_agent import MultiAgentWorkflow

st.set_page_config(page_title="Multi-Agent Governance & Cognitive Hub", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #030712; color: #F9FAFB; }
    .stSidebar { background-color: #0B0F19; border-right: 1px solid #1E293B; }
    .header-box { background: #0B0F19; border: 1px solid #1E293B; border-radius: 8px; padding: 1.25rem 1.75rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }
    .header-title { font-size: 1.5rem; font-weight: 700; color: #F8FAFC; }
    .status-tag { background-color: #0284C7; color: #FFFFFF; padding: 0.3rem 0.75rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .agent-card { background: #0B0F19; border: 1px solid #1E293B; border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }
    .audit-pass { border-left: 4px solid #10B981; }
    .audit-fail { border-left: 4px solid #EF4444; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div>
        <div class="header-title">MULTI-AGENT GOVERNANCE & HUMAN-IN-THE-LOOP HUB</div>
        <div style="font-size:0.825rem; color:#64748B;">Multi-Role Agent Verification | Security Audit Intercepts | Human-in-the-Loop Approvals</div>
    </div>
    <div>
        <span class="status-tag">SYSTEM ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "agent_engine" not in st.session_state:
    st.session_state.agent_engine = None

with st.sidebar:
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#0EA5E9;'>1. SYSTEM AUTHENTICATION</div>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI Access Key", type="password")
    
    if st.button("Initialize Multi-Agent Engine"):
        if api_key:
            st.session_state.agent_engine = MultiAgentWorkflow(openai_api_key=api_key)
            st.success("Multi-Agent Cluster Online")
        else:
            st.error("Key required")

tab_agents, tab_graph = st.tabs(["Multi-Agent Execution & Human Approval", "3D Knowledge Matrix"])

with tab_agents:
    st.markdown("### Multi-Agent Autonomous Query Resolution")
    query = st.text_area("Engineering Task Input", height=90, placeholder="Define architecture request or system configuration requirement...")
    
    if st.button("Run Multi-Agent Pipeline"):
        if query and st.session_state.agent_engine:
            with st.spinner("Agent 1 (Architect) drafting & Agent 2 (Security) auditing..."):
                results = st.session_state.agent_engine.execute_workflow(query, context="K8s NodePort, Redis StatefulSet, PVC Storage")
                st.session_state["pending_approval"] = results
        else:
            st.error("Initialize engine and enter query first.")

    if "pending_approval" in st.session_state:
        res = st.session_state["pending_approval"]
        
        col_draft, col_audit = st.columns(2)
        with col_draft:
            st.markdown("<div class='agent-card'><b>AGENT 1: ARCHITECT PROPOSAL</b></div>", unsafe_allow_html=True)
            st.text_area("Draft Proposal", value=res["draft_response"], height=200, key="draft_area")

        with col_audit:
            audit_class = "audit-pass" if res["security_passed"] else "audit-fail"
            st.markdown(f"<div class='agent-card {audit_class}'><b>AGENT 2: SECURITY & COMPLIANCE AUDIT</b></div>", unsafe_allow_html=True)
            st.write(res["security_notes"])

        st.markdown("<hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
        st.markdown("### Human-in-the-Loop Intervention")
        
        c_app, c_rej = st.columns(2)
        if c_app.button("APPROVE & DEPLOY TO CLUSTER"):
            st.success("Decision Approved by Human Operator. Deployment Triggered.")
            del st.session_state["pending_approval"]
        
        if c_rej.button("REJECT / REQUEST REVISION"):
            st.warning("Decision Rejected by Human Operator. Feedback sent back to Architect Agent.")
            del st.session_state["pending_approval"]

with tab_graph:
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
