import streamlit as st
import requests
import os

st.set_page_config(page_title="Cognitive Intelligence Platform", layout="wide", initial_sidebar_state="expanded")
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stAppViewContainer {
        background-color: #0F172A;
    }
    .stSidebar {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    .stButton>button {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        border: none;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .context-box {
        background-color: #1E293B;
        border-left: 3px solid #38BDF8;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
        color: #CBD5E1;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Custom Navbar Header
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 1.5rem; border-bottom: 1px solid #334155; margin-bottom: 2rem;">
    <div>
        <h1 style="margin:0; font-size: 1.75rem;">Enterprise Cognitive Platform</h1>
        <p style="margin:0; color: #94A3B8; font-size: 0.9rem;">Hybrid Vector Search | Autonomous Agent Framework | Telemetry Audit</p>
    </div>
    <div style="text-align: right;">
        <span style="background-color: #065F46; color: #34D399; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: AI Assistant Console & Configuration
with st.sidebar:
    st.markdown("<h3 style='font-size: 1.1rem;'>Assistant Configuration</h3>", unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter platform access key")
    
    if st.button("Initialize Engine"):
        if api_key:
            res = requests.post(f"{API_BASE_URL}/init", json={"openai_api_key": api_key})
            if res.status_code == 200:
                st.session_state["engine_active"] = True
                st.success("Platform initialized successfully.")
            else:
                st.error("Authentication failed.")
        else:
            st.warning("Key required.")

    st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem;'>Document Ingestion</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Technical Documents", accept_multiple_files=True, type=["pdf", "txt"])
    if uploaded_files and st.button("Index Content"):
        saved_paths = []
        os.makedirs("./temp_docs", exist_ok=True)
        for f in uploaded_files:
            path = os.path.join("./temp_docs", f.name)
            with open(path, "wb") as file:
                file.write(f.getbuffer())
            saved_paths.append(path)
        
        res = requests.post(f"{API_BASE_URL}/index", json={"file_paths": saved_paths})
        if res.status_code == 200:
            st.success(f"Processed {res.json()['indexed_chunks']} document chunks.")

# Primary Operations Workspace
tab_query, tab_analytics = st.tabs(["Execution Console", "Platform Telemetry"])

with tab_query:
    col_input, col_voice = st.columns([2, 1])

    with col_input:
        st.markdown("<h3 style='font-size: 1.2rem;'>Query Input</h3>", unsafe_allow_html=True)
        text_prompt = st.text_area("Enter technical query or architecture question", height=100)

    with col_voice:
        st.markdown("<h3 style='font-size: 1.2rem;'>Voice Interface</h3>", unsafe_allow_html=True)
        audio_file = st.audio_input("Record Voice Prompt")

    active_query = None

    if audio_file:
        with st.spinner("Transcribing audio stream..."):
            files = {"file": ("audio.wav", audio_file.getvalue(), "audio/wav")}
            res = requests.post(f"{API_BASE_URL}/transcribe", files=files)
            if res.status_code == 200:
                active_query = res.json()["transcript"]
                st.info(f"Transcribed Input: {active_query}")

    if text_prompt and not active_query:
        if st.button("Execute Query"):
            active_query = text_prompt

    if active_query:
        with st.spinner("Executing retrieval & inference..."):
            res = requests.post(f"{API_BASE_URL}/query", json={"session_id": "session-prod-01", "query": active_query})
            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]

                st.markdown("<h3 style='font-size: 1.2rem; margin-top: 1.5rem;'>Assistant Output</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #1E293B; border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; font-size: 1rem; color: #E2E8F0; line-height: 1.6;'>{answer}</div>", unsafe_allow_html=True)

                # Text-To-Speech Execution
                tts_script = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance('{answer.replace("'", "")}');
                    window.speechSynthesis.speak(msg);
                </script>
                """
                st.components.v1.html(tts_script, height=0)

                st.markdown("<h4 style='font-size: 1rem; margin-top: 1.5rem; color: #94A3B8;'>Retrieved Context Chunks</h4>", unsafe_allow_html=True)
                for idx, src in enumerate(data["sources"]):
                    st.markdown(f"<div class='context-box'><b>Rank {idx+1}:</b><br>{src}</div>", unsafe_allow_html=True)

with tab_analytics:
    st.markdown("<h3 style='font-size: 1.2rem;'>System Telemetry & Audit Logs</h3>", unsafe_allow_html=True)
    if st.button("Refresh Telemetry Data"):
        res = requests.get(f"{API_BASE_URL}/analytics")
        if res.status_code == 200:
            metrics = res.json()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Queries Executed</div>
                    <div class='metric-value'>{metrics['total_queries_processed']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Cluster Operational Status</div>
                    <div class='metric-value' style='color: #34D399;'>{metrics['platform_status']}</div>
                </div>
                """, unsafe_allow_html=True)
