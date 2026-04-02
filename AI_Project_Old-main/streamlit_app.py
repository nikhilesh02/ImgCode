"""Streamlit UI for Agentic RAG System - Singtel ECOM Professional UI"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.document_ingestion.image_processor import ImageProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECOM AI Assistant | Singtel",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Singtel Brand CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Singtel Design Tokens ── */
:root {
    --singtel-red:       #E0001B;
    --singtel-red-dark:  #B5001A;
    --singtel-red-light: #FF2D46;
    --singtel-navy:      #1A1F36;
    --singtel-charcoal:  #2D3148;
    --singtel-slate:     #4A5568;
    --singtel-silver:    #F4F6FA;
    --singtel-white:     #FFFFFF;
    --singtel-border:    #E2E8F0;
    --singtel-muted:     #718096;
    --singtel-success:   #00875A;
    --singtel-dot:       #E0001B;
    --radius-sm:         6px;
    --radius-md:         10px;
    --radius-lg:         16px;
    --shadow-sm:         0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:         0 4px 12px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06);
    --shadow-lg:         0 10px 30px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.08);
}

/* ── Global Reset ── */
* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--singtel-navy);
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main layout ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── TOP HEADER ── */
.singtel-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--singtel-white);
    border-bottom: 3px solid var(--singtel-red);
    padding: 0 2rem;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-md);
}
.singtel-header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.singtel-logo-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-decoration: none;
}
.singtel-wordmark {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--singtel-navy);
    line-height: 1;
}
.singtel-wordmark span {
    color: var(--singtel-red);
}
.singtel-dots {
    display: flex;
    gap: 3px;
    align-items: center;
    margin-bottom: 2px;
}
.singtel-dot-item {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--singtel-red);
}
.header-divider {
    width: 1px;
    height: 28px;
    background: var(--singtel-border);
    margin: 0 0.25rem;
}
.header-app-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--singtel-slate);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.header-badge {
    background: var(--singtel-red);
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-env-badge {
    background: var(--singtel-silver);
    border: 1px solid var(--singtel-border);
    color: var(--singtel-slate);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.header-user {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--singtel-slate);
    font-weight: 500;
}
.user-avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--singtel-red);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--singtel-navy) !important;
    border-right: none !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
}
.sidebar-inner {
    padding: 1.5rem 1.25rem;
}
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.sidebar-status-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}
.sidebar-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
.sidebar-status-row:last-child { margin-bottom: 0; }
.sidebar-status-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.55);
    font-weight: 400;
}
.sidebar-status-value {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.status-dot-green {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #48BB78;
    margin-right: 5px;
    box-shadow: 0 0 6px #48BB78;
}
.status-dot-red {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--singtel-red);
    margin-right: 5px;
}
.search-history-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 3px solid var(--singtel-red);
    border-radius: var(--radius-sm);
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: background 0.15s ease;
}
.search-history-item:hover {
    background: rgba(255,255,255,0.09);
}
.search-history-q {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
    margin-bottom: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.search-history-meta {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.38);
    font-family: 'DM Mono', monospace;
}
.no-history {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.3);
    text-align: center;
    padding: 1.5rem 0;
    font-style: italic;
}

/* ── MAIN CONTENT AREA ── */
.main-content-wrapper {
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    max-width: 900px;
    margin: 0 auto;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 2rem;
}
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--singtel-navy);
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.92rem;
    color: var(--singtel-muted);
    font-weight: 400;
    margin: 0;
}
.breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--singtel-muted);
    margin-bottom: 0.75rem;
}
.breadcrumb-sep { color: var(--singtel-border); }
.breadcrumb-active { color: var(--singtel-red); font-weight: 600; }

/* ── Init status banner ── */
.init-banner {
    background: linear-gradient(135deg, #FFF5F5 0%, #FFF 100%);
    border: 1px solid #FED7D7;
    border-left: 4px solid var(--singtel-red);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.init-banner-icon { font-size: 1.1rem; }
.init-banner-text { font-size: 0.85rem; color: var(--singtel-charcoal); font-weight: 500; }

.success-banner {
    background: linear-gradient(135deg, #F0FFF4 0%, #FFF 100%);
    border: 1px solid #C6F6D5;
    border-left: 4px solid var(--singtel-success);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.success-banner-icon { font-size: 1.1rem; }
.success-banner-text { font-size: 0.85rem; color: #276749; font-weight: 500; }

/* ── Search card ── */
.search-card {
    background: var(--singtel-white);
    border: 1px solid var(--singtel-border);
    border-radius: var(--radius-lg);
    padding: 1.75rem 2rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.5rem;
}
.search-card-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--singtel-slate);
    letter-spacing: 0.4px;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}

/* ── Override Streamlit input ── */
.stTextInput > div > div > input {
    border: 2px solid var(--singtel-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--singtel-navy) !important;
    background: var(--singtel-silver) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    outline: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--singtel-red) !important;
    background: white !important;
    box-shadow: 0 0 0 3px rgba(224,0,27,0.1) !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--singtel-muted) !important;
    font-style: italic;
}
.stTextInput > label {
    display: none !important;
}

/* ── Override Streamlit button ── */
.stButton > button {
    background: var(--singtel-red) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.65rem 1.75rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: background 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(224,0,27,0.3) !important;
}
.stButton > button:hover {
    background: var(--singtel-red-dark) !important;
    box-shadow: 0 4px 12px rgba(224,0,27,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--singtel-white);
    border: 1px solid var(--singtel-border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.5rem;
}
.answer-card-header {
    background: linear-gradient(135deg, var(--singtel-navy) 0%, var(--singtel-charcoal) 100%);
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.answer-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.answer-card-meta {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.45);
    font-family: 'DM Mono', monospace;
}
.answer-card-body {
    padding: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--singtel-charcoal);
}

/* ── Source docs expander ── */
.stExpander {
    border: 1px solid var(--singtel-border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--singtel-white) !important;
    overflow: hidden !important;
}
.stExpander summary {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--singtel-slate) !important;
    padding: 0.75rem 1rem !important;
    background: var(--singtel-silver) !important;
}
.stExpander summary:hover {
    background: #EDF2F7 !important;
}

/* ── Timing chip ── */
.timing-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--singtel-silver);
    border: 1px solid var(--singtel-border);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--singtel-muted);
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin-top: 0.5rem;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: var(--singtel-red) !important;
}

/* ── Override st.success / st.error ── */
.stAlert {
    border-radius: var(--radius-md) !important;
    font-size: 0.88rem !important;
}

/* ── Text area override ── */
.stTextArea > div > div > textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: var(--radius-sm) !important;
    border-color: var(--singtel-border) !important;
    color: var(--singtel-slate) !important;
    background: var(--singtel-silver) !important;
}

/* ── FOOTER ── */
.singtel-footer {
    background: var(--singtel-navy);
    border-top: 3px solid var(--singtel-red);
    padding: 1.5rem 2.5rem;
    margin-top: 3rem;
}
.footer-inner {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.footer-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.footer-wordmark {
    font-size: 1rem;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
}
.footer-wordmark span { color: var(--singtel-red); }
.footer-tagline {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    margin-top: 2px;
}
.footer-links {
    display: flex;
    gap: 1.5rem;
}
.footer-link {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s;
}
.footer-link:hover { color: rgba(255,255,255,0.8); }
.footer-copy {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.25);
    font-family: 'DM Mono', monospace;
}

/* ── Sidebar scrollbar ── */
section[data-testid="stSidebar"] ::-webkit-scrollbar { width: 4px; }
section[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: transparent; }
section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }

</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session_state():
    if 'rag_system'      not in st.session_state: st.session_state.rag_system      = None
    if 'initialized'   not in st.session_state: st.session_state.initialized   = False
    if 'history'       not in st.session_state: st.session_state.history       = []
    if 'last_answer'   not in st.session_state: st.session_state.last_answer   = None
    if 'doc_count'     not in st.session_state: st.session_state.doc_count     = 0
    if 'image_result'  not in st.session_state: st.session_state.image_result  = None
    if 'image_analyzer' not in st.session_state: st.session_state.image_analyzer = None
    if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
    if 'image_analysis' not in st.session_state: st.session_state.image_analysis = None


# ── RAG Init ──────────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    try:
        llm           = Config.get_llm()
        doc_processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        vector_store  = VectorStore()
        urls          = Config.DEFAULT_URLS
        documents     = doc_processor.process_urls(urls)
        vector_store.create_vectorstore(documents)
        graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
        graph_builder.build()
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Initialisation failed: {str(e)}")
        return None, 0


# ── Render Header ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="singtel-header">
        <div class="singtel-header-left">
            <div class="singtel-logo-wrap">
                <div>
                    <div class="singtel-dots">
                        <div class="singtel-dot-item"></div>
                        <div class="singtel-dot-item"></div>
                        <div class="singtel-dot-item"></div>
                        <div class="singtel-dot-item"></div>
                        <div class="singtel-dot-item"></div>
                    </div>
                    <div class="singtel-wordmark">Singtel</div>
                </div>
            </div>
            <div class="header-divider"></div>
            <span class="header-app-name">ECOM</span>
            <span class="header-badge">AI Assistant</span>
        </div>
        <div class="header-right">
            <span class="header-env-badge">Production</span>
            <div class="header-user">
                <div class="user-avatar">EC</div>
                ECOM Team
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Render Sidebar ────────────────────────────────────────────────────────────
def render_sidebar(initialized: bool, doc_count: int, history: list):
    with st.sidebar:
        st.markdown('<div class="sidebar-inner">', unsafe_allow_html=True)

        # System status
        st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
        status_dot  = '<span class="status-dot-green"></span>' if initialized else '<span class="status-dot-red"></span>'
        status_text = "Online" if initialized else "Initialising…"
        chunks_val  = f"{doc_count:,}" if doc_count else "—"
        st.markdown(f"""
        <div class="sidebar-status-card">
            <div class="sidebar-status-row">
                <span class="sidebar-status-label">RAG Engine</span>
                <span class="sidebar-status-value">{status_dot}{status_text}</span>
            </div>
            <div class="sidebar-status-row">
                <span class="sidebar-status-label">Document Chunks</span>
                <span class="sidebar-status-value">{chunks_val}</span>
            </div>
            <div class="sidebar-status-row">
                <span class="sidebar-status-label">Knowledge Base</span>
                <span class="sidebar-status-value">ECOM Docs</span>
            </div>
            <div class="sidebar-status-row">
                <span class="sidebar-status-label">Model</span>
                <span class="sidebar-status-value">Agentic RAG</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recent searches
        st.markdown('<div class="sidebar-section-label">Recent Searches</div>', unsafe_allow_html=True)
        if history:
            for item in reversed(history[-8:]):
                q_short = item['question'][:55] + "…" if len(item['question']) > 55 else item['question']
                st.markdown(f"""
                <div class="search-history-item">
                    <div class="search-history-q">{q_short}</div>
                    <div class="search-history-meta">⏱ {item['time']:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-history">No searches yet</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ── Render Footer ─────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div class="singtel-footer">
        <div class="footer-inner">
            <div class="footer-brand">
                <div>
                    <div class="footer-wordmark">singtel</div>
                    <div class="footer-tagline">Tomorrow starts with Singtel</div>
                </div>
            </div>
            <div class="footer-links">
                <a href="#" class="footer-link">ECOM Portal</a>
                <a href="#" class="footer-link">Documentation</a>
                <a href="#" class="footer-link">Support</a>
                <a href="#" class="footer-link">Privacy Policy</a>
            </div>
            <div class="footer-copy">© 2025 Singtel Group. Internal Use Only.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Header
    render_header()

    # Sidebar
    render_sidebar(
        st.session_state.initialized,
        st.session_state.doc_count,
        st.session_state.history,
    )

    # ── Content wrapper
    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)

    # Breadcrumb + page title
    st.markdown("""
    <div class="page-header">
        <div class="breadcrumb">
            <span>ECOM</span>
            <span class="breadcrumb-sep">›</span>
            <span>Tools</span>
            <span class="breadcrumb-sep">›</span>
            <span class="breadcrumb-active">AI Document Assistant</span>
        </div>
        <h1 class="page-title">ECOM AI Document Assistant</h1>
        <p class="page-subtitle">Ask questions about Singtel ECOM documentation (PDFs & Images). Powered by Agentic RAG.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Init RAG
    if not st.session_state.initialized:
        st.markdown("""
        <div class="init-banner">
            <span class="init-banner-icon">⚙️</span>
            <span class="init-banner-text">Initialising knowledge base — loading and indexing ECOM documents…</span>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Loading RAG engine…"):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system  = rag_system
                st.session_state.initialized = True
                st.session_state.doc_count   = num_chunks
                st.rerun()
    else:
        st.markdown(f"""
        <div class="success-banner">
            <span class="success-banner-icon">✅</span>
            <span class="success-banner-text">
                Knowledge base ready &mdash; <strong>{st.session_state.doc_count:,}</strong> document chunks indexed and available for search.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Question Input Section ─────────────────────────────────────────────
    st.markdown('<div class="search-card-label">Ask a Question</div>', unsafe_allow_html=True)
    
    with st.form("search_form", clear_on_submit=False):
        question = st.text_input(
            label="question",
            placeholder="Ask about documents or images in the data folder...",
        )
        col_btn, col_tip = st.columns([1, 4])
        with col_btn:
            submit = st.form_submit_button("Search")
        with col_tip:
            st.markdown(
                '<p style="font-size:0.78rem;color:#718096;margin-top:0.55rem;font-style:italic;">'
                'Search across all PDFs and images in the data folder.</p>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Image Upload Section ─────────────────────────────────────────────────────
    st.markdown('<div class="search-card-label">Upload Error/Screenshot Image</div>', unsafe_allow_html=True)
    
    with st.form("image_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image (screenshot, error message, etc.)",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'],
                help="Upload an image containing error messages, screenshots, or any visual content to analyze"
            )
        with col2:
            analyze_btn = st.form_submit_button("🔍 Analyze")
        
        st.markdown(
            '<p style="font-size:0.78rem;color:#718096;margin-top:0.25rem;font-style:italic;">'
            'Upload an image to analyze its content and find relevant information from documents.</p>',
            unsafe_allow_html=True,
        )
    
    # ── Handle Image Upload and Analysis ──────────────────────────────────
    if uploaded_file is not None:
        # Display the uploaded image
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Initialize image processor if not already done
        if st.session_state.image_analyzer is None:
            st.session_state.image_analyzer = ImageProcessor()
        
        # Analyze the uploaded image
        if analyze_btn or st.session_state.image_analysis is None:
            with st.spinner("Analyzing image content..."):
                try:
                    analysis = st.session_state.image_analyzer.process_uploaded_image(uploaded_file)
                    st.session_state.image_analysis = analysis
                    st.session_state.uploaded_image = uploaded_file.name
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                    st.session_state.image_analysis = None
    
    # ── Display Image Analysis and Search Results ────────────────────────
    if st.session_state.image_analysis:
        st.markdown("""
        <div class="answer-card">
            <div class="answer-card-header">
                <span class="answer-card-title">Image Analysis</span>
                <span class="answer-card-meta">AI Vision</span>
            </div>
            <div class="answer-card-body">{}</div>
        </div>
        """.format(st.session_state.image_analysis), unsafe_allow_html=True)
        
        # Now search for related documents based on image analysis
        if st.session_state.rag_system:
            with st.spinner("Finding relevant documentation..."):
                search_query = st.session_state.image_analysis
                result = st.session_state.rag_system.run(search_query)
                
                st.session_state.history.append({
                    'question': f"[Image] {st.session_state.uploaded_image}",
                    'answer': result['answer'],
                    'time': result.get('time', 0),
                })
                st.session_state.last_answer = {
                    'question': search_query,
                    'answer': result['answer'],
                    'docs': result.get('retrieved_docs', []),
                    'time': result.get('time', 0),
                    'is_image_search': True,
                }

    # ── Handle search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching knowledge base (PDFs & Images)…"):
                start_time = time.time()
                result     = st.session_state.rag_system.run(question)
                elapsed    = time.time() - start_time

            st.session_state.history.append({
                'question': question,
                'answer':   result['answer'],
                'time':     elapsed,
            })
            st.session_state.last_answer = {
                'question': question,
                'answer':   result['answer'],
                'docs':     result.get('retrieved_docs', []),
                'time':     elapsed,
            }

    # ── Display last answer
    if st.session_state.last_answer:
        ans = st.session_state.last_answer
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-card-header">
                <span class="answer-card-title">AI Response</span>
                <span class="answer-card-meta">⏱ {ans['time']:.2f}s</span>
            </div>
            <div class="answer-card-body">{ans['answer']}</div>
        </div>
        """, unsafe_allow_html=True)

        if ans.get('docs'):
            with st.expander(f"📄 View Source Documents ({len(ans['docs'])} retrieved)"):
                for i, doc in enumerate(ans['docs'], 1):
                    # Show source type in the label
                    source_type = doc.metadata.get('source_type', 'document')
                    source_file = doc.metadata.get('source', 'Unknown')
                    
                    st.text_area(
                        f"Source {i} ({source_type}: {Path(source_file).name})",
                        doc.page_content[:500] + "…" if len(doc.page_content) > 500 else doc.page_content,
                        height=120,
                        disabled=True,
                        key=f"doc_{i}_{id(ans)}",
                    )

    st.markdown('</div>', unsafe_allow_html=True)  # close main-content-wrapper

    # Footer
    render_footer()


if __name__ == "__main__":
    main()