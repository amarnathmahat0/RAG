"""
NexusRAG — Fixed Premium White Streamlit Frontend
Fixes applied:
  1. Sidebar hidden / toggle invisible → Sidebar is now stats-only; upload+docs moved to main area
  2. Upload section inaccessible → Moved to always-visible collapsible top panel in main area
  3. Document scope selector broken → Moved to main area top panel, always accessible
  4. White theme inconsistency → Targeted CSS overrides for every Streamlit internal component
  5. Sidebar min-width conflict → Removed conflicting min-width; sidebar is slim stats-only panel
  6. File uploader label visibility → label_visibility="collapsed" removed; proper label used
  7. Session state lost on rerun → file references stored in session_state before rerun
  8. SSE cursor blink not applying → Blink animation moved to <style> block, not inline markdown

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import time
import re
from pathlib import Path

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"
SUPPORTED_TYPES = ["pdf", "txt", "md", "docx", "xlsx", "png", "jpg", "jpeg"]

TIER_META = {
    "TIER_1": {"label": "⚡ TIER 1", "color": "#059669", "bg": "#ECFDF5", "border": "#A7F3D0"},
    "TIER_2": {"label": "🔄 TIER 2", "color": "#D97706", "bg": "#FFFBEB", "border": "#FDE68A"},
    "TIER_3": {"label": "🕸 TIER 3",  "color": "#7C3AED", "bg": "#F5F3FF", "border": "#DDD6FE"},
}

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusRAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: all fixes consolidated ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0.5rem 1.25rem 0 !important; max-width: 100% !important; }
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Kill random gaps between st.markdown elements ────────────────────────── */
/* Every st.markdown gets wrapped in .element-container with margin — zero it  */
.stMarkdown { margin-bottom: 0 !important; }
.element-container { margin-bottom: 0 !important; padding-bottom: 0 !important; }
/* Vertical rhythm for non-chat sections only */
[data-testid="stSidebar"] .element-container { margin-bottom: 4px !important; }
div[data-testid="stExpander"] .element-container { margin-bottom: 4px !important; }

/* ── Chat area: pin input at bottom, messages scroll ──────────────────────── */
/* The tab panel gets a fixed height so the chat input NEVER jumps down.       */
/* Messages live in .chat-scroll which overflows independently.                */
[data-testid="stTabsContent"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stTabsContent"] > div:first-child {
    display: flex !important;
    flex-direction: column !important;
    height: calc(100vh - 230px) !important;
    min-height: 380px !important;
    overflow: hidden !important;
}
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem 0;
    scrollbar-width: thin;
    scrollbar-color: #E5E7EB transparent;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #E5E7EB; border-radius: 2px; }
/* Pin chat input to bottom of tab panel */
.stChatInputContainer {
    flex-shrink: 0 !important;
    background: #FFFFFF !important;
    border-top: 1px solid #F3F4F6 !important;
    padding: 0.5rem 0 0.25rem !important;
    z-index: 10 !important;
}

/* ── FIX 1 + 5: Sidebar — stats-only, slim, always visible ───────────────── */
/* Remove min-width fight; let Streamlit manage width naturally */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    height: 100vh !important;
    padding-bottom: 2rem !important;
}
[data-testid="stSidebar"] * { color: #111827 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #6B7280 !important; font-size: 0.82rem; }

/* FIX 1: Make sidebar toggle arrow visible against white ─────────────────── */
/* The collapse button sits outside the sidebar; force it visible */
[data-testid="collapsedControl"] {
    background: #F3F4F6 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 0 6px 6px 0 !important;
    color: #374151 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #374151 !important;
    stroke: #374151 !important;
}
button[kind="header"] {
    background: #F3F4F6 !important;
    color: #374151 !important;
}

/* ── FIX 4: Streamlit internal component white-theme overrides ─────────────── */

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px !important;
    border-bottom: 1px solid #E5E7EB !important;
}
.stTabs [data-baseweb="tab"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #6B7280 !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #F5F3FF !important;
    border-color: #C7D2FE !important;
    border-bottom-color: #F5F3FF !important;
    color: #6366F1 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF !important;
    padding: 1rem 0 !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    background: #FAFAFA !important;
    border-color: #E5E7EB !important;
}
[data-baseweb="tag"] {
    background: #EEF2FF !important;
    color: #6366F1 !important;
    border: 1px solid #C7D2FE !important;
}
[data-baseweb="tag"] span { color: #6366F1 !important; }

/* Expander */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 9px !important;
}
div[data-testid="stExpander"] summary {
    background: #FAFAFA !important;
    color: #374151 !important;
}

/* Chat input */
.stChatInputContainer {
    border-top: 1px solid #F3F4F6 !important;
    background: #FFFFFF !important;
    padding: 0.5rem 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #FAFAFA !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 14px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 10px 16px !important;
    resize: none !important;
    transition: border-color .15s, box-shadow .15s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
    background: #FFFFFF !important;
    outline: none !important;
}
/* Send button */
[data-testid="stChatInput"] button {
    background: #6366F1 !important;
    border-radius: 10px !important;
    border: none !important;
    color: #FFFFFF !important;
    transition: background .15s !important;
}
[data-testid="stChatInput"] button:hover {
    background: #4F46E5 !important;
}

/* File uploader — FIX 6: always visible label + styled zone */
[data-testid="stFileUploader"] {
    background: #F5F3FF !important;
    border: 1.5px dashed #C7D2FE !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: #6366F1 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] button {
    background: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Selectbox / dropdowns */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #FAFAFA !important;
    border-color: #E5E7EB !important;
}

/* Progress bar */
.stProgress > div > div { background: #6366F1 !important; }

/* Metric */
[data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.75rem !important; }

/* Info / success / error boxes */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #374151 !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .15s !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    border-color: #6366F1 !important;
    color: #6366F1 !important;
    background: #F5F3FF !important;
}

/* ── Typography helpers ────────────────────────────────────────────────────── */
.sec {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9CA3AF !important;
    margin: 1.2rem 0 0.5rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.logo { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.04em; color: #111827; line-height: 1; }
.logo em { color: #6366F1; font-style: normal; }
.logo-sub { font-size: 0.65rem; color: #9CA3AF; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em; margin-top: 3px; }

/* ── Status dots ───────────────────────────────────────────────────────────── */
.sp { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 0.8rem; }
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dok { background: #10B981; box-shadow: 0 0 5px #10B98155; }
.derr { background: #EF4444; }
.dunk { background: #D1D5DB; }
.sn { flex: 1; color: #374151 !important; font-weight: 500; }
.sv { font-size: 0.68rem; color: #9CA3AF !important; font-family: 'JetBrains Mono', monospace; }

/* ── RAM bar ───────────────────────────────────────────────────────────────── */
.ram-bg   { background: #F3F4F6; border-radius: 4px; height: 5px; margin-top: 4px; }
.ram-fill { height: 5px; border-radius: 4px; transition: width .4s; }

/* ── Top panel (moved from sidebar) ───────────────────────────────────────── */
.top-panel {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.top-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.tp-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #374151;
    letter-spacing: -0.01em;
}
.tp-badge {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    color: #6366F1;
    font-weight: 600;
}

/* ── File rows ─────────────────────────────────────────────────────────────── */
.frow {
    display: flex; align-items: center; gap: 9px;
    padding: 8px 10px; border-radius: 8px;
    background: #FAFAFA; border: 1px solid #E5E7EB;
    margin-bottom: 5px;
}
.frow .fi  { font-size: 14px; flex-shrink: 0; }
.frow .fn  { flex: 1; font-size: 0.78rem; color: #374151 !important; font-weight: 500;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.frow .fch { font-size: 0.65rem; color: #9CA3AF !important;
             font-family: 'JetBrains Mono', monospace; white-space: nowrap; }

/* ── Scope cards ───────────────────────────────────────────────────────────── */
.scope-card {
    background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px;
    padding: 7px 11px; font-size: 0.75rem; color: #059669 !important;
    font-family: 'JetBrains Mono', monospace; margin-top: 6px;
}
.scope-card-all {
    background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 7px 11px; font-size: 0.75rem; color: #6B7280 !important;
    font-family: 'JetBrains Mono', monospace; margin-top: 6px;
}

/* ── Top bar ───────────────────────────────────────────────────────────────── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.3rem 0 0.9rem; border-bottom: 1px solid #F3F4F6;
    margin-bottom: 1rem;
}
.toplogo { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.03em; color: #111827; }
.toplogo em { color: #6366F1; font-style: normal; }
.pill {
    font-size: 0.68rem; font-family: 'JetBrains Mono', monospace;
    padding: 4px 11px; border-radius: 20px; font-weight: 500;
}
.pill-indigo { color: #6366F1; background: #EEF2FF; border: 1px solid #C7D2FE; }
.pill-green  { color: #059669; background: #ECFDF5; border: 1px solid #A7F3D0; }

/* ── Chat bubbles ──────────────────────────────────────────────────────────── */
.msg-u { display: flex; justify-content: flex-end; margin-bottom: 8px; }
.msg-b { display: flex; justify-content: flex-start; margin-bottom: 8px; }
.bub-u {
    background: #6366F1; color: #FFFFFF;
    padding: 10px 16px; border-radius: 18px 18px 4px 18px;
    max-width: 72%; font-size: 0.88rem; line-height: 1.65;
    box-shadow: 0 2px 8px rgba(99,102,241,.28);
}
.bub-b {
    background: #FFFFFF; color: #111827;
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    max-width: 80%; font-size: 0.88rem; line-height: 1.7;
    border: 1px solid #E5E7EB; box-shadow: 0 2px 8px rgba(0,0,0,.06);
}

/* ── Agent steps ───────────────────────────────────────────────────────────── */
.steps { display: flex; flex-direction: column; gap: 3px; margin: 6px 0 10px; }
.step {
    display: flex; align-items: flex-start; gap: 9px;
    padding: 7px 12px; border-radius: 8px;
    font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;
    transition: all .2s;
}
.step-done    { background: #F9FAFB; border: 1px solid #E5E7EB; color: #9CA3AF; }
.step-active  { background: #F5F3FF; border: 1px solid #C7D2FE; color: #6366F1; }
.step-pending { background: #FAFAFA; border: 1px solid #F3F4F6; color: #D1D5DB; }
.step .si { flex-shrink: 0; }
.step .st { flex: 1; line-height: 1.5; }

/* ── FIX 8: Streaming cursor — defined here in page CSS, NOT inline markdown ── */
/* This ensures @keyframes is in the document <style> scope and always applies  */
.cursor {
    display: inline-block;
    width: 8px;
    height: 14px;
    background: #6366F1;
    margin-left: 2px;
    vertical-align: middle;
    border-radius: 1px;
    animation: nexus-blink 0.8s step-end infinite;
}
@keyframes nexus-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
.mrow { display: flex; gap: 7px; flex-wrap: wrap; margin: 10px 0 7px; }
.mc {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 9px 12px; min-width: 80px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.mc .mv { font-size: 1.3rem; font-weight: 700; line-height: 1.1;
          font-family: 'JetBrains Mono', monospace; }
.mc .ml { font-size: 0.62rem; color: #9CA3AF; text-transform: uppercase;
          letter-spacing: 0.09em; margin-top: 3px; }
.mg { color: #059669; } .my { color: #D97706; } .mr { color: #DC2626; }
.mb { color: #6366F1; } .mp { color: #7C3AED; }

/* ── Sources box ───────────────────────────────────────────────────────────── */
.srcbox {
    background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 10px 14px; margin-top: 9px;
}
.srcttl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: .1em;
          color: #9CA3AF; font-family: 'JetBrains Mono', monospace; margin-bottom: 7px; }
.srcrow { display: flex; align-items: center; gap: 9px; padding: 3px 0;
          font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
.srci  { color: #9CA3AF; min-width: 22px; }
.srcn  { flex: 1; color: #374151; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.srcbb { width: 55px; height: 4px; background: #E5E7EB; border-radius: 2px; }
.srcbf { height: 4px; border-radius: 2px; }
.srcs  { min-width: 44px; text-align: right; font-size: 0.68rem; }

/* ── Empty state ───────────────────────────────────────────────────────────── */
.empty { text-align: center; padding: 5rem 2rem; }
.empty .ei { font-size: 3rem; opacity: .2; margin-bottom: 14px; }
.empty .et { font-size: 1rem; font-weight: 700; color: #6B7280; margin-bottom: 5px; }
.empty .es { font-size: 0.8rem; color: #9CA3AF; }

/* ── Eval cards ────────────────────────────────────────────────────────────── */
.ecard {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.ecard .ev { font-size: 2.2rem; font-weight: 800; line-height: 1;
             font-family: 'JetBrains Mono', monospace; }
.ecard .el { font-size: 0.65rem; color: #9CA3AF; text-transform: uppercase;
             letter-spacing: .1em; margin-top: 5px; }
.ecard .et2 { font-size: 0.72rem; margin-top: 5px;
              font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* ── Sparkline ─────────────────────────────────────────────────────────────── */
.spark { display: flex; align-items: flex-end; gap: 3px; height: 44px; }
.spk   { flex: 1; border-radius: 3px 3px 0 0; min-height: 4px; }

/* ── Table ─────────────────────────────────────────────────────────────────── */
.htbl { width: 100%; border-collapse: collapse;
        font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; }
.htbl th { text-align: left; padding: 7px 10px; font-size: 0.62rem; color: #9CA3AF;
           text-transform: uppercase; letter-spacing: .1em;
           border-bottom: 1px solid #F3F4F6; }
.htbl td { padding: 7px 10px; border-bottom: 1px solid #F9FAFB; color: #6B7280; }
.htbl tr:last-child td { border-bottom: none; }

/* ── Bar chart ─────────────────────────────────────────────────────────────── */
.bar-wrap { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px 18px; }
.barrow { display: flex; align-items: center; gap: 10px; font-size: 0.75rem;
          font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; }
.barl { color: #9CA3AF; min-width: 100px; text-align: right; font-size: 0.7rem; }
.bart { flex: 1; height: 9px; background: #F3F4F6; border-radius: 4px; overflow: hidden; }
.barf { height: 9px; border-radius: 4px; transition: width .5s; }
.barv { min-width: 46px; text-align: right; font-size: 0.7rem; font-weight: 600; }

/* ── Doc list item ─────────────────────────────────────────────────────────── */
.docrow {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 10px; border-radius: 8px;
    background: #FAFAFA; border: 1px solid #E5E7EB;
    margin-bottom: 5px; transition: background .15s, border-color .15s;
}
.docrow.sel { background: #F5F3FF; border-color: #C7D2FE; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def _init() -> None:
    defaults = {
        "messages":       [],
        "health":         None,
        "avail_docs":     [],
        "selected_docs":  [],
        # FIX 7: store uploaded file info before rerun so it survives
        "pending_files":  [],
        "ingest_jobs":    {},
        "queries":        0,
        "latencies":      [],
        "tier_counts":    {"TIER_1": 0, "TIER_2": 0, "TIER_3": 0},
        "ragas_history":  [],
        "panel_open":     True,   # top panel collapsed state
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── API helpers ───────────────────────────────────────────────────────────────
def api_health() -> dict:
    try:
        return httpx.get(f"{API_BASE}/health", timeout=4).json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

def api_list_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE}/documents", timeout=8)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []

def api_ingest_path(path: str) -> dict:
    try:
        r = httpx.post(f"{API_BASE}/ingest", json={"sources": [path]}, timeout=30)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}

def api_ingest_status(job_id: str) -> dict:
    try:
        return httpx.get(f"{API_BASE}/ingest/status/{job_id}", timeout=8).json()
    except Exception as exc:
        return {"error": str(exc)}

def api_upload_files(files) -> dict:
    try:
        file_tuples = [
            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
            for f in files
        ]
        r = httpx.post(f"{API_BASE}/ingest/upload", files=file_tuples, timeout=120)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}

def save_upload_locally(name: str, data: bytes) -> str:
    """FIX 7: accepts raw bytes so we don't depend on the UploadedFile object surviving rerun."""
    d = Path("data/raw")
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(data)
    return str(p)

# ── SSE stream ────────────────────────────────────────────────────────────────
def stream_query(query: str, documents: list[str] | None):
    """SSE generator.

    FIX — GeneratorExit:
      Yielding inside a `finally` block raises RuntimeError when the caller
      breaks early (Python closes the generator via GeneratorExit and you
      cannot yield during that).  Solution: track whether we already emitted
      "done" with a flag and only emit it in the normal code path.
    """
    payload = json.dumps({
        "query": query, "stream": True,
        "documents": documents if documents else None,
    })
    done_sent = False
    try:
        with httpx.stream(
            "POST", f"{API_BASE}/query",
            content=payload,
            headers={"Content-Type": "application/json"},
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    raw = line[6:]
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        yield {"type": "token", "text": raw}
        # Normal completion — emit done here, NOT in finally
        yield {"type": "done"}
        done_sent = True
    except Exception as exc:
        yield {"type": "error", "text": str(exc)}
        if not done_sent:
            yield {"type": "done"}

# ── Render helpers ────────────────────────────────────────────────────────────
def score_cls(v: float, invert: bool = False) -> str:
    if invert:
        return "mg" if v <= 0.2 else ("my" if v <= 0.4 else "mr")
    return "mg" if v >= 0.7 else ("my" if v >= 0.4 else "mr")

def pct(v) -> str:
    return f"{float(v)*100:.0f}%" if v is not None else "—"

def metrics_html(meta: dict) -> str:
    def mc(val, label, cls):
        return (f'<div class="mc"><div class="mv {cls}">{val}</div>'
                f'<div class="ml">{label}</div></div>')
    cards = []
    if (v := meta.get("faithfulness_score")) is not None:
        cards.append(mc(pct(v), "Faithfulness", score_cls(v)))
    if (v := meta.get("answer_relevancy")) is not None:
        cards.append(mc(pct(v), "Relevancy", score_cls(v)))
    if (v := meta.get("context_precision")) is not None:
        cards.append(mc(pct(v), "Precision", score_cls(v)))
    if (v := meta.get("hallucination_score")) is not None:
        cards.append(mc(pct(v), "Hallucination", score_cls(v, invert=True)))
    ms = meta.get("latency_ms", 0)
    cards.append(mc(f"{ms:.0f}ms", "Latency", "mb"))
    tier = meta.get("tier", "TIER_1")
    t = TIER_META.get(tier, TIER_META["TIER_1"])
    cards.append(
        f'<div class="mc">'
        f'<div class="mv" style="font-size:0.72rem;font-weight:700;color:{t["color"]}">{t["label"]}</div>'
        f'<div class="ml">Tier</div></div>'
    )
    return f'<div class="mrow">{"".join(cards)}</div>'

def sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    seen: dict[str, float] = {}
    for s in sources:
        src = str(s.get("source", "?"))
        sc  = float(s.get("score", 0))
        if src not in seen or sc > seen[src]:
            seen[src] = sc
    unique = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    rows = ""
    for i, (src, sc) in enumerate(unique):
        c = "#059669" if sc >= 0.7 else ("#D97706" if sc >= 0.4 else "#7C3AED")
        w = min(sc * 100, 100) if sc > 0.01 else 100
        lbl = f"{sc:.3f}" if sc > 0.001 else "reranked"
        rows += (
            f'<div class="srcrow">'
            f'<span class="srci">[{i+1}]</span>'
            f'<span class="srcn" title="{src}">{src.split("/")[-1]}</span>'
            f'<div class="srcbb"><div class="srcbf" style="width:{w:.0f}%;background:{c}"></div></div>'
            f'<span class="srcs" style="color:{c}">{lbl}</span>'
            f'</div>'
        )
    n = len(unique)
    return (
        f'<div class="srcbox"><div class="srcttl">📚 Sources — {n} document{"s" if n!=1 else ""}</div>'
        f'{rows}</div>'
    )

def render_message(msg: dict) -> None:
    role    = msg["role"]
    content = msg.get("content", "")
    meta    = msg.get("meta", {})
    steps   = msg.get("steps", [])

    if role == "user":
        st.markdown(
            f'<div class="msg-u" style="margin:6px 0 4px"><div class="bub-u">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Wrap the whole assistant turn in one block so no gap appears between
    # steps → bubble → metrics → sources
    parts: list[str] = []

    if steps:
        step_html = "".join(
            f'<div class="step step-done"><span class="si">✓</span><span class="st">{s}</span></div>'
            for s in steps
        )
        parts.append(f'<div class="steps" style="margin-bottom:6px">{step_html}</div>')

    fmt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    fmt = fmt.replace("\n", "<br>")
    parts.append(f'<div class="msg-b" style="margin-bottom:6px"><div class="bub-b">{fmt}</div></div>')

    if meta:
        parts.append(metrics_html(meta))
        src = sources_html(meta.get("sources", []))
        if src:
            parts.append(src)

    st.markdown(
        f'<div style="margin:4px 0 12px">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )

    if meta:
        with st.expander("📋 Copy raw answer", expanded=False):
            st.code(content, language="")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — stats + health only (FIX 1, 2, 3, 5: no upload/docs here)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 0.75rem">'
        '<div class="logo">⬡ Nexus<em>RAG</em></div>'
        '<div class="logo-sub">MULTI-AGENT · 3-TIER ROUTING</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec">System Status</p>', unsafe_allow_html=True)

    col_ref, _ = st.columns([1, 2])
    with col_ref:
        if st.button("↻ Refresh", key="btn_health"):
            st.session_state.health = api_health()

    if st.session_state.health is None:
        st.session_state.health = api_health()

    h       = st.session_state.health
    overall = h.get("status", "unknown")
    comps   = h.get("components", {})

    def _pill(label, status):
        cls = "dok" if status == "ok" else ("derr" if status == "error" else "dunk")
        val = "online" if status == "ok" else ("error" if status == "error" else "—")
        st.markdown(
            f'<div class="sp"><div class="dot {cls}"></div>'
            f'<span class="sn">{label}</span>'
            f'<span class="sv">{val}</span></div>',
            unsafe_allow_html=True,
        )

    _pill("API Server", overall)
    for name, info in comps.items():
        if name == "models":
            continue
        _pill(name.capitalize(), info.get("status", "unknown"))

    model_info = comps.get("models", {})
    free_ram   = model_info.get("free_ram_mb", 0)
    total_ram  = model_info.get("total_ram_mb", 1) or 1
    used_pct   = min((total_ram - free_ram) / total_ram, 1.0)
    ram_col    = "#EF4444" if free_ram < 512 else "#6366F1"

    st.markdown(
        f'<div style="font-size:0.75rem;margin-top:6px;display:flex;justify-content:space-between">'
        f'<span style="color:#6B7280;font-family:JetBrains Mono,monospace;font-size:0.7rem">RAM</span>'
        f'<span style="color:#9CA3AF;font-size:0.7rem;font-family:JetBrains Mono,monospace">{free_ram:.0f} MB free</span>'
        f'</div>'
        f'<div class="ram-bg"><div class="ram-fill" style="width:{used_pct*100:.0f}%;background:{ram_col}"></div></div>',
        unsafe_allow_html=True,
    )
    loaded = model_info.get("loaded_models", [])
    if loaded:
        m = loaded[0]["name"]
        st.markdown(
            f'<div style="font-size:0.7rem;color:#6B7280;margin-top:5px;'
            f'font-family:JetBrains Mono,monospace">active: <code style="color:#6366F1">{m}</code></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Session stats ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec">Session Stats</p>', unsafe_allow_html=True)

    tc   = st.session_state.tier_counts
    lats = st.session_state.latencies
    avg_ms = int(sum(lats) / len(lats)) if lats else 0

    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.queries)
    c2.metric("Avg ms",  avg_ms)

    st.markdown(
        f'<div style="display:flex;gap:6px;margin-top:6px">'
        f'<div style="flex:1;background:#ECFDF5;border:1px solid #A7F3D0;border-radius:7px;padding:7px;text-align:center">'
        f'<div style="font-size:1.1rem;font-weight:700;color:#059669;font-family:JetBrains Mono,monospace">{tc["TIER_1"]}</div>'
        f'<div style="font-size:0.6rem;color:#6B7280;text-transform:uppercase;letter-spacing:.08em">T1</div></div>'
        f'<div style="flex:1;background:#FFFBEB;border:1px solid #FDE68A;border-radius:7px;padding:7px;text-align:center">'
        f'<div style="font-size:1.1rem;font-weight:700;color:#D97706;font-family:JetBrains Mono,monospace">{tc["TIER_2"]}</div>'
        f'<div style="font-size:0.6rem;color:#6B7280;text-transform:uppercase;letter-spacing:.08em">T2</div></div>'
        f'<div style="flex:1;background:#F5F3FF;border:1px solid #DDD6FE;border-radius:7px;padding:7px;text-align:center">'
        f'<div style="font-size:1.1rem;font-weight:700;color:#7C3AED;font-family:JetBrains Mono,monospace">{tc["TIER_3"]}</div>'
        f'<div style="font-size:0.6rem;color:#6B7280;text-transform:uppercase;letter-spacing:.08em">T3</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True, key="btn_clear"):
        st.session_state.messages      = []
        st.session_state.ragas_history = []
        st.session_state.queries       = 0
        st.session_state.latencies     = []
        st.session_state.tier_counts   = {"TIER_1": 0, "TIER_2": 0, "TIER_3": 0}
        st.rerun()

    st.markdown('<p class="sec">Quick Examples</p>', unsafe_allow_html=True)
    examples = [
        "What experience does the author have?",
        "Summarise the key skills mentioned",
        "What projects are described?",
        "Compare dense vs sparse retrieval",
    ]
    for ex in examples:
        lbl = ex[:40] + ("…" if len(ex) > 40 else "")
        if st.button(lbl, key=f"ex_{ex[:10]}", use_container_width=True):
            st.session_state["_pending"] = ex
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
loaded_name = ""
if st.session_state.health:
    mi = st.session_state.health.get("components", {}).get("models", {})
    lm = mi.get("loaded_models", [])
    if lm:
        loaded_name = lm[0].get("name", "")

sel_docs = st.session_state.selected_docs
scope_badge = (
    f'<span class="pill pill-green">🎯 {len(sel_docs)} doc{"s" if len(sel_docs)!=1 else ""} scoped</span>'
    if sel_docs else
    '<span class="pill pill-indigo">⬡ all documents</span>'
)
model_badge = f'<span class="pill pill-indigo">🤖 {loaded_name}</span>' if loaded_name else ""

st.markdown(
    f'<div class="topbar">'
    f'<div class="toplogo">⬡ Nexus<em>RAG</em></div>'
    f'<div style="display:flex;gap:8px;align-items:center">{scope_badge}{model_badge}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 + 3: Upload & Document Selector moved here — always visible, collapsible
# ─────────────────────────────────────────────────────────────────────────────

# Poll pending jobs before rendering panel (FIX 7: do this before any rerun)
refreshed = False
for jid, status in list(st.session_state.ingest_jobs.items()):
    if status == "pending":
        r = api_ingest_status(jid)
        if r.get("status") == "complete":
            st.session_state.ingest_jobs[jid] = "complete"
            if not refreshed:
                st.session_state.avail_docs = api_list_documents()
                refreshed = True

# Collapsible panel using native expander so it always renders in main area
with st.expander("📂 Documents & Upload", expanded=st.session_state.panel_open):
    # Track open state so it remembers between reruns
    st.session_state.panel_open = True  # expander handles its own state visually

    up_col, doc_col = st.columns([1, 1], gap="large")

    # ── Left: Upload ──────────────────────────────────────────────────────────
    with up_col:
        st.markdown(
            '<p class="sec" style="margin-top:0">Upload Documents</p>',
            unsafe_allow_html=True,
        )

        # FIX 6: Use a real, visible label — do NOT collapse it
        uploaded = st.file_uploader(
            "Select files to ingest (PDF · DOCX · XLSX · TXT · MD · PNG · JPG)",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#6366F1;font-weight:600;margin:6px 0 8px;'
                f'font-family:JetBrains Mono,monospace">{len(uploaded)} file(s) ready</div>',
                unsafe_allow_html=True,
            )
            for f in uploaded[:5]:
                ext  = f.name.split(".")[-1].lower()
                icon = {"pdf":"📄","docx":"📋","xlsx":"📊","md":"📝","txt":"📄"}.get(ext, "📁")
                size_kb = len(f.getvalue()) / 1024
                st.markdown(
                    f'<div class="frow">'
                    f'<span class="fi">{icon}</span>'
                    f'<span class="fn" title="{f.name}">{f.name}</span>'
                    f'<span class="fch">{size_kb:.0f} KB</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            if st.button("⬆ Ingest All Files", use_container_width=True, key="btn_ingest"):
                # FIX 7: Read all file data into memory BEFORE any rerun
                file_snapshots = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploaded]

                prog = st.progress(0, text="Uploading files to API…")

                # Try direct API upload first
                file_tuples = [("files", snap) for snap in file_snapshots]
                try:
                    r = httpx.post(f"{API_BASE}/ingest/upload", files=file_tuples, timeout=120)
                    result = r.json()
                except Exception as exc:
                    result = {"error": str(exc)}

                prog.progress(50)

                if "error" in result:
                    # Fallback: save locally using the in-memory snapshots (FIX 7)
                    prog.progress(0, text="Saving files locally…")
                    all_ok = True
                    for i, (name, data, _mime) in enumerate(file_snapshots):
                        path = save_upload_locally(name, data)
                        r2   = api_ingest_path(path)
                        prog.progress(int(100 * (i + 1) / len(file_snapshots)))
                        if "error" not in r2:
                            msg_txt = r2.get("message", "")
                            job_id  = msg_txt.split("job_id=")[-1].split(")")[0] if "job_id=" in msg_txt else ""
                            if job_id:
                                st.session_state.ingest_jobs[job_id] = "pending"
                        else:
                            st.error(f"✗ {name}: {r2['error']}")
                            all_ok = False
                    if all_ok:
                        st.success(f"✓ {len(file_snapshots)} file(s) saved and ingestion started!")
                else:
                    prog.progress(100)
                    msg    = result.get("message", "")
                    job_id = msg.split("job_id=")[-1].split(")")[0] if "job_id=" in msg else ""
                    n      = result.get("accepted", len(file_snapshots))
                    if job_id:
                        st.session_state.ingest_jobs[job_id] = "pending"
                    st.success(f"✓ {n} file(s) ingestion started! (job: {job_id})")

                # Refresh doc list — happens after data is already captured
                st.session_state.avail_docs = api_list_documents()
                st.rerun()

    # ── Right: Document Scope Selector ────────────────────────────────────────
    with doc_col:
        st.markdown(
            '<p class="sec" style="margin-top:0">Query Scope</p>',
            unsafe_allow_html=True,
        )

        rl_col, _ = st.columns([1, 3])
        with rl_col:
            if st.button("↻ Reload", key="btn_reload"):
                st.session_state.avail_docs = api_list_documents()
                st.rerun()

        if not st.session_state.avail_docs:
            st.session_state.avail_docs = api_list_documents()

        avail = st.session_state.avail_docs

        if not avail:
            st.markdown(
                '<div style="font-size:0.8rem;color:#9CA3AF;padding:8px 0">'
                'No documents indexed yet. Upload files to get started.</div>',
                unsafe_allow_html=True,
            )
            st.session_state.selected_docs = []
        else:
            # Show doc list preview
            for doc in avail[:6]:
                src   = doc["source"]
                name  = doc["name"]
                ext   = name.split(".")[-1].lower() if "." in name else ""
                icon  = {"pdf":"📄","docx":"📋","xlsx":"📊","md":"📝","txt":"📝"}.get(ext, "📁")
                is_sel = src in st.session_state.selected_docs
                bg     = "#F5F3FF" if is_sel else "#FAFAFA"
                border = "#C7D2FE" if is_sel else "#E5E7EB"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:9px;'
                    f'padding:7px 10px;border-radius:8px;background:{bg};'
                    f'border:1px solid {border};margin-bottom:5px">'
                    f'<span style="font-size:14px">{icon}</span>'
                    f'<span style="flex:1;font-size:0.78rem;color:#374151;font-weight:500;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="{name}">{name}</span>'
                    f'<span style="font-size:0.65rem;color:#9CA3AF;font-family:JetBrains Mono,monospace">'
                    f'{doc["chunks"]}ch</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            doc_map  = {d["name"]: d["source"] for d in avail}
            pre_sel  = [n for n, s in doc_map.items() if s in st.session_state.selected_docs]

            sel_names = st.multiselect(
                "Select documents to query:",
                options=list(doc_map.keys()),
                default=pre_sel,
                placeholder="Leave empty to query all…",
                key="doc_ms",
            )
            st.session_state.selected_docs = [doc_map[n] for n in sel_names]

            if sel_names:
                st.markdown(
                    f'<div class="scope-card">🎯 Querying {len(sel_names)} of {len(avail)} document(s)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="scope-card-all">⬡ Querying all documents</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

# ─────────────────────────────────────────────────────────────────────────────
# CHAT TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    # Open the scrollable message area — this div fills available space
    # and scrolls independently so the chat input never jumps
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            '<div class="empty">'
            '<div class="ei">⬡</div>'
            '<div class="et">Upload documents, then ask anything</div>'
            '<div class="es">TIER 1 · TIER 2 · TIER 3 GraphRAG</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            render_message(msg)

    # Close scrollable area before the input (input sits outside scroll)
    st.markdown('</div>', unsafe_allow_html=True)

    pending    = st.session_state.pop("_pending", None)
    user_input = st.chat_input("Ask anything about your documents…")
    if pending:
        user_input = pending

    if user_input and user_input.strip():
        query_text  = user_input.strip()
        docs_filter = st.session_state.selected_docs or None

        st.session_state.messages.append({"role": "user", "content": query_text})

        # Re-open scroll area for live streaming output
        st.markdown('<div class="chat-scroll" style="flex:unset;overflow:visible">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="msg-u"><div class="bub-u">{query_text}</div></div>',
            unsafe_allow_html=True,
        )

        steps_ph  = st.empty()
        answer_ph = st.empty()
        meta_ph   = st.empty()
        src_ph    = st.empty()

        completed_steps: list[str] = []
        active_step: dict | None   = None
        answer_tokens: list[str]   = []
        final_meta: dict           = {}

        def _draw_steps(active=None):
            parts = [
                f'<div class="step step-done"><span class="si">✓</span><span class="st">{s}</span></div>'
                for s in completed_steps
            ]
            if active:
                parts.append(
                    f'<div class="step step-active">'
                    f'<span class="si">{active["icon"]}</span>'
                    f'<span class="st">{active["text"]} …</span>'
                    f'</div>'
                )
            if parts:
                steps_ph.markdown(
                    f'<div class="steps">{"".join(parts)}</div>',
                    unsafe_allow_html=True,
                )

        for frame in stream_query(query_text, docs_filter):
            ftype = frame.get("type", "")

            if ftype == "step":
                if frame.get("done"):
                    if active_step:
                        completed_steps.append(active_step["text"])
                        active_step = None
                    _draw_steps()
                else:
                    active_step = frame
                    _draw_steps(active=active_step)

            elif ftype == "token":
                answer_tokens.append(frame.get("text", ""))
                raw = "".join(answer_tokens)
                fmt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', raw)
                fmt = fmt.replace("\n", "<br>")
                # FIX 8: cursor class defined in page-level CSS — animation always works
                answer_ph.markdown(
                    f'<div class="msg-b"><div class="bub-b">{fmt}'
                    f'<span class="cursor"></span></div></div>',
                    unsafe_allow_html=True,
                )

            elif ftype == "meta":
                final_meta = frame
                raw = "".join(answer_tokens)
                fmt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', raw)
                fmt = fmt.replace("\n", "<br>")
                answer_ph.markdown(
                    f'<div class="msg-b"><div class="bub-b">{fmt}</div></div>',
                    unsafe_allow_html=True,
                )
                meta_ph.markdown(metrics_html(final_meta), unsafe_allow_html=True)
                src_ph.markdown(sources_html(final_meta.get("sources", [])), unsafe_allow_html=True)

            elif ftype == "error":
                answer_ph.error(f"⚠️ {frame.get('text', 'Unknown error')}")

            elif ftype == "done":
                break

        final_answer = "".join(answer_tokens) or "No answer received."
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "steps": completed_steps,
            "meta": final_meta,
        })

        # Close the streaming output div
        st.markdown('</div>', unsafe_allow_html=True)

        tier = final_meta.get("tier", "TIER_1")
        st.session_state.queries += 1
        st.session_state.latencies.append(final_meta.get("latency_ms", 0))
        if tier in st.session_state.tier_counts:
            st.session_state.tier_counts[tier] += 1

        st.session_state.ragas_history.append({
            "query":         query_text[:55] + ("…" if len(query_text) > 55 else ""),
            "tier":          tier,
            "faithfulness":  final_meta.get("faithfulness_score"),
            "relevancy":     final_meta.get("answer_relevancy"),
            "precision":     final_meta.get("context_precision"),
            "hallucination": final_meta.get("hallucination_score"),
            "latency_ms":    final_meta.get("latency_ms", 0),
        })
        if len(st.session_state.ragas_history) > 20:
            st.session_state.ragas_history = st.session_state.ragas_history[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# EVAL TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    history = st.session_state.ragas_history

    if not history:
        st.info("No queries yet. Start chatting to see evaluation metrics here.")
    else:
        def _avg(key):
            vals = [r[key] for r in history if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        avg_f   = _avg("faithfulness")
        avg_r   = _avg("relevancy")
        avg_p   = _avg("precision")
        avg_h   = _avg("hallucination")
        avg_lat = sum(st.session_state.latencies) / len(st.session_state.latencies) \
                  if st.session_state.latencies else 0

        def _trend(v, inv=False):
            if v is None:
                return "no data", "#9CA3AF"
            good = (v <= 0.2) if inv else (v >= 0.7)
            med  = (v <= 0.4) if inv else (v >= 0.4)
            return (
                ("↑ Good" if good else ("→ Fair" if med else "↓ Low")),
                ("#059669" if good else ("#D97706" if med else "#DC2626")),
            )

        st.markdown("### Aggregate Metrics")
        cols = st.columns(4)
        for col, val, label, inv in [
            (cols[0], avg_f, "Faithfulness",  False),
            (cols[1], avg_r, "Relevancy",     False),
            (cols[2], avg_p, "Precision",     False),
            (cols[3], avg_h, "Hallucination", True),
        ]:
            tl, tc_col = _trend(val, inv)
            cls = score_cls(val, invert=inv) if val is not None else "mb"
            col.markdown(
                f'<div class="ecard">'
                f'<div class="ev {cls}">{pct(val)}</div>'
                f'<div class="el">{label}</div>'
                f'<div class="et2" style="color:{tc_col}">{tl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("### Tier Distribution")
        tc = st.session_state.tier_counts
        t1c, t2c, t3c = st.columns(3)
        for col, key, label, color, bg, br in [
            (t1c, "TIER_1", "⚡ Tier 1 — Simple",    "#059669", "#ECFDF5", "#A7F3D0"),
            (t2c, "TIER_2", "🔄 Tier 2 — Multi-hop", "#D97706", "#FFFBEB", "#FDE68A"),
            (t3c, "TIER_3", "🕸 Tier 3 — GraphRAG",  "#7C3AED", "#F5F3FF", "#DDD6FE"),
        ]:
            col.markdown(
                f'<div class="ecard" style="background:{bg};border-color:{br}">'
                f'<div class="ev" style="color:{color}">{tc[key]}</div>'
                f'<div class="el" style="color:{color}88">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("### Score Breakdown")
        def brow(label, val, max_val, color):
            w    = min((val / max_val) * 100, 100) if val else 0
            disp = pct(val) if max_val == 1 else f"{val:.0f}ms"
            return (
                f'<div class="barrow">'
                f'<span class="barl">{label}</span>'
                f'<div class="bart"><div class="barf" style="width:{w:.0f}%;background:{color}"></div></div>'
                f'<span class="barv" style="color:{color}">{disp if val is not None else "—"}</span>'
                f'</div>'
            )
        bars = (
            brow("Faithfulness",  avg_f,   1,    "#059669") +
            brow("Relevancy",     avg_r,   1,    "#6366F1") +
            brow("Precision",     avg_p,   1,    "#7C3AED") +
            brow("Hallucination", avg_h,   1,    "#DC2626") +
            brow("Avg Latency",   avg_lat, 3000, "#D97706")
        )
        st.markdown(f'<div class="bar-wrap">{bars}</div>', unsafe_allow_html=True)

        st.markdown("---")

        fv = [r["faithfulness"] for r in history if r.get("faithfulness") is not None]
        if fv:
            st.markdown("### Faithfulness Trend")
            spark = "".join(
                f'<div class="spk" style="height:{max(int(v*44),4)}px;'
                f'background:{"#059669" if v>=0.7 else ("#D97706" if v>=0.4 else "#DC2626")}"></div>'
                for v in fv
            )
            st.markdown(
                f'<div class="bar-wrap">'
                f'<div class="spark">{spark}</div>'
                f'<div style="font-size:0.65rem;color:#9CA3AF;font-family:JetBrains Mono,monospace;'
                f'margin-top:6px">← older  newer →  |  '
                + "  ".join(f"{v:.0%}" for v in fv) +
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

        st.markdown("### Query History")
        rows = ""
        for r in reversed(history[-15:]):
            tier  = r.get("tier", "TIER_1")
            t     = TIER_META.get(tier, TIER_META["TIER_1"])
            badge = (
                f'<span style="background:{t["bg"]};color:{t["color"]};'
                f'border:1px solid {t["border"]};padding:2px 8px;border-radius:10px;'
                f'font-size:0.65rem;font-weight:600">{t["label"]}</span>'
            )
            rows += (
                f'<tr>'
                f'<td style="color:#374151">{r["query"]}</td>'
                f'<td>{badge}</td>'
                f'<td style="color:#059669">{pct(r.get("faithfulness"))}</td>'
                f'<td style="color:#6366F1">{pct(r.get("relevancy"))}</td>'
                f'<td style="color:#DC2626">{pct(r.get("hallucination"))}</td>'
                f'<td style="color:#D97706">{r.get("latency_ms",0):.0f}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;'
            f'padding:14px;overflow-x:auto;box-shadow:0 1px 3px rgba(0,0,0,.05)">'
            f'<table class="htbl"><thead><tr>'
            f'<th>Query</th><th>Tier</th><th>Faith</th>'
            f'<th>Relev</th><th>Hall</th><th>ms</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        if st.button("💾 Export Session JSON", use_container_width=True):
            export = {
                "session_info": {
                    "total_queries":    st.session_state.queries,
                    "avg_latency_ms":   avg_lat,
                    "tier_distribution": st.session_state.tier_counts,
                    "exported_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "conversations": [
                    {k: v for k, v in m.items() if k != "steps"}
                    for m in st.session_state.messages
                ],
                "ragas_history": history,
            }
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(export, indent=2, default=str),
                file_name=f"nexusrag_session_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )