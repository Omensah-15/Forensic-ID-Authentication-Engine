"""
SuperPoint Forensic Verification Engine — Streamlit Web Application
====================================================================
Run with:  streamlit run app.py
"""

import os
import io
import json
import time
import uuid
import hashlib
import tempfile
import warnings
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forensic Verification Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — refined dark industrial aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0a0c0f;
    --bg-secondary:  #111418;
    --bg-card:       #161b22;
    --bg-elevated:   #1c2128;
    --border:        #2a3140;
    --border-subtle: #1e2530;
    --accent:        #00d4aa;
    --accent-dim:    #00a882;
    --accent-glow:   rgba(0, 212, 170, 0.12);
    --warn:          #f0b429;
    --danger:        #e85d5d;
    --danger-dim:    rgba(232, 93, 93, 0.12);
    --success:       #00d4aa;
    --success-dim:   rgba(0, 212, 170, 0.10);
    --warn-dim:      rgba(240, 180, 41, 0.12);
    --text-primary:  #e6edf3;
    --text-secondary:#8b98a9;
    --text-muted:    #495664;
    --mono:          'IBM Plex Mono', monospace;
    --sans:          'IBM Plex Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    box-shadow: 0 0 16px var(--accent-glow) !important;
}
.stButton > button:disabled {
    background: var(--text-muted) !important;
    color: var(--bg-secondary) !important;
    cursor: not-allowed !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Inputs & selects ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
.stSelectbox select {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] .st-emotion-cache-1dp5vir {
    background: var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--bg-card) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    padding: 0.75rem 1rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Custom components ── */
.fve-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.fve-header-title {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-primary);
}
.fve-header-sub {
    font-family: var(--mono);
    font-size: 0.70rem;
    letter-spacing: 0.10em;
    color: var(--text-muted);
    text-transform: uppercase;
}

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.verdict-low    { background: var(--success-dim); color: var(--success); border: 1px solid var(--success); }
.verdict-medium { background: var(--warn-dim);    color: var(--warn);    border: 1px solid var(--warn);    }
.verdict-high   { background: var(--danger-dim);  color: var(--danger);  border: 1px solid var(--danger);  }

.score-bar-wrap { margin: 0.4rem 0 0.9rem 0; }
.score-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
}
.score-bar-track {
    height: 6px;
    background: var(--bg-elevated);
    border-radius: 3px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
}
.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.region-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--border-subtle);
    font-family: var(--mono);
    font-size: 0.75rem;
}
.region-row:last-child { border-bottom: none; }
.region-name { color: var(--text-secondary); letter-spacing: 0.05em; text-transform: uppercase; width: 120px; }
.region-status { font-weight: 600; width: 100px; text-align: center; }
.region-conf   { color: var(--text-secondary); width: 70px; text-align: right; }

.status-verified   { color: var(--success); }
.status-suspicious { color: var(--warn);    }
.status-failed     { color: var(--danger);  }

.flag-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0.15rem;
    background: var(--danger-dim);
    color: var(--danger);
    border: 1px solid var(--danger);
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    font-family: var(--mono);
    font-size: 0.75rem;
}
.info-row {
    display: flex;
    flex-direction: column;
    padding: 0.6rem 0.8rem;
    background: var(--bg-elevated);
    border-radius: 4px;
    border: 1px solid var(--border-subtle);
}
.info-key   { color: var(--text-muted);    font-size: 0.65rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; }
.info-value { color: var(--text-primary);  font-weight: 500; }

.section-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-subtle);
}

.mono-block {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    word-break: break-all;
    line-height: 1.6;
}

.chain-valid   { color: var(--success); font-family: var(--mono); font-size: 0.80rem; font-weight: 600; }
.chain-invalid { color: var(--danger);  font-family: var(--mono); font-size: 0.80rem; font-weight: 600; }

.upload-hint {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}

.processing-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-subtle);
}
.step-dot-active   { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
.step-dot-pending  { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.step-dot-complete { width: 8px; height: 8px; border-radius: 50%; background: var(--success); flex-shrink: 0; }

.sidebar-logo {
    font-family: var(--mono);
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.stAlert {
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Engine import with graceful degradation
# ---------------------------------------------------------------------------
ENGINE_AVAILABLE = False
engine_error_msg = ""

try:
    import torch
    import scipy
    from script_v3 import (
        ForensicVerificationEngine,
        FraudScoreCalibrator,
        LogSigner,
        RiskLevel,
        ForensicReport,
        CRYPTO_AVAILABLE,
        SKLEARN_AVAILABLE,
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    engine_error_msg = str(e)
except Exception as e:
    engine_error_msg = str(e)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "engine":        None,
        "report":        None,
        "db_dir":        None,
        "weights_path":  None,
        "engine_ready":  False,
        "history":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def risk_colour(level: str) -> str:
    return {"LOW": "#00d4aa", "MEDIUM": "#f0b429", "HIGH": "#e85d5d"}.get(level, "#8b98a9")


def risk_badge_class(level: str) -> str:
    return {"LOW": "verdict-low", "MEDIUM": "verdict-medium", "HIGH": "verdict-high"}.get(level, "")


def score_bar(label: str, value: float, invert: bool = False, colour: Optional[str] = None):
    """Render a labelled horizontal score bar."""
    display_val = value * 100
    if colour is None:
        if invert:
            colour = "#e85d5d" if value > 0.60 else "#f0b429" if value > 0.25 else "#00d4aa"
        else:
            colour = "#00d4aa" if value > 0.75 else "#f0b429" if value > 0.45 else "#e85d5d"

    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-label">
            <span>{label}</span>
            <span>{display_val:.1f}%</span>
        </div>
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{display_val:.1f}%;background:{colour};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def info_grid(items: List[Tuple[str, str]]):
    """Render a two-column key-value info grid."""
    html = '<div class="info-grid">'
    for key, value in items:
        html += f'<div class="info-row"><span class="info-key">{key}</span><span class="info-value">{value}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def save_upload_to_temp(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp file and return the path."""
    suffix = Path(uploaded_file.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def generate_heatmap_overlay(query_path: str, heatmap_bgr: np.ndarray) -> Optional[Image.Image]:
    """Overlay the tamper heatmap onto the query image at 50% opacity."""
    base = cv2.imread(query_path)
    if base is None or heatmap_bgr is None:
        return None
    h, w = base.shape[:2]
    heat = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return bgr_to_pil(overlay)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">FVE // v3.0</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Engine Configuration</div>', unsafe_allow_html=True)

        weights_path = st.text_input(
            "Weights Path",
            value=st.session_state.get("weights_path") or "superpoint_v1.pth",
            help="Path to superpoint_v1.pth weights file",
            placeholder="superpoint_v1.pth",
        )

        db_dir = st.text_input(
            "Reference Database Directory",
            value=st.session_state.get("db_dir") or "",
            help="Folder containing genuine reference document images",
            placeholder="/path/to/database",
        )

        st.markdown('<div class="section-label">Verification Parameters</div>', unsafe_allow_html=True)

        match_threshold = st.slider(
            "Match Threshold", 0.40, 0.95, 0.70, 0.01,
            help="Minimum descriptor similarity to accept a database match",
        )
        max_keypoints = st.select_slider(
            "Max Keypoints", options=[250, 500, 750, 1000, 1500, 2000], value=1000,
            help="Maximum keypoints to extract per image",
        )
        align_template = st.toggle("Template Alignment", value=False,
            help="Pre-align query to template before analysis (slower but more robust)")

        st.markdown('<div class="section-label">Analysis Modules</div>', unsafe_allow_html=True)

        run_multiscale = st.toggle("Multi-Scale Verification", value=True)
        run_regions    = st.toggle("Region Verification",      value=True)
        run_antispoof  = st.toggle("Anti-Spoof Detection",     value=True)
        use_gpu        = st.toggle("GPU Acceleration",         value=False,
            help="Use CUDA if available")

        st.markdown('<div class="section-label">Audit</div>', unsafe_allow_html=True)

        audit_dir = st.text_input("Audit Log Directory", value="./audit_logs")
        chain_dir = st.text_input("Chain Store Directory", value="./audit_store")

        st.markdown("---")

        if st.button("Initialise Engine", use_container_width=True):
            _init_engine(
                weights_path, db_dir, match_threshold, max_keypoints,
                align_template, run_multiscale, run_regions,
                run_antispoof, use_gpu, audit_dir, chain_dir,
            )

        if st.session_state.engine_ready:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.72rem;'
                'color:#00d4aa;margin-top:0.5rem;letter-spacing:0.06em;">'
                'ENGINE READY</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.72rem;'
                'color:#e85d5d;margin-top:0.5rem;letter-spacing:0.06em;">'
                'ENGINE NOT INITIALISED</div>', unsafe_allow_html=True
            )

        # Store config in session
        st.session_state["db_dir"]          = db_dir
        st.session_state["weights_path"]    = weights_path
        st.session_state["match_threshold"] = match_threshold
        st.session_state["max_keypoints"]   = max_keypoints
        st.session_state["align_template"]  = align_template


def _init_engine(weights, db_dir, threshold, kp, align, ms, regions, spoof, gpu, audit, chain):
    if not ENGINE_AVAILABLE:
        st.sidebar.error(f"Engine unavailable: {engine_error_msg}")
        return

    if not os.path.isfile(weights):
        st.sidebar.error(f"Weights file not found: {weights}")
        return

    if not os.path.isdir(db_dir):
        st.sidebar.error(f"Database directory not found: {db_dir}")
        return

    with st.sidebar:
        with st.spinner("Loading SuperPoint weights..."):
            try:
                engine = ForensicVerificationEngine(
                    weights_path            = weights,
                    conf_thresh             = 0.005,
                    nms_dist                = 4,
                    border_remove           = 4,
                    cuda                    = gpu,
                    audit_log_dir           = audit,
                    chain_store_dir         = chain or None,
                    run_multiscale          = ms,
                    run_region_verification = regions,
                    run_anti_spoof          = spoof,
                    base_size               = (640, 480),
                    cache_size              = 512,
                    max_workers             = 4,
                )
                st.session_state["engine"]       = engine
                st.session_state["engine_ready"] = True
                st.session_state["db_dir"]       = db_dir
                st.success("Engine initialised.")
            except Exception as exc:
                st.error(f"Initialisation failed: {exc}")
                st.session_state["engine_ready"] = False


# ---------------------------------------------------------------------------
# Main verification tab
# ---------------------------------------------------------------------------

def render_verify_tab():
    st.markdown("""
    <div class="fve-header">
        <span class="fve-header-title">Document Verification</span>
        <span class="fve-header-sub">Upload a document image to run the full forensic pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.engine_ready:
        st.warning("Configure and initialise the engine in the sidebar before running verification.")
        return

    uploaded = st.file_uploader(
        "Query Document",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="The document you want to verify",
        label_visibility="collapsed",
    )
    st.markdown('<div class="upload-hint">Supported formats: JPG, PNG, BMP, TIFF, WEBP</div>',
                unsafe_allow_html=True)

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        run_btn = st.button(
            "Run Verification",
            disabled=(uploaded is None),
            use_container_width=True,
        )

    if uploaded and run_btn:
        _run_verification(uploaded)

    if st.session_state.report is not None:
        render_report(st.session_state.report)


def _run_verification(uploaded_file):
    engine  = st.session_state["engine"]
    db_dir  = st.session_state["db_dir"]
    thresh  = st.session_state.get("match_threshold", 0.70)
    kp      = st.session_state.get("max_keypoints",  1000)
    align   = st.session_state.get("align_template", False)

    query_path = save_upload_to_temp(uploaded_file)

    steps = [
        "Extracting keypoints and descriptors",
        "Scanning reference database",
        "Geometric consistency check",
        "Tamper localisation",
        "Anti-spoof analysis",
        "Multi-scale verification",
        "Region verification",
        "Computing fraud score",
        "Writing audit log",
    ]

    progress_container = st.empty()
    step_container     = st.empty()

    def update_progress(i):
        progress_container.progress((i + 1) / len(steps))
        dots = ""
        html = ""
        for idx, step in enumerate(steps):
            if idx < i:
                dot_cls = "step-dot-complete"
                colour  = "var(--text-muted)"
            elif idx == i:
                dot_cls = "step-dot-active"
                colour  = "var(--text-primary)"
            else:
                dot_cls = "step-dot-pending"
                colour  = "var(--text-muted)"
            html += (
                f'<div class="processing-step" style="color:{colour}">'
                f'<div class="{dot_cls}"></div>{step}</div>'
            )
        step_container.markdown(html, unsafe_allow_html=True)

    try:
        update_progress(0)
        time.sleep(0.05)

        t_start = time.perf_counter()

        # Simulate step updates during the blocking call
        update_progress(1)
        report = engine.verify(
            query_path      = query_path,
            database_dir    = db_dir,
            match_threshold = thresh,
            max_keypoints   = kp,
            visualize       = False,
            align_template  = align,
        )

        for i in range(2, len(steps)):
            update_progress(i)
            time.sleep(0.06)

        progress_container.empty()
        step_container.empty()

        st.session_state["report"] = report
        st.session_state["history"].append({
            "session_id":       report.session_id,
            "timestamp":        time.strftime("%H:%M:%S"),
            "risk":             report.risk_level.value,
            "fraud_pct":        f"{report.fraud_probability * 100:.1f}%",
            "auth_pct":         f"{report.authenticity_score * 100:.1f}%",
            "query":            uploaded_file.name,
        })

    except Exception as exc:
        progress_container.empty()
        step_container.empty()
        st.error(f"Verification failed: {exc}")

    finally:
        try:
            os.unlink(query_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(report: "ForensicReport"):
    st.markdown("---")
    risk = report.risk_level.value if hasattr(report.risk_level, "value") else str(report.risk_level)

    # ── Top verdict banner ──────────────────────────────────────────────────
    badge_cls = risk_badge_class(risk)
    col_verdict, col_sid = st.columns([3, 2])
    with col_verdict:
        st.markdown(
            f'<div class="verdict-badge {badge_cls}">'
            f'RISK: {risk}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_sid:
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:0.72rem;'
            f'color:var(--text-muted);text-align:right;padding-top:0.5rem;">'
            f'SESSION: {report.session_id}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Fraud Probability",  f"{report.fraud_probability * 100:.1f}%")
    with m2:
        st.metric("Authenticity Score", f"{report.authenticity_score * 100:.1f}%")
    with m3:
        st.metric("Descriptor Sim.",    f"{report.descriptor_similarity:.4f}")
    with m4:
        st.metric("Processing Time",    f"{report.processing_time_sec:.2f}s")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabbed detail panels ────────────────────────────────────────────────
    tabs = st.tabs([
        "Overview",
        "Geometric",
        "Tamper",
        "Anti-Spoof",
        "Regions",
        "Multi-Scale",
        "Audit",
    ])

    # ── TAB 1 : Overview ────────────────────────────────────────────────────
    with tabs[0]:
        col_scores, col_match = st.columns([1, 1])

        with col_scores:
            section_label("Signal Scores")
            score_bar("Authenticity",         report.authenticity_score,       invert=False)
            score_bar("Fraud Probability",    report.fraud_probability,        invert=True)
            score_bar("Descriptor Similarity",report.descriptor_similarity,    invert=False)

            if report.geometric:
                score_bar("Geometric Inlier Ratio", report.geometric.inlier_ratio, invert=False)
            if report.anti_spoof:
                score_bar("Spoof Probability", report.anti_spoof.overall_spoof_probability, invert=True)

        with col_match:
            section_label("Match Information")
            info_grid([
                ("Best Match",    os.path.basename(report.best_match_path) if report.best_match_path else "No Match"),
                ("Risk Level",    risk),
                ("Query",         os.path.basename(report.query_path)),
                ("Session ID",    report.session_id),
            ])

            if report.db_stats:
                st.markdown("<br>", unsafe_allow_html=True)
                section_label("Database Statistics")
                info_grid([
                    ("Documents Evaluated", str(report.db_stats.total_documents)),
                    ("Cached Descriptors",  str(report.db_stats.cached_documents)),
                    ("Cache Hit Rate",      f"{report.db_stats.cache_hit_rate * 100:.1f}%"),
                    ("DB Scan Time",        f"{report.db_stats.index_build_time:.2f}s"),
                ])

    # ── TAB 2 : Geometric ───────────────────────────────────────────────────
    with tabs[1]:
        if report.geometric is None:
            st.info("Geometric analysis not available for this report.")
        else:
            g = report.geometric
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                section_label("Homography Statistics")
                info_grid([
                    ("Inlier Count",       str(g.inlier_count)),
                    ("Inlier Ratio",       f"{g.inlier_ratio * 100:.1f}%"),
                    ("Reprojection Error", f"{g.reprojection_error:.2f} px"
                     if g.reprojection_error < float('inf') else "N/A"),
                    ("H Stability",        f"{g.homography_stability:.4f}"),
                    ("Condition Number",   f"{g.condition_number:.2f}"
                     if g.condition_number < float('inf') else "Degenerate"),
                ])

            with col_g2:
                section_label("Interpretation")
                if g.inlier_ratio >= 0.60:
                    st.success("High inlier ratio — strong geometric consistency with the reference template.")
                elif g.inlier_ratio >= 0.30:
                    st.warning("Moderate inlier ratio — some regions may be inconsistent.")
                else:
                    st.error("Low inlier ratio — poor geometric consistency. Document likely tampered or incorrect type.")

                if g.reprojection_error < float('inf'):
                    if g.reprojection_error < 3.0:
                        st.success(f"Reprojection error {g.reprojection_error:.2f}px — sub-pixel accuracy.")
                    elif g.reprojection_error < 8.0:
                        st.warning(f"Reprojection error {g.reprojection_error:.2f}px — acceptable but elevated.")
                    else:
                        st.error(f"Reprojection error {g.reprojection_error:.2f}px — geometric fit is poor.")

    # ── TAB 3 : Tamper ──────────────────────────────────────────────────────
    with tabs[2]:
        if report.tamper is None:
            st.info("Tamper analysis not available for this report.")
        else:
            t = report.tamper
            col_t1, col_t2 = st.columns([1, 1])

            with col_t1:
                section_label("Tamper Metrics")
                info_grid([
                    ("Unmatched Ratio",     f"{t.unmatched_ratio * 100:.1f}%"),
                    ("Structural Deviation",f"{t.structural_deviation:.4f}"),
                    ("Suspicious Zones",    ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None"),
                ])

                if t.quadrant_scores:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_label("Quadrant Deviation Scores")
                    for qname, qscore in t.quadrant_scores.items():
                        score_bar(
                            qname.replace("-", " ").title(),
                            min(qscore / 100.0, 1.0),
                            invert=True,
                        )

            with col_t2:
                section_label("Deviation Heatmap")
                if t.deviation_heatmap is not None and report.best_match_path:
                    try:
                        overlay = generate_heatmap_overlay(report.query_path, t.deviation_heatmap)
                        if overlay:
                            st.image(overlay, use_container_width=True,
                                     caption="Tamper deviation overlay — bright areas indicate suspicious regions")
                        else:
                            hm_pil = bgr_to_pil(t.deviation_heatmap)
                            st.image(hm_pil, use_container_width=True, caption="Tamper deviation heatmap")
                    except Exception:
                        st.info("Heatmap rendering unavailable.")
                else:
                    st.info("No deviation heatmap generated — query image may not have matched a reference.")

    # ── TAB 4 : Anti-Spoof ──────────────────────────────────────────────────
    with tabs[3]:
        if report.anti_spoof is None:
            st.info("Anti-spoof analysis was not enabled for this run.")
        else:
            a = report.anti_spoof
            col_a1, col_a2 = st.columns(2)

            with col_a1:
                section_label("Signal Scores")
                score_bar("Moire Pattern",         a.moire_score,             invert=True)
                score_bar("Photocopy",             a.photocopy_score,         invert=True)
                score_bar("Screen Replay",         a.screen_replay_score,     invert=True)
                score_bar("Print-Scan Artefact",   a.print_scan_score,        invert=True)
                score_bar("Compression Artefact",  a.compression_score,       invert=True)
                score_bar("Overall Spoof Probability", a.overall_spoof_probability, invert=True)

            with col_a2:
                section_label("Triggered Flags")
                if a.flags:
                    flags_html = "".join(
                        f'<span class="flag-pill">{f}</span>' for f in a.flags
                    )
                    st.markdown(flags_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="font-family:var(--mono);font-size:0.78rem;'
                        'color:var(--success);padding:0.5rem 0;">No spoof flags triggered</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                section_label("What Each Signal Means")
                with st.expander("Moire Pattern"):
                    st.markdown(
                        "Detected when a document is photographed from a printed surface. "
                        "Ink dots and camera sensor interfere, creating wavy fringe patterns "
                        "visible in the frequency spectrum.",
                        unsafe_allow_html=False,
                    )
                with st.expander("Photocopy"):
                    st.markdown(
                        "Photocopied documents show flattened tonal range — deep blacks and "
                        "bright whites both compress toward grey. Measured via histogram IQR spread "
                        "and shadow/highlight clipping.",
                    )
                with st.expander("Screen Replay"):
                    st.markdown(
                        "When a document is displayed on a screen and re-photographed, the screen's "
                        "horizontal scan-lines leave a periodic frequency signature. Also detects "
                        "the blur introduced by re-capture optics.",
                    )
                with st.expander("Print-Scan Artefact"):
                    st.markdown(
                        "Professional printing introduces halftone rosette patterns. Subsequent "
                        "scanning adds its own compression signature. Detected via 8x8 DCT "
                        "block coefficient distribution analysis.",
                    )
                with st.expander("Compression Artefact"):
                    st.markdown(
                        "Heavy JPEG re-compression from repeated save operations leaves "
                        "elevated gradient discontinuities at 8-pixel block boundaries. "
                        "A strong signal indicates the image has been opened, edited, and re-saved multiple times.",
                    )

    # ── TAB 5 : Regions ─────────────────────────────────────────────────────
    with tabs[4]:
        if not report.region_results:
            st.info("Region verification was not enabled or produced no results.")
        else:
            section_label("Per-Zone Verification Results")

            header_html = """
            <div class="region-row" style="border-bottom:1px solid var(--border);margin-bottom:0.2rem;">
                <span class="region-name" style="color:var(--text-muted);font-size:0.65rem;">Region</span>
                <span class="region-status" style="color:var(--text-muted);font-size:0.65rem;">Status</span>
                <span class="region-conf"   style="color:var(--text-muted);font-size:0.65rem;">Confidence</span>
            </div>
            """
            rows_html = ""
            for rr in report.region_results:
                status_cls = {
                    "VERIFIED":   "status-verified",
                    "SUSPICIOUS": "status-suspicious",
                    "FAILED":     "status-failed",
                }.get(rr.status, "")
                rows_html += f"""
                <div class="region-row">
                    <span class="region-name">{rr.name.replace("_", " ").upper()}</span>
                    <span class="region-status {status_cls}">{rr.status}</span>
                    <span class="region-conf">{rr.confidence * 100:.1f}%</span>
                </div>
                """
            st.markdown(
                f'<div style="background:var(--bg-card);border:1px solid var(--border);'
                f'border-radius:6px;padding:0.5rem;">'
                f'{header_html}{rows_html}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            section_label("Region Confidence Breakdown")
            for rr in report.region_results:
                colour = (
                    "#00d4aa" if rr.status == "VERIFIED" else
                    "#f0b429" if rr.status == "SUSPICIOUS" else
                    "#e85d5d"
                )
                score_bar(
                    rr.name.replace("_", " ").title(),
                    rr.confidence,
                    colour=colour,
                )

    # ── TAB 6 : Multi-Scale ─────────────────────────────────────────────────
    with tabs[5]:
        if report.multiscale is None:
            st.info("Multi-scale verification was not enabled for this run.")
        else:
            ms = report.multiscale
            col_ms1, col_ms2 = st.columns(2)

            with col_ms1:
                section_label("Scale Scores")
                score_bar("Original Resolution",  ms.original_score,   invert=False)
                score_bar("Downscale (0.5x)",      ms.downscale_score,  invert=False)
                score_bar("Upscale (2.0x)",        ms.upscale_score,    invert=False)

            with col_ms2:
                section_label("Consistency Analysis")
                info_grid([
                    ("Consistency Score", f"{ms.consistency_score:.4f}"),
                    ("Scale Variance",    f"{ms.scale_variance:.6f}"),
                    ("Stability",         "Stable" if ms.consistency_score < 0.05 else "Unstable"),
                ])

                st.markdown("<br>", unsafe_allow_html=True)
                if ms.consistency_score < 0.05:
                    st.success(
                        "Scores are consistent across all three scales. "
                        "This is consistent with a genuine document."
                    )
                elif ms.consistency_score < 0.12:
                    st.warning(
                        "Minor score variation across scales. "
                        "Could indicate resolution-specific manipulation."
                    )
                else:
                    st.error(
                        "High score variance across scales. "
                        "Document may have been generated or edited at a specific resolution."
                    )

    # ── TAB 7 : Audit ───────────────────────────────────────────────────────
    with tabs[6]:
        col_fp, col_sig = st.columns(2)

        with col_fp:
            section_label("Integrity Fingerprint")
            info_grid([
                ("Image SHA-256",    report.fingerprint.image_sha256[:32] + "..."),
                ("Descriptor Hash",  report.fingerprint.descriptor_hash[:32] + "..."),
                ("File Size",        f"{report.fingerprint.file_size_bytes:,} bytes"),
                ("Timestamp (UTC)",  time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.gmtime(report.fingerprint.timestamp))),
            ])

            if report.audit_log_path:
                st.markdown("<br>", unsafe_allow_html=True)
                section_label("Audit Log")
                st.markdown(
                    f'<div class="mono-block">{report.audit_log_path}</div>',
                    unsafe_allow_html=True,
                )

        with col_sig:
            section_label("Cryptographic Signatures")
            if report.audit_signature:
                sig = report.audit_signature
                st.markdown(
                    f'<div class="mono-block">'
                    f'<div style="color:var(--text-muted);margin-bottom:0.4rem;">RSA-PSS-4096</div>'
                    f'{sig.get("rsa_signature", "N/A")[:64]}...<br><br>'
                    f'<div style="color:var(--text-muted);margin-bottom:0.4rem;">ECDSA-P384</div>'
                    f'{sig.get("ec_signature", "N/A")[:64]}...'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    "Audit log signing is disabled. "
                    "Generate key pairs and provide them to the engine to enable signed logs."
                )

        # Chain verification
        engine = st.session_state.get("engine")
        if engine and hasattr(engine, "chain_store") and engine.chain_store:
            st.markdown("<br>", unsafe_allow_html=True)
            section_label("Audit Chain Integrity")
            if st.button("Verify Chain"):
                with st.spinner("Verifying chain..."):
                    valid, violations = engine.chain_store.verify_chain()
                if valid:
                    st.markdown(
                        '<div class="chain-valid">CHAIN VALID — All entries intact and unmodified</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chain-invalid">CHAIN COMPROMISED — Violations detected</div>',
                        unsafe_allow_html=True,
                    )
                    for v in violations:
                        st.error(v)


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

def render_history_tab():
    st.markdown("""
    <div class="fve-header">
        <span class="fve-header-title">Verification History</span>
        <span class="fve-header-sub">Session log of all verifications run this session</span>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    if not history:
        st.info("No verifications have been run this session.")
        return

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

    header_html = """
    <div class="region-row" style="border-bottom:1px solid var(--border);">
        <span style="width:100px;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);">TIME</span>
        <span style="width:120px;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);">SESSION</span>
        <span style="flex:1;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);">FILE</span>
        <span style="width:80px;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);text-align:center;">RISK</span>
        <span style="width:80px;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);text-align:right;">FRAUD</span>
        <span style="width:80px;font-family:var(--mono);font-size:0.65rem;color:var(--text-muted);text-align:right;">AUTH</span>
    </div>
    """
    rows_html = ""
    for entry in reversed(history):
        risk_col = {"LOW": "#00d4aa", "MEDIUM": "#f0b429", "HIGH": "#e85d5d"}.get(entry["risk"], "#8b98a9")
        rows_html += f"""
        <div class="region-row">
            <span style="width:100px;font-family:var(--mono);font-size:0.72rem;color:var(--text-muted);">{entry['timestamp']}</span>
            <span style="width:120px;font-family:var(--mono);font-size:0.72rem;color:var(--text-secondary);">{entry['session_id']}</span>
            <span style="flex:1;font-family:var(--mono);font-size:0.72rem;color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{entry['query']}</span>
            <span style="width:80px;font-family:var(--mono);font-size:0.72rem;font-weight:600;color:{risk_col};text-align:center;">{entry['risk']}</span>
            <span style="width:80px;font-family:var(--mono);font-size:0.72rem;color:var(--text-secondary);text-align:right;">{entry['fraud_pct']}</span>
            <span style="width:80px;font-family:var(--mono);font-size:0.72rem;color:var(--text-secondary);text-align:right;">{entry['auth_pct']}</span>
        </div>
        """

    st.markdown(
        f'<div style="background:var(--bg-card);border:1px solid var(--border);'
        f'border-radius:6px;padding:0.5rem;">'
        f'{header_html}{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

def render_about_tab():
    st.markdown("""
    <div class="fve-header">
        <span class="fve-header-title">About This System</span>
        <span class="fve-header-sub">SuperPoint Forensic Verification Engine v3.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:var(--sans);font-size:0.88rem;color:var(--text-secondary);
                line-height:1.8;max-width:800px;">
    This system verifies the authenticity of identity documents using a 15-layer
    forensic pipeline built on the SuperPoint neural network. It does not simply
    classify a document as real or fake — it provides a detailed, evidence-backed
    report covering geometric consistency, tamper localisation, anti-spoof signals,
    region-by-region analysis, and a cryptographically signed audit trail.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    layers = [
        ("01", "Neural Descriptor Extraction",   "SuperPoint reciprocal cosine keypoint matching"),
        ("02", "Geometric Consistency",           "RANSAC homography with inlier/reprojection analysis"),
        ("03", "Tamper Localisation",             "Quadrant deviation heatmap with continuous scoring"),
        ("04", "Multi-Scale Verification",        "Three-resolution consistency check"),
        ("05", "Region Verification",             "Per-zone independent analysis (photo, hologram, serial, logo)"),
        ("06", "Integrity Fingerprint",           "SHA-256 image and descriptor hash"),
        ("07", "Statistical Fraud Scoring",       "Weighted composite with optional calibration"),
        ("08", "Anti-Spoof Detection",            "Moire, photocopy, screen replay, print-scan, compression"),
        ("09", "Adaptive Threshold Management",   "Deployment-aware self-updating signal thresholds"),
        ("10", "Signed Audit Log",                "RSA-PSS-4096 and ECDSA-P384 dual signatures"),
        ("11", "Hash-Chained Audit Storage",      "Append-only tamper-evident blockchain-style chain"),
        ("12", "Template Alignment",              "ORB-RANSAC homographic pre-alignment"),
        ("13", "Descriptor Cache",                "LRU cache with file-mtime invalidation"),
        ("14", "Batch Verification",              "Concurrent multi-document processing"),
        ("15", "Confidence Calibration",          "Isotonic regression with ROC-optimal thresholds"),
    ]

    html = '<div style="display:flex;flex-direction:column;gap:0.3rem;max-width:900px;">'
    for num, name, desc in layers:
        html += f"""
        <div style="display:flex;align-items:baseline;gap:1rem;padding:0.55rem 0.8rem;
                    background:var(--bg-card);border:1px solid var(--border-subtle);
                    border-radius:4px;">
            <span style="font-family:var(--mono);font-size:0.68rem;color:var(--accent);
                          width:28px;flex-shrink:0;">{num}</span>
            <span style="font-family:var(--mono);font-size:0.78rem;font-weight:600;
                          color:var(--text-primary);width:240px;flex-shrink:0;">{name}</span>
            <span style="font-family:var(--sans);font-size:0.78rem;color:var(--text-secondary);">{desc}</span>
        </div>
        """
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_label("Dependencies")

    deps = [
        ("torch",         ">=1.12.0",  ENGINE_AVAILABLE),
        ("opencv-python", ">=4.5.0",   True),
        ("numpy",         ">=1.22.0",  True),
        ("scipy",         ">=1.8.0",   True),
        ("streamlit",     ">=1.28.0",  True),
        ("cryptography",  ">=41.0.0",  CRYPTO_AVAILABLE if ENGINE_AVAILABLE else False),
        ("scikit-learn",  ">=1.2.0",   SKLEARN_AVAILABLE if ENGINE_AVAILABLE else False),
    ]

    dep_html = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">'
    for pkg, ver, available in deps:
        colour  = "#00d4aa" if available else "#e85d5d"
        bg      = "rgba(0,212,170,0.08)" if available else "rgba(232,93,93,0.08)"
        border  = "rgba(0,212,170,0.25)" if available else "rgba(232,93,93,0.25)"
        dep_html += (
            f'<div style="font-family:var(--mono);font-size:0.72rem;padding:0.3rem 0.7rem;'
            f'border-radius:3px;background:{bg};border:1px solid {border};color:{colour};">'
            f'{pkg} {ver}</div>'
        )
    dep_html += '</div>'
    st.markdown(dep_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:baseline;justify-content:space-between;
                margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border);">
        <div>
            <div style="font-family:var(--mono);font-size:1.3rem;font-weight:600;
                         letter-spacing:0.14em;color:var(--text-primary);text-transform:uppercase;">
                Forensic Verification Engine
            </div>
            <div style="font-family:var(--mono);font-size:0.68rem;letter-spacing:0.12em;
                         color:var(--text-muted);text-transform:uppercase;margin-top:0.2rem;">
                Document Authenticity Analysis System &nbsp;/&nbsp; v3.0
            </div>
        </div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:var(--text-muted);
                     letter-spacing:0.06em;text-align:right;">
            SuperPoint Neural Engine<br>15-Layer Forensic Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not ENGINE_AVAILABLE:
        st.error(
            f"Engine modules could not be imported: {engine_error_msg}. "
            "Ensure script_v3.py is in the same directory and all dependencies are installed."
        )

    render_sidebar()

    tab_verify, tab_history, tab_about = st.tabs([
        "Verification",
        "History",
        "About",
    ])

    with tab_verify:
        render_verify_tab()

    with tab_history:
        render_history_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
