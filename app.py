"""
Forensic ID Authentication Engine UI
"""

import io
import os
import sys
import time
import json
import datetime
import tempfile
import shutil
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic ID Authentication Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  GLOBAL STYLES — clean, theme-adaptive (like CreditIQ)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* CSS variables that adapt to both light/dark themes */
:root {
    /* Base colors - will be overridden by Streamlit's theme */
    --bg-primary: var(--background-color, #FFFFFF);
    --bg-secondary: var(--secondary-background-color, #F7F8FA);
    --text-primary: var(--text-color, #111827);
    --text-secondary: var(--text-color, #4B5563);
    --text-tertiary: #9CA3AF;
    --border-light: rgba(128, 128, 128, 0.2);
    --border-medium: rgba(128, 128, 128, 0.3);
    
    /* Accent colors - neutral professional palette */
    --accent: #4B5563;
    --accent-hover: #1F2937;
    --accent-soft: rgba(75, 85, 99, 0.1);
    --green: #059669;
    --green-soft: rgba(5, 150, 105, 0.1);
    --amber: #D97706;
    --amber-soft: rgba(217, 119, 6, 0.1);
    --red: #DC2626;
    --red-soft: rgba(220, 38, 38, 0.1);
    
    /* Effects */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 25px rgba(0,0,0,0.1), 0 4px 10px rgba(0,0,0,0.04);
    
    /* Border radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

/* Base typography */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}

/* Headings */
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
h2 { font-size: 1.15rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }

/* Labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stRadio label {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.055em !important;
    color: var(--text-secondary) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-primary) !important;
    border: 1.5px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    transition: all 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-primary) !important;
    border: 1.5px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* Radio pills - modern toggle style */
.stRadio > div {
    gap: 4px !important;
    background: var(--bg-secondary) !important;
    padding: 4px !important;
    border-radius: 40px !important;
    display: inline-flex !important;
    border: 1px solid var(--border-light) !important;
}
.stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 5px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin: 0 !important;
}
.stRadio > div > label:hover {
    color: var(--accent) !important;
}
.stRadio > div > label[data-checked="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── BUTTONS — clean, modern, like CreditIQ ── */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em !important;
    border-radius: var(--radius-md) !important;
    padding: 0.62rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Primary button — neutral gradient */
.stFormSubmitButton > button,
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(75, 85, 99, 0.3) !important;
}
.stFormSubmitButton > button:hover,
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(75, 85, 99, 0.4) !important;
}
.stFormSubmitButton > button:active,
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* Secondary button — glass morphism */
.stButton > button {
    background: rgba(128, 128, 128, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: rgba(128, 128, 128, 0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text-primary) !important;
    border: 1.5px dashed var(--border-medium) !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
    transform: translateY(-2px) !important;
}

/* Metrics/KPI cards */
[data-testid="stMetric"],
.kpi-card {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s !important;
}
[data-testid="stMetric"]:hover,
.kpi-card:hover {
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.055em !important;
    color: var(--text-tertiary) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Alerts */
.stSuccess { 
    background: var(--green-soft) !important; 
    border-left: 3px solid var(--green) !important; 
    border-radius: var(--radius-sm) !important; 
}
.stError { 
    background: var(--red-soft) !important; 
    border-left: 3px solid var(--red) !important; 
    border-radius: var(--radius-sm) !important; 
}
.stWarning { 
    background: var(--amber-soft) !important; 
    border-left: 3px solid var(--amber) !important; 
    border-radius: var(--radius-sm) !important; 
}

/* Dividers */
hr { 
    border-color: var(--border-light) !important; 
    margin: 1.8rem 0 !important; 
}

/* Utility classes */
.kpi-label {
    font-size: 0.7rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.07em; 
    color: var(--text-tertiary); 
    margin-bottom: 0.4rem;
}
.kpi-value { 
    font-size: 2rem; 
    font-weight: 700; 
    line-height: 1.1; 
    color: var(--text-primary); 
}
.kpi-sub { 
    font-size: 0.78rem; 
    color: var(--text-tertiary); 
    margin-top: 0.2rem; 
}

.section-eyebrow {
    font-size: 0.7rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.08em; 
    color: var(--accent);
    padding-bottom: 0.55rem; 
    border-bottom: 1.5px solid var(--border-light); 
    margin-bottom: 1.3rem;
}

/* Data rows */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 0.85rem;
}
.data-row:last-child { border-bottom: none; }
.data-key { color: var(--text-tertiary); }
.data-value { color: var(--text-primary); font-weight: 500; }

/* Hash block */
.hash-block {
    background: var(--bg-secondary); 
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm); 
    padding: 14px 18px;
}
.hash-label {
    font-size: 0.68rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.07em; 
    color: var(--text-tertiary); 
    margin-bottom: 5px;
}
.hash-value {
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.78rem;
    color: var(--green); 
    word-break: break-all;
}

/* Spoof bars */
.sb-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.4rem 0;
    font-size: 0.85rem;
}
.sb-lbl {
    color: var(--text-tertiary);
    width: 120px;
    flex-shrink: 0;
}
.sb-bg {
    flex: 1;
    height: 6px;
    background: var(--border-light);
    position: relative;
    border-radius: 3px;
    overflow: hidden;
}
.sb-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
}
.sb-val {
    color: var(--text-secondary);
    width: 50px;
    text-align: right;
    font-weight: 500;
}

/* Region table */
.region-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.region-table th {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-tertiary);
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border-light);
    text-align: left;
}
.region-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border-light);
    color: var(--text-secondary);
}
.region-table tr:last-child td { border-bottom: none; }

.status-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
}
.status-verified {
    color: var(--green);
    background: var(--green-soft);
    border: 1px solid var(--green);
}
.status-suspicious {
    color: var(--amber);
    background: var(--amber-soft);
    border: 1px solid var(--amber);
}
.status-failed {
    color: var(--red);
    background: var(--red-soft);
    border: 1px solid var(--red);
}

/* Chips */
.chip {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 10px;
    margin: 2px 4px 2px 0;
    color: var(--amber);
    background: var(--amber-soft);
    border: 1px solid var(--amber);
    border-radius: 4px;
}

/* File upload zone */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-secondary) !important;
    border: 1px dashed var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    color: var(--text-primary) !important;
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-sm) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Resolve root and add to sys.path so script_v2 is importable ──────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Optional cv2 (headless-safe) ─────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── Persistent session temp dir — survives reruns within same session ─────────
if "session_dir" not in st.session_state:
    st.session_state["session_dir"] = Path(tempfile.mkdtemp(prefix="fiae_session_"))

SESSION_DIR = st.session_state["session_dir"]
DB_SESSION  = SESSION_DIR / "database"   # reference images stored here per session
Q_DIR       = SESSION_DIR / "query"      # query image stored here
DB_REPO     = ROOT / "database"          # committed database folder (if present)

DB_SESSION.mkdir(parents=True, exist_ok=True)
Q_DIR.mkdir(parents=True, exist_ok=True)

# ── Session state for results ─────────────────────────────────────────────────
if "report" not in st.session_state:
    st.session_state["report"] = None
if "q_path" not in st.session_state:
    st.session_state["q_path"] = None
if "elapsed" not in st.session_state:
    st.session_state["elapsed"] = None

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct_col(v: float) -> str:
    return "#059669" if v >= 75 else ("#D97706" if v >= 45 else "#DC2626")

def bar_col(s: float) -> str:
    return "#059669" if s < .25 else ("#D97706" if s < .55 else "#DC2626")

def risk_classes(r: str):
    if r == "LOW":
        return "#059669", "#059669"
    elif r == "MEDIUM":
        return "#D97706", "#D97706"
    else:
        return "#DC2626", "#DC2626"

def verdict_text(r: str, fp: float):
    if r == "LOW":
        return "Document Authenticated", \
               f"All forensic layers confirm structural authenticity. Fraud probability {fp:.1f}%."
    if r == "MEDIUM":
        return "Requires Manual Review", \
               f"Ambiguous signals on one or more layers. Fraud probability {fp:.1f}%."
    return "Potential Fraud Detected", \
           f"Multiple layers flagged structural anomalies. Fraud probability {fp:.1f}%. Do not authenticate."

def steps_html(active: int, steps: list) -> str:
    rows = ""
    for i, (lbl, tag) in enumerate(steps):
        if i < active:
            dot = '<div style="width:8px;height:8px;border-radius:50%;background:#059669;"></div>'
            tc = "#059669"
        elif i == active:
            dot = '<div style="width:8px;height:8px;border-radius:50%;background:#4B5563;animation:blink 1s infinite;"></div>'
            tc = "#4B5563"
        else:
            dot = '<div style="width:8px;height:8px;border-radius:50%;background:var(--border-light);"></div>'
            tc = "var(--text-tertiary)"
        
        rows += (f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.4rem 0;">{dot}'
                 f'<span style="min-width:70px;color:{tc};">{tag}</span>'
                 f'<span style="color:var(--text-secondary);">{lbl}</span></div>')
    
    return f'<div style="padding:0.5rem 0;">{rows}</div>'

def write_upload(f, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f.name
    p.write_bytes(f.getbuffer())
    return p

def get_db_dir() -> tuple[Path, list]:
    """
    Returns (db_path, image_list).
    Priority: session uploads > committed database/ folder.
    """
    session_imgs = sorted([
        p for p in DB_SESSION.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]) if DB_SESSION.exists() else []

    repo_imgs = sorted([
        p for p in DB_REPO.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]) if DB_REPO.exists() else []

    if session_imgs:
        return DB_SESSION, session_imgs
    if repo_imgs:
        return DB_REPO, repo_imgs
    return DB_SESSION, []

def side_by_side(q: Path, m: Path) -> Image.Image:
    qi = Image.open(str(q)).convert("RGB")
    mi = Image.open(str(m)).convert("RGB")
    h  = 320
    qi = qi.resize((int(qi.width * h / qi.height), h), Image.LANCZOS)
    mi = mi.resize((int(mi.width * h / mi.height), h), Image.LANCZOS)
    gap   = 8
    total = qi.width + gap + mi.width
    out   = Image.new("RGB", (total, h), "#F7F8FA")
    out.paste(qi, (0, 0))
    out.paste(mi, (qi.width + gap, 0))
    return out

def generate_deviation_heatmap(query_img, reference_img):
    """
    Generate a proper deviation heatmap for tamper detection
    """
    try:
        # Convert images to grayscale
        if isinstance(query_img, (str, Path)):
            query = cv2.imread(str(query_img), cv2.IMREAD_GRAYSCALE)
            reference = cv2.imread(str(reference_img), cv2.IMREAD_GRAYSCALE)
        else:
            query = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2GRAY)
            reference = cv2.cvtColor(np.array(reference_img), cv2.COLOR_RGB2GRAY)
        
        # Resize reference to match query if needed
        if query.shape != reference.shape:
            reference = cv2.resize(reference, (query.shape[1], query.shape[0]))
        
        # Compute absolute difference
        diff = cv2.absdiff(query, reference)
        
        # Apply Gaussian blur to smooth the difference
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Normalize to 0-255 range
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap for visualization
        heatmap = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        query_color = cv2.cvtColor(query, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(query_color, 0.3, heatmap, 0.7, 0)
        
        return blended
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Engine loader — cached, only rebuilds when params change
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine(weights: str, conf: float, nms: int,
                ms: bool, reg: bool, asp: bool):
    from script_v2 import ForensicVerificationEngine

    signer = None
    try:
        from script_v2 import LogSigner, CRYPTO_AVAILABLE
        if CRYPTO_AVAILABLE:
            rp = ROOT / "keys" / "rsa_private.pem"
            ep = ROOT / "keys" / "ec_private.pem"
            if rp.exists() and ep.exists():
                signer = LogSigner(rp.read_bytes(), ep.read_bytes())
    except Exception:
        pass

    calibrator = None
    try:
        from script_v2 import FraudScoreCalibrator
        cp = ROOT / "calibrator.json"
        if cp.exists():
            calibrator = FraudScoreCalibrator.load(str(cp))
    except Exception:
        pass

    return ForensicVerificationEngine(
        weights_path=weights, conf_thresh=conf, nms_dist=nms,
        cuda=False,
        audit_log_dir=str(ROOT / "audit_logs"),
        chain_store_dir=str(ROOT / "audit_store"),
        run_multiscale=ms, run_region_verification=reg,
        run_anti_spoof=asp, base_size=(640, 480),
        signer=signer, calibrator=calibrator,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def sb_label(txt):
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
        f'color:var(--text-tertiary);padding:0.8rem 0 0.35rem">{txt}</div>',
        unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:0.2rem;">
        <span style="font-family:'Playfair Display',Georgia,serif;font-size:1.55rem;
                     font-weight:600;">FIAE</span>
    </div>
    <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;
                color:var(--text-tertiary);margin-bottom:2rem;">Forensic ID Authentication Engine</div>
    """, unsafe_allow_html=True)

    sb_label("Engine Configuration")
    weights_input = st.text_input("Weights file", value="superpoint_v1.pth",
                                   label_visibility="collapsed",
                                   help="Path to SuperPoint weights file")
    
    if Path(weights_input).exists():
        st.markdown(f'<div style="font-size:0.7rem;color:#059669;margin-top:-0.4rem;margin-bottom:0.4rem">✓ Weights file found</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.7rem;color:#DC2626;margin-top:-0.4rem;margin-bottom:0.4rem">✗ Weights file not found</div>', 
                   unsafe_allow_html=True)

    sb_label("Detection Parameters")
    conf_thresh     = st.slider("Keypoint confidence threshold", 0.001, 0.050, 0.003, 0.001, format="%.3f")
    nms_dist        = st.slider("Non-maximum suppression distance", 1, 8, 3)
    match_threshold = st.slider("Match threshold", 0.40, 0.99, 0.70, 0.01, format="%.2f")
    max_keypoints   = st.select_slider("Maximum keypoints",
                        options=[250, 500, 750, 1000, 1500, 2000], value=1000)

    sb_label("Analysis Modules")
    run_multiscale = st.checkbox("Multi-scale verification",  value=True)
    run_region     = st.checkbox("Region-based verification", value=True)
    run_anti_spoof = st.checkbox("Anti-spoof detection",      value=True)
    align_template = st.checkbox("Template alignment",        value=False)

    # ── Database ──────────────────────────────────────────────────────────
    sb_label("Reference Database")

    db_upload = st.file_uploader(
        "Upload reference images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="db_uploader",
    )

    # Write newly uploaded files into session db dir
    if db_upload:
        for f in db_upload:
            dest = DB_SESSION / f.name
            if not dest.exists():
                dest.write_bytes(f.getbuffer())

    db_dir, db_imgs = get_db_dir()
    source_label = "session uploads" if db_dir == DB_SESSION else "database/ folder"

    if db_imgs:
        items = "".join(
            f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;border-bottom:1px solid var(--border-light);font-size:0.8rem;color:var(--text-secondary);">'
            f'<div style="width:4px;height:4px;border-radius:50%;background:#4B5563;"></div>{p.name}</div>'
            for p in db_imgs[:25]
        )
        extra = (f'<div style="font-size:0.7rem;color:var(--text-tertiary);padding:0.25rem 0">'
                 f'+ {len(db_imgs)-25} more</div>') if len(db_imgs) > 25 else ""
        st.markdown(
            f'<div style="margin-bottom:0.3rem">{items}{extra}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#059669;margin-top:0.3rem">'
            f'{len(db_imgs)} image{"s" if len(db_imgs)>1 else ""} loaded '
            f'<span style="color:var(--text-tertiary)">({source_label})</span></div>',
            unsafe_allow_html=True)

        # Button to clear session uploads
        if db_dir == DB_SESSION and st.button("Clear uploaded database", use_container_width=True):
            shutil.rmtree(DB_SESSION, ignore_errors=True)
            DB_SESSION.mkdir(parents=True, exist_ok=True)
            st.rerun()
    else:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#D97706;line-height:1.6">'
            'No reference images loaded. Upload images above, or add them to the '
            '<code style="background:transparent">database/</code> folder.</div>',
            unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-top:1.5rem;padding-top:0.9rem;border-top:1px solid var(--border-light);'
        f'font-size:0.7rem;color:var(--text-tertiary);">{datetime.datetime.now().strftime("%d %b %Y, %H:%M")}</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Forensic ID Authentication Engine")
st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;margin:-0.4rem 0 2rem;">SuperPoint Architecture • 10-Layer Verification Pipeline</p>', unsafe_allow_html=True)

# ── Query upload ──────────────────────────────────────────────────────────────
upload_col, layer_col = st.columns([1.8, 1], gap="large")

with upload_col:
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
        f'color:var(--text-tertiary);margin-bottom:0.5rem">Query Document</div>',
        unsafe_allow_html=True)
    query_file = st.file_uploader(
        "Upload query document for verification", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", 
        key="query_upload")

with layer_col:
    st.markdown(f"""
    <div style="background:var(--bg-secondary);padding:1rem;border-radius:6px;border:1px solid var(--border-light);">
      <div style="font-weight:600;margin-bottom:0.5rem;color:var(--text-primary);">Verification Layers</div>
      <div class="data-row"><span class="data-key">Layer 1</span><span class="data-value">Descriptor Matching</span></div>
      <div class="data-row"><span class="data-key">Layer 2</span><span class="data-value">Geometric Consistency</span></div>
      <div class="data-row"><span class="data-key">Layer 3</span><span class="data-value">Tamper Localization</span></div>
      <div class="data-row"><span class="data-key">Layer 4</span><span class="data-value">Multi-Scale Analysis</span></div>
      <div class="data-row"><span class="data-key">Layer 5</span><span class="data-value">Region Verification</span></div>
      <div class="data-row"><span class="data-key">Layer 6</span><span class="data-value">Integrity Fingerprint</span></div>
      <div class="data-row"><span class="data-key">Layer 7</span><span class="data-value">Fraud Scoring</span></div>
      <div class="data-row"><span class="data-key">Layer 8</span><span class="data-value">Anti-Spoof Analysis</span></div>
      <div class="data-row"><span class="data-key">Layer 9</span><span class="data-value">Audit Log Generation</span></div>
      <div class="data-row"><span class="data-key">Layer 10</span><span class="data-value">Blockchain Hash</span></div>
    </div>""", unsafe_allow_html=True)

# Query preview
if query_file:
    raw = query_file.read(); query_file.seek(0)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
            f'color:var(--text-tertiary);margin:1rem 0 0.5rem">Query Preview</div>',
            unsafe_allow_html=True)
        st.image(raw, use_container_width=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Weights + readiness check ─────────────────────────────────────────────────
weights_path = str(ROOT / weights_input)
weights_ok   = Path(weights_path).exists()
_, db_imgs   = get_db_dir()
can_run      = bool(query_file and db_imgs and weights_ok)

btn_c, hint_c = st.columns([1, 5], gap="small")
with btn_c:
    run_btn = st.button("Run Verification", type="primary",
                         disabled=not can_run, use_container_width=True)
with hint_c:
    if not weights_ok:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#DC2626;padding-top:0.5rem">'
            f'Weights file not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not db_imgs:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#D97706;padding-top:0.5rem">'
            'No reference images. Upload images in the database section.</div>',
            unsafe_allow_html=True)
    elif not query_file:
        st.markdown(
            f'<div style="font-size:0.8rem;color:var(--text-tertiary);padding-top:0.5rem">'
            'Upload a query document to begin verification.</div>',
            unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
STEPS = [
    ("Loading verification engine",           "Init"),
    ("Extracting keypoints",     "Layer 1"),
    ("Searching database",       "Layer 1"),
    ("Geometric consistency",    "Layer 2"),
    ("Tamper localization",      "Layer 3"),
    ("Multi-scale verification", "Layer 4"),
    ("Region verification",      "Layer 5"),
    ("Integrity fingerprint",    "Layer 6"),
    ("Fraud scoring",            "Layer 7"),
    ("Anti-spoof analysis",      "Layer 8"),
    ("Writing audit log",        "Layer 9-10"),
]

if run_btn and can_run:
    # Save query to session dir
    q_path = Q_DIR / query_file.name
    q_path.write_bytes(query_file.getbuffer())

    db_dir_path, _ = get_db_dir()

    prog = st.empty()
    prog.markdown(steps_html(0, STEPS), unsafe_allow_html=True)

    # Load engine
    engine = None
    try:
        engine = load_engine(weights_path, conf_thresh, nms_dist,
                              run_multiscale, run_region, run_anti_spoof)
    except Exception:
        prog.empty()
        st.error("Engine failed to load.")
        st.code(traceback.format_exc(), language="python")
        st.stop()

    prog.markdown(steps_html(2, STEPS), unsafe_allow_html=True)

    # Verify
    report = None
    t0 = time.time()
    try:
        report = engine.verify(
            query_path=str(q_path),
            database_dir=str(db_dir_path),
            match_threshold=match_threshold,
            max_keypoints=max_keypoints,
            visualize=False,
            align_template=align_template,
        )
    except Exception:
        prog.empty()
        st.error("Verification failed.")
        st.code(traceback.format_exc(), language="python")
        st.stop()

    elapsed = time.time() - t0
    prog.markdown(steps_html(len(STEPS), STEPS), unsafe_allow_html=True)
    time.sleep(0.3)
    prog.empty()

    # Persist to session state
    st.session_state["report"] = report
    st.session_state["q_path"] = q_path
    st.session_state["elapsed"] = elapsed
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["report"] is not None:
    rep = st.session_state["report"]
    q_path = st.session_state["q_path"]
    elapsed = st.session_state["elapsed"]

    risk = rep.risk_level.value
    fraud_p = rep.fraud_probability * 100
    auth_p = rep.authenticity_score * 100
    desc_sim = rep.descriptor_similarity

    c_color, _ = risk_classes(risk)
    v_title, v_sub = verdict_text(risk, fraud_p)

    # Verdict banner
    bg_color = "#05966910" if risk == "LOW" else "#D9770610" if risk == "MEDIUM" else "#DC262610"
    border_color = "#059669" if risk == "LOW" else "#D97706" if risk == "MEDIUM" else "#DC2626"
    
    st.markdown(f"""
    <div style="background:{bg_color};border:1px solid {border_color};border-left:4px solid {border_color};
                padding:1.5rem 2rem;margin-bottom:2rem;border-radius:8px;">
        <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;
                    color:var(--text-tertiary);margin-bottom:0.5rem;">Verification Result | Session {rep.session_id}</div>
        <div style="font-size:2rem;font-weight:700;color:{border_color};">{v_title}</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:0.5rem;">{v_sub} — Processed in {elapsed:.2f} seconds</div>
    </div>""", unsafe_allow_html=True)

    # Score tiles
    geo_p = rep.geometric.inlier_ratio * 100 if rep.geometric else 0.0
    spoof_p = rep.anti_spoof.overall_spoof_probability * 100 if rep.anti_spoof else 0.0

    t1, t2, t3, t4 = st.columns(4, gap="medium")
    
    with t1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Authenticity Score</div>
            <div class="kpi-value" style="color:{pct_col(auth_p)};">{auth_p:.1f}%</div>
            <div class="kpi-sub">Composite weighted score</div>
        </div>""", unsafe_allow_html=True)
    
    with t2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Probability</div>
            <div class="kpi-value" style="color:{pct_col(100-fraud_p)};">{fraud_p:.1f}%</div>
            <div class="kpi-sub">Calibrated estimate</div>
        </div>""", unsafe_allow_html=True)
    
    with t3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Geometric Inliers</div>
            <div class="kpi-value" style="color:{pct_col(geo_p)};">{geo_p:.1f}%</div>
            <div class="kpi-sub">RANSAC homography</div>
        </div>""", unsafe_allow_html=True)
    
    with t4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Spoof Probability</div>
            <div class="kpi-value" style="color:{pct_col(100-spoof_p)};">{spoof_p:.1f}%</div>
            <div class="kpi-sub">Anti-spoof analysis</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Match Visualization</div>', unsafe_allow_html=True)

    if rep.best_match_path and Path(rep.best_match_path).exists():
        match_path = Path(rep.best_match_path)
        match_name = match_path.name

        col_q, col_m = st.columns(2, gap="large")
        with col_q:
            st.markdown(
                f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
                f'color:var(--text-tertiary);margin-bottom:0.5rem">Query Document</div>',
                unsafe_allow_html=True)
            st.image(str(q_path), use_container_width=True)

        with col_m:
            st.markdown(
                f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
                f'color:var(--text-tertiary);margin-bottom:0.5rem">Best Match — {match_name}</div>',
                unsafe_allow_html=True)
            st.image(str(match_path), use_container_width=True)

        # Side-by-side composite
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
            f'color:var(--text-tertiary);margin:1rem 0 0.5rem">Side-by-Side Comparison</div>',
            unsafe_allow_html=True)
        try:
            combo = side_by_side(q_path, match_path)
            st.image(combo, use_container_width=True)
        except Exception as e:
            st.caption(f"Comparison render failed: {e}")

        # Score strip
        st.markdown(f"""
        <div style="display:flex;gap:2rem;padding:1rem 1.5rem;background:var(--bg-secondary);border:1px solid var(--border-light);margin-top:0.5rem;font-size:0.85rem;border-radius:6px;">
          <span style="color:var(--text-tertiary);">Matched to:</span>
          <span style="font-weight:600;color:var(--text-primary);">{match_name}</span>
          <span style="color:var(--text-tertiary);margin-left:auto;">Similarity:</span>
          <span style="font-weight:600;color:{pct_col(desc_sim*100)};">{desc_sim:.4f}</span>
          <span style="color:var(--text-tertiary);">Inliers:</span>
          <span style="font-weight:600;color:{pct_col(geo_p)};">{geo_p:.1f}%</span>
          <span style="color:var(--text-tertiary);">Risk Level:</span>
          <span style="font-weight:600;color:{c_color};">{risk}</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="padding:3rem 2rem;border:1px dashed var(--border-light);border-radius:6px;text-align:center">
          <div style="font-size:1rem;color:#D97706;margin-bottom:0.5rem">No Match Found in Database</div>
          <div style="font-size:0.8rem;color:var(--text-tertiary);line-height:1.6">
            The query image did not meet the match threshold against any reference image.
            Try lowering the match threshold or adding more reference images.
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Tamper heatmap ────────────────────────────────────────────────────
    if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
        st.markdown('<div class="section-eyebrow">Tamper Detection Heatmap</div>', unsafe_allow_html=True)
        
        # Generate heatmap
        heatmap = generate_deviation_heatmap(q_path, rep.best_match_path)
        
        if heatmap is not None:
            hc1, hc2 = st.columns([2, 1], gap="large")
            with hc1:
                st.markdown(f'<div style="border:1px solid var(--border-light);border-radius:6px;overflow:hidden;background:var(--bg-secondary);padding:1rem;">', unsafe_allow_html=True)
                st.image(heatmap, use_container_width=True, caption="Deviation Heatmap (Red = High Deviation)")
                st.markdown('</div>', unsafe_allow_html=True)
            with hc2:
                st.markdown(f"""
                <div style="background:var(--bg-secondary);padding:1.5rem;border-radius:6px;border:1px solid var(--border-light);">
                  <div style="font-weight:600;margin-bottom:1rem;color:var(--text-primary);">Analysis</div>
                  <div class="data-row"><span class="data-key">Heatmap Generated</span><span class="data-value">✓</span></div>
                  <div class="data-row"><span class="data-key">Method</span><span class="data-value">Structural Deviation</span></div>
                  <div class="data-row"><span class="data-key">Color Scale</span><span class="data-value">Blue → Red</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                if rep.tamper and rep.tamper.suspicious_quadrants:
                    chips = "".join(
                        f'<span class="chip">{z}</span>' for z in rep.tamper.suspicious_quadrants)
                    st.markdown(
                        f'<div style="margin-top:1rem"><div style="font-weight:600;margin-bottom:0.5rem;color:var(--text-primary);">Flagged Zones</div>{chips}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="margin-top:1rem;padding:0.75rem;background:var(--green-soft);border:1px solid #059669;border-radius:4px;color:#059669;font-size:0.8rem">No suspicious zones detected</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="padding:1rem;background:var(--amber-soft);border:1px solid #D97706;border-radius:4px;color:#D97706;">Unable to generate heatmap</div>',
                unsafe_allow_html=True)

    # ── Detail columns ────────────────────────────────────────────────────
    LC, RC = st.columns([1.05, 1], gap="large")

    with LC:
        st.markdown('<div class="section-eyebrow">Descriptor Analysis</div>', unsafe_allow_html=True)
        mn = Path(rep.best_match_path).name if rep.best_match_path else "No match found"
        st.markdown(f"""
        <div class="data-row"><span class="data-key">Best match</span><span class="data-value">{mn}</span></div>
        <div class="data-row"><span class="data-key">Descriptor similarity</span>
             <span class="data-value" style="color:{pct_col(desc_sim*100)};">{desc_sim:.4f}</span></div>
        <div class="data-row"><span class="data-key">Risk classification</span>
             <span class="data-value" style="color:{c_color};font-weight:600">{risk}</span></div>
        <div class="data-row"><span class="data-key">Session ID</span>
             <span class="data-value" style="font-size:0.7rem">{rep.session_id}</span></div>
        """, unsafe_allow_html=True)

        if rep.geometric:
            g = rep.geometric
            st.markdown('<div class="section-eyebrow">Geometric Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Inlier count</span>
                 <span class="data-value">{g.inlier_count} keypoints</span></div>
            <div class="data-row"><span class="data-key">Inlier ratio</span>
                 <span class="data-value" style="color:{pct_col(g.inlier_ratio*100)};">{g.inlier_ratio*100:.1f}%</span></div>
            <div class="data-row"><span class="data-key">Reprojection error</span>
                 <span class="data-value">{g.reprojection_error:.2f} px</span></div>
            <div class="data-row"><span class="data-key">Homography stability</span>
                 <span class="data-value">{g.homography_stability:.4f}</span></div>
            """, unsafe_allow_html=True)

        if rep.tamper:
            t = rep.tamper
            zv = ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None"
            st.markdown('<div class="section-eyebrow">Tamper Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Unmatched keypoints</span>
                 <span class="data-value">{t.unmatched_ratio*100:.1f}%</span></div>
            <div class="data-row"><span class="data-key">Structural deviation</span>
                 <span class="data-value">{t.structural_deviation:.4f}</span></div>
            <div class="data-row"><span class="data-key">Suspicious zones</span>
                 <span class="data-value">{zv}</span></div>
            """, unsafe_allow_html=True)

        if rep.multiscale:
            m = rep.multiscale
            st.markdown('<div class="section-eyebrow">Multi-Scale Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Original</span><span class="data-value">{m.original_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Downscale (0.5x)</span><span class="data-value">{m.downscale_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Upscale (2x)</span><span class="data-value">{m.upscale_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Consistency score</span>
                 <span class="data-value">{m.consistency_score:.4f}</span></div>
            """, unsafe_allow_html=True)

    with RC:
        if rep.anti_spoof:
            a = rep.anti_spoof
            st.markdown('<div class="section-eyebrow">Anti-Spoof Detection</div>', unsafe_allow_html=True)
            for lbl, sc in [
                ("Moire Pattern",  a.moire_score),
                ("Photocopy",      a.photocopy_score),
                ("Screen Replay",  a.screen_replay_score),
                ("Print/Scan",     a.print_scan_score),
                ("Overall Spoof",  a.overall_spoof_probability),
            ]:
                pct  = sc * 100
                fill = bar_col(sc)
                st.markdown(f"""
                <div class="sb-row">
                  <span class="sb-lbl">{lbl}</span>
                  <div class="sb-bg"><div class="sb-fill"
                    style="width:{pct:.1f}%;background:{fill};"></div></div>
                  <span class="sb-val">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            if a.flags:
                chips = "".join(f'<span class="chip">{f}</span>' for f in a.flags)
                st.markdown(f'<div style="margin-top:0.7rem">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="margin-top:0.7rem;font-size:0.8rem;color:#059669;">No spoof flags triggered</div>',
                    unsafe_allow_html=True)

        if rep.region_results:
            st.markdown('<div class="section-eyebrow">Region Verification</div>', unsafe_allow_html=True)
            rows = ""
            for rr in rep.region_results:
                status_class = "status-verified" if rr.status == "VERIFIED" else "status-suspicious" if rr.status == "SUSPICIOUS" else "status-failed"
                rows += f"<tr><td>{rr.name.replace('_',' ').title()}</td>"
                rows += f"<td><span class='status-badge {status_class}'>{rr.status}</span></td>"
                rows += f"<td>{rr.confidence*100:.1f}%</td>"
                rows += f"<td>{rr.inlier_ratio*100:.1f}%</td>"
                rows += f"<td>{rr.matched_keypoints}</td></tr>"
            st.markdown(f"""
            <table class="region-table">
              <thead><tr><th>Zone</th><th>Status</th><th>Conf.</th><th>Inliers</th><th>Matches</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

        st.markdown('<div class="section-eyebrow">Integrity Fingerprint</div>', unsafe_allow_html=True)
        fp = rep.fingerprint
        ts = datetime.datetime.fromtimestamp(fp.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="data-row"><span class="data-key">Image SHA-256</span>
             <span class="data-value" style="font-size:0.7rem">{fp.image_sha256[:40]}...</span></div>
        <div class="data-row"><span class="data-key">Descriptor hash</span>
             <span class="data-value" style="font-size:0.7rem">{fp.descriptor_hash[:40]}...</span></div>
        <div class="data-row"><span class="data-key">Timestamp</span>
             <span class="data-value">{ts}</span></div>
        """, unsafe_allow_html=True)

    # ── Audit record ──────────────────────────────────────────────────────
    st.markdown('<div class="section-eyebrow">Audit Record</div>', unsafe_allow_html=True)
    with st.expander("View Full Audit Log"):
        if rep.audit_log_path and Path(rep.audit_log_path).exists():
            st.code(Path(rep.audit_log_path).read_text(), language="json")
        else:
            st.markdown(
                f'<div style="font-size:0.8rem;color:var(--text-tertiary);">Log not available</div>',
                unsafe_allow_html=True)

    sig = rep.audit_signature
    if sig:
        st.markdown(f"""
        <div class="hash-block">
          <span class="hash-label">RSA-PSS-SHA256-4096</span><br>
          <span class="hash-value">{sig.get('rsa_signature','')[:96]}...</span><br><br>
          <span class="hash-label">ECDSA-P384-SHA256</span><br>
          <span class="hash-value">{sig.get('ec_signature','')[:96]}...</span><br><br>
          <span class="hash-label">Payload SHA-256</span><br>
          <span class="hash-value">{sig.get('payload_sha256','')}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="font-size:0.8rem;color:var(--text-tertiary);padding:0.5rem 0">'
            'Log signing inactive — add key pairs to ./keys/ to enable.</div>',
            unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-eyebrow">Export</div>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3, gap="small")

    try:
        from script_v2 import build_log_payload
        with ex1:
            st.download_button(
                "Download Report",
                data=json.dumps(build_log_payload(rep), indent=2).encode(),
                file_name=f"fiae_report_{rep.session_id}.json",
                mime="application/json", use_container_width=True)
    except Exception:
        pass

    if rep.audit_log_path and Path(rep.audit_log_path).exists():
        with ex2:
            st.download_button(
                "Download Audit Log",
                data=Path(rep.audit_log_path).read_bytes(),
                file_name=f"audit_{rep.session_id}.json",
                mime="application/json", use_container_width=True)

    if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
        heatmap = generate_deviation_heatmap(q_path, rep.best_match_path)
        if heatmap is not None:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
            with ex3:
                st.download_button(
                    "Download Heatmap",
                    data=buf.getvalue(),
                    file_name=f"heatmap_{rep.session_id}.png",
                    mime="image/png", use_container_width=True)

    # Button to run a new verification
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run New Verification", use_container_width=False):
        st.session_state["report"] = None
        st.session_state["q_path"] = None
        st.session_state["elapsed"] = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["report"] is None and not run_btn:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:5rem 2rem;
                border:1px dashed var(--border-light);border-radius:6px;margin-top:0.5rem;">
      <div style="font-weight:600;font-size:1rem;color:var(--text-tertiary);margin-bottom:0.75rem;">
        No Verification in Progress
      </div>
      <div style="font-size:0.85rem;color:var(--text-secondary);text-align:center;
                  max-width:450px;line-height:1.8;">
        Upload reference images in the sidebar, upload a query document above,
        then click <strong>Run Verification</strong> to execute the complete
        10-layer forensic analysis pipeline.
      </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.7rem;font-weight:600;text-transform:uppercase;
            letter-spacing:0.06em;color:var(--text-tertiary);">
    Forensic ID Authentication Engine
</div>""", unsafe_allow_html=True)
