"""
Forensic ID Authentication Engine — Streamlit Interface
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
    initial_sidebar_state="expanded",
)

# ── Theme configuration ───────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default theme

def toggle_theme():
    if st.session_state.theme == "dark":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"

# Theme colors
def get_theme_colors():
    if st.session_state.theme == "dark":
        return {
            "bg_primary": "#0a0e14",
            "bg_secondary": "#05080c",
            "bg_tertiary": "#040710",
            "border": "#1e2a36",
            "border_light": "#0f161e",
            "text_primary": "#e6edf5",
            "text_secondary": "#9aaec5",
            "text_muted": "#4a6075",
            "accent_success": "#10b981",
            "accent_warning": "#f59e0b",
            "accent_danger": "#ef4444",
            "accent_info": "#3b82f6",
            "bg_success": "rgba(16, 185, 129, 0.1)",
            "bg_warning": "rgba(245, 158, 11, 0.1)",
            "bg_danger": "rgba(239, 68, 68, 0.1)",
            "bg_info": "rgba(59, 130, 246, 0.1)",
        }
    else:
        return {
            "bg_primary": "#f8fafc",
            "bg_secondary": "#f1f5f9",
            "bg_tertiary": "#ffffff",
            "border": "#cbd5e1",
            "border_light": "#e2e8f0",
            "text_primary": "#0f172a",
            "text_secondary": "#334155",
            "text_muted": "#64748b",
            "accent_success": "#059669",
            "accent_warning": "#d97706",
            "accent_danger": "#dc2626",
            "accent_info": "#2563eb",
            "bg_success": "rgba(5, 150, 105, 0.1)",
            "bg_warning": "rgba(217, 119, 6, 0.1)",
            "bg_danger": "rgba(220, 38, 38, 0.1)",
            "bg_info": "rgba(37, 99, 235, 0.1)",
        }

colors = get_theme_colors()

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

# ─────────────────────────────────────────────────────────────────────────────
# CSS with dynamic theming
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: {colors["bg_primary"]};
    color: {colors["text_primary"]};
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem 5rem; max-width: 1440px; }}

[data-testid="stSidebar"] {{
    background: {colors["bg_secondary"]};
    border-right: 1px solid {colors["border"]};
}}

[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.2rem; }}

/* masthead */
.mh {{
    display: flex;
    align-items: center;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid {colors["border_light"]};
    margin-bottom: 2rem;
}}

.mh-title {{
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: {colors["text_primary"]};
    letter-spacing: -0.02em;
}}

.mh-sub {{
    font-size: 0.7rem;
    color: {colors["text_muted"]};
    margin-top: 4px;
}}

.mh-pill {{
    margin-left: auto;
    font-size: 0.7rem;
    padding: 4px 12px;
    border: 1px solid {colors["border"]};
    color: {colors["accent_info"]};
    background: {colors["bg_info"]};
    border-radius: 20px;
}}

/* verdict banners */
.v-ok {{
    background: {colors["bg_success"]};
    border: 1px solid {colors["accent_success"]};
    border-left: 4px solid {colors["accent_success"]};
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border-radius: 8px;
}}

.v-med {{
    background: {colors["bg_warning"]};
    border: 1px solid {colors["accent_warning"]};
    border-left: 4px solid {colors["accent_warning"]};
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border-radius: 8px;
}}

.v-bad {{
    background: {colors["bg_danger"]};
    border: 1px solid {colors["accent_danger"]};
    border-left: 4px solid {colors["accent_danger"]};
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border-radius: 8px;
}}

.v-lbl {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {colors["text_muted"]};
    margin-bottom: 0.5rem;
}}

.v-h-ok {{ font-size: 2rem; font-weight: 700; color: {colors["accent_success"]}; }}
.v-h-med {{ font-size: 2rem; font-weight: 700; color: {colors["accent_warning"]}; }}
.v-h-bad {{ font-size: 2rem; font-weight: 700; color: {colors["accent_danger"]}; }}

.v-meta {{
    font-size: 0.85rem;
    color: {colors["text_secondary"]};
    margin-top: 0.5rem;
}}

/* score tiles */
.tiles {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: {colors["border_light"]};
    border: 1px solid {colors["border_light"]};
    margin-bottom: 2rem;
    border-radius: 8px;
    overflow: hidden;
}}

.tile {{
    background: {colors["bg_tertiary"]};
    padding: 1.2rem 1.5rem;
}}

.tile-lbl {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {colors["text_muted"]};
    margin-bottom: 0.5rem;
}}

.tile-val {{
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.2;
    color: {colors["text_primary"]};
}}

.tile-sub {{
    font-size: 0.7rem;
    color: {colors["text_muted"]};
    margin-top: 0.3rem;
}}

/* section header */
.sh {{
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {colors["text_secondary"]};
    padding: 0.5rem 0;
    border-bottom: 1px solid {colors["border_light"]};
    margin: 2rem 0 1rem;
}}

/* data rows */
.dr {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.6rem 0;
    border-bottom: 1px solid {colors["border_light"]};
    font-size: 0.85rem;
}}

.dr:last-child {{ border-bottom: none; }}
.dk {{ color: {colors["text_muted"]}; }}
.dv {{ color: {colors["text_secondary"]}; font-weight: 500; }}

/* spoof bars */
.sb-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0;
    font-size: 0.85rem;
}}

.sb-lbl {{
    color: {colors["text_muted"]};
    width: 120px;
    flex-shrink: 0;
}}

.sb-bg {{
    flex: 1;
    height: 6px;
    background: {colors["border_light"]};
    position: relative;
    border-radius: 3px;
    overflow: hidden;
}}

.sb-fill {{
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    transition: width 0.3s ease;
}}

.sb-val {{
    color: {colors["text_secondary"]};
    width: 50px;
    text-align: right;
    font-weight: 500;
}}

/* region table */
.rt {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}}

.rt th {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {colors["text_muted"]};
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid {colors["border_light"]};
    text-align: left;
}}

.rt td {{
    padding: 0.7rem 0.8rem;
    border-bottom: 1px solid {colors["border_light"]};
    color: {colors["text_secondary"]};
}}

.rt tr:last-child td {{ border-bottom: none; }}

.bv {{
    color: {colors["accent_success"]};
    background: {colors["bg_success"]};
    border: 1px solid {colors["accent_success"]};
}}

.bs {{
    color: {colors["accent_warning"]};
    background: {colors["bg_warning"]};
    border: 1px solid {colors["accent_warning"]};
}}

.bf {{
    color: {colors["accent_danger"]};
    background: {colors["bg_danger"]};
    border: 1px solid {colors["accent_danger"]};
}}

.bdg {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
}}

/* chips */
.chip {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 10px;
    margin: 2px 4px 2px 0;
    color: {colors["accent_warning"]};
    background: {colors["bg_warning"]};
    border: 1px solid {colors["accent_warning"]};
    border-radius: 4px;
}}

/* audit block */
.ab {{
    background: {colors["bg_secondary"]};
    border: 1px solid {colors["border"]};
    padding: 1.2rem 1.5rem;
    font-size: 0.8rem;
    line-height: 1.6;
    word-break: break-all;
    border-radius: 6px;
}}

.ak {{ color: {colors["text_muted"]}; }}
.av {{ color: {colors["text_secondary"]}; }}
.as {{ color: {colors["text_muted"]}; font-size: 0.7rem; }}

/* step list */
.step {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0;
    font-size: 0.85rem;
}}

.dot-done {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {colors["accent_success"]};
    flex-shrink: 0;
}}

.dot-live {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {colors["accent_info"]};
    flex-shrink: 0;
    animation: blink 1s infinite;
}}

.dot-wait {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {colors["border_light"]};
    flex-shrink: 0;
}}

@keyframes blink {{ 0%,100%{{ opacity:1 }} 50%{{ opacity:0.3 }} }}

/* match strip */
.mstrip {{
    display: flex;
    gap: 2rem;
    padding: 1rem 1.5rem;
    background: {colors["bg_secondary"]};
    border: 1px solid {colors["border"]};
    margin-top: 0.5rem;
    font-size: 0.85rem;
    flex-wrap: wrap;
    border-radius: 6px;
}}

.ms-k {{ color: {colors["text_muted"]}; }}
.ms-v {{ font-weight: 600; color: {colors["text_primary"]}; }}

/* db file list */
.dbl {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.8rem;
    color: {colors["text_secondary"]};
    border-bottom: 1px solid {colors["border_light"]};
}}

.dbl:last-child {{ border-bottom: none; }}
.dbl-dot {{
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: {colors["accent_info"]};
    flex-shrink: 0;
}}

/* upload zone */
[data-testid="stFileUploadDropzone"] {{
    background: {colors["bg_secondary"]} !important;
    border: 1px dashed {colors["border"]} !important;
    border-radius: 6px !important;
}}

[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {colors["accent_info"]} !important;
}}

/* buttons */
.stButton > button {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    background: {colors["bg_secondary"]};
    border: 1px solid {colors["border"]};
    color: {colors["text_primary"]};
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
    border-radius: 6px;
}}

.stButton > button:hover {{
    background: {colors["bg_info"]};
    border-color: {colors["accent_info"]};
    color: {colors["accent_info"]};
}}

.stButton > button[kind="primary"] {{
    background: {colors["bg_info"]};
    border-color: {colors["accent_info"]};
    color: {colors["accent_info"]};
}}

/* form inputs */
.stTextInput > div > div,
.stSelectbox > div > div,
.stNumberInput > div > div {{
    background: {colors["bg_secondary"]} !important;
    border: 1px solid {colors["border"]} !important;
    border-radius: 6px !important;
    color: {colors["text_primary"]} !important;
}}

.stTextInput input,
.stSelectbox select,
.stNumberInput input {{
    color: {colors["text_primary"]} !important;
}}

.stCheckbox > label {{
    font-size: 0.85rem;
    color: {colors["text_secondary"]};
}}

.stExpander {{
    border: 1px solid {colors["border"]} !important;
    border-radius: 6px !important;
    background: {colors["bg_secondary"]} !important;
}}

details summary p {{
    font-size: 0.8rem !important;
    color: {colors["text_primary"]} !important;
}}

/* theme toggle button */
.theme-toggle {{
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 999;
}}

/* heatmap container */
.heatmap-container {{
    border: 1px solid {colors["border"]};
    border-radius: 6px;
    overflow: hidden;
    background: {colors["bg_secondary"]};
    padding: 1rem;
}}

/* color classes */
.cg {{ color: {colors["accent_success"]}; }}
.ca {{ color: {colors["accent_warning"]}; }}
.cr {{ color: {colors["accent_danger"]}; }}
.cb {{ color: {colors["accent_info"]}; }}
.cd {{ color: {colors["text_muted"]}; }}
</style>
""", unsafe_allow_html=True)

# Theme toggle button
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light", 
                 key="theme_toggle", help="Toggle theme"):
        toggle_theme()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct_col(v: float) -> str:
    return "cg" if v >= 75 else ("ca" if v >= 45 else "cr")

def bar_col(s: float) -> str:
    return colors["accent_success"] if s < .25 else (colors["accent_warning"] if s < .55 else colors["accent_danger"])

def risk_classes(r: str):
    return {
        "LOW":    ("cg", "v-ok",  "v-h-ok"),
        "MEDIUM": ("ca", "v-med", "v-h-med"),
    }.get(r, ("cr", "v-bad", "v-h-bad"))

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
            dot, tc = "dot-done", "cg"
        elif i == active:
            dot, tc = "dot-live", "cb"
        else:
            dot, tc = "dot-wait", "cd"
        rows += (f'<div class="step"><div class="{dot}"></div>'
                 f'<span class="{tc}" style="min-width:70px">{tag}</span>'
                 f'<span style="color:{colors["text_secondary"]}">{lbl}</span></div>')
    return f'<div style="padding:.5rem 0">{rows}</div>'

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
    out   = Image.new("RGB", (total, h), colors["bg_secondary"])
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
        f'color:{colors["text_muted"]};padding:0.8rem 0 0.35rem">{txt}</div>',
        unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom:1rem;border-bottom:1px solid {colors['border']}">
        <div style="font-weight:700;font-size:1.2rem;color:{colors['text_primary']}">
            Forensic ID Authentication Engine
        </div>
        <div style="font-size:0.7rem;color:{colors['text_muted']};margin-top:4px">
            Advanced Document Verification System
        </div>
    </div>""", unsafe_allow_html=True)

    sb_label("Engine Configuration")
    weights_input = st.text_input("Weights file", value="superpoint_v1.pth",
                                   label_visibility="collapsed",
                                   help="Path to SuperPoint weights file")
    
    if Path(weights_input).exists():
        st.markdown(f'<div style="font-size:0.7rem;color:{colors["accent_success"]};margin-top:-0.4rem;margin-bottom:0.4rem">✓ Weights file found</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.7rem;color:{colors["accent_danger"]};margin-top:-0.4rem;margin-bottom:0.4rem">✗ Weights file not found</div>', 
                   unsafe_allow_html=True)

    sb_label("Detection Parameters")
    conf_thresh     = st.slider("Keypoint confidence threshold", 0.001, 0.050, 0.003, 0.001, format="%.3f")
    nms_dist        = st.slider("Non-maximum suppression distance (pixels)", 1, 8, 3)
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
            f'<div class="dbl"><div class="dbl-dot"></div>{p.name}</div>'
            for p in db_imgs[:25]
        )
        extra = (f'<div style="font-size:0.7rem;color:{colors["text_muted"]};padding:0.25rem 0">'
                 f'+ {len(db_imgs)-25} more</div>') if len(db_imgs) > 25 else ""
        st.markdown(
            f'<div style="margin-bottom:0.3rem">{items}{extra}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.75rem;color:{colors["accent_success"]};margin-top:0.3rem">'
            f'{len(db_imgs)} image{"s" if len(db_imgs)>1 else ""} loaded '
            f'<span style="color:{colors["text_muted"]}">({source_label})</span></div>',
            unsafe_allow_html=True)

        # Button to clear session uploads
        if db_dir == DB_SESSION and st.button("Clear uploaded database", use_container_width=True):
            shutil.rmtree(DB_SESSION, ignore_errors=True)
            DB_SESSION.mkdir(parents=True, exist_ok=True)
            st.rerun()
    else:
        st.markdown(
            f'<div style="font-size:0.75rem;color:{colors["accent_warning"]};line-height:1.6">'
            'No reference images loaded. Upload images above, or add them to the '
            '<code style="background:transparent">database/</code> folder.</div>',
            unsafe_allow_html=True)

    st.markdown(
        f'<div style="margin-top:1.5rem;padding-top:0.9rem;border-top:1px solid {colors["border_light"]};'
        f'font-size:0.7rem;color:{colors["text_muted"]}">v2.0 — Forensic Edition</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="mh">
  <div>
    <div class="mh-title">Forensic ID Authentication Engine</div>
    <div class="mh-sub">SuperPoint Architecture • 10-Layer Verification Pipeline</div>
  </div>
  <div class="mh-pill">Production Ready</div>
</div>
""", unsafe_allow_html=True)

# ── Query upload ──────────────────────────────────────────────────────────────
upload_col, layer_col = st.columns([1.8, 1], gap="large")

with upload_col:
    st.markdown(
        f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
        f'color:{colors["text_muted"]};margin-bottom:0.5rem">Query Document</div>',
        unsafe_allow_html=True)
    query_file = st.file_uploader(
        "Upload query document for verification", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", 
        key="query_upload")

with layer_col:
    st.markdown(f"""
    <div style="background:{colors['bg_secondary']};padding:1rem;border-radius:6px;border:1px solid {colors['border']}">
      <div style="font-weight:600;margin-bottom:0.5rem;color:{colors['text_primary']}">Verification Layers</div>
      <div class="dr"><span class="dk">Layer 1</span><span class="dv">Descriptor Matching</span></div>
      <div class="dr"><span class="dk">Layer 2</span><span class="dv">Geometric Consistency</span></div>
      <div class="dr"><span class="dk">Layer 3</span><span class="dv">Tamper Localization</span></div>
      <div class="dr"><span class="dk">Layer 4</span><span class="dv">Multi-Scale Analysis</span></div>
      <div class="dr"><span class="dk">Layer 5</span><span class="dv">Region Verification</span></div>
      <div class="dr"><span class="dk">Layer 6</span><span class="dv">Integrity Fingerprint</span></div>
      <div class="dr"><span class="dk">Layer 7</span><span class="dv">Fraud Scoring</span></div>
      <div class="dr"><span class="dk">Layer 8</span><span class="dv">Anti-Spoof Analysis</span></div>
      <div class="dr"><span class="dk">Layer 9</span><span class="dv">Audit Log Generation</span></div>
      <div class="dr"><span class="dk">Layer 10</span><span class="dv">Blockchain Hash</span></div>
    </div>""", unsafe_allow_html=True)

# Query preview
if query_file:
    raw = query_file.read(); query_file.seek(0)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
            f'color:{colors["text_muted"]};margin:1rem 0 0.5rem">Query Preview</div>',
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
            f'<div style="font-size:0.8rem;color:{colors["accent_danger"]};padding-top:0.5rem">'
            f'Weights file not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not db_imgs:
        st.markdown(
            f'<div style="font-size:0.8rem;color:{colors["accent_warning"]};padding-top:0.5rem">'
            'No reference images. Upload images in the database section.</div>',
            unsafe_allow_html=True)
    elif not query_file:
        st.markdown(
            f'<div style="font-size:0.8rem;color:{colors["text_muted"]};padding-top:0.5rem">'
            'Upload a query document to begin verification.</div>',
            unsafe_allow_html=True)

st.markdown(
    f'<div style="height:1px;background:{colors["border_light"]};margin:1.5rem 0 2rem"></div>',
    unsafe_allow_html=True)


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
    st.session_state["report"]  = report
    st.session_state["q_path"]  = q_path
    st.session_state["elapsed"] = elapsed
    st.rerun()  # force clean render of results without the progress list


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if "report" in st.session_state:
    rep     = st.session_state["report"]
    q_path  = st.session_state["q_path"]
    elapsed = st.session_state["elapsed"]

    risk     = rep.risk_level.value
    fraud_p  = rep.fraud_probability * 100
    auth_p   = rep.authenticity_score * 100
    desc_sim = rep.descriptor_similarity

    c_cls, v_cls, h_cls = risk_classes(risk)
    v_title, v_sub = verdict_text(risk, fraud_p)

    # Verdict banner
    st.markdown(f"""
    <div class="{v_cls}">
      <div class="v-lbl">Verification Result | Session {rep.session_id}</div>
      <div class="{h_cls}">{v_title}</div>
      <div class="v-meta">{v_sub} — Processed in {elapsed:.2f} seconds</div>
    </div>""", unsafe_allow_html=True)

    # Score tiles
    geo_p   = rep.geometric.inlier_ratio * 100 if rep.geometric else 0.0
    spoof_p = rep.anti_spoof.overall_spoof_probability * 100 if rep.anti_spoof else 0.0

    st.markdown(f"""
    <div class="tiles">
      <div class="tile">
        <div class="tile-lbl">Authenticity Score</div>
        <div class="tile-val">{auth_p:.1f}%</div>
        <div class="tile-sub">Composite weighted score</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Fraud Probability</div>
        <div class="tile-val">{fraud_p:.1f}%</div>
        <div class="tile-sub">Calibrated estimate</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Geometric Inliers</div>
        <div class="tile-val">{geo_p:.1f}%</div>
        <div class="tile-sub">RANSAC homography</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Spoof Probability</div>
        <div class="tile-val">{spoof_p:.1f}%</div>
        <div class="tile-sub">Anti-spoof analysis</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── MATCH VISUALISATION — front and centre ────────────────────────────
    st.markdown('<div class="sh">Match Visualization</div>', unsafe_allow_html=True)

    if rep.best_match_path and Path(rep.best_match_path).exists():
        match_path = Path(rep.best_match_path)
        match_name = match_path.name

        col_q, col_m = st.columns(2, gap="large")
        with col_q:
            st.markdown(
                f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
                f'color:{colors["text_muted"]};margin-bottom:0.5rem">Query Document</div>',
                unsafe_allow_html=True)
            st.image(str(q_path), use_container_width=True)

        with col_m:
            st.markdown(
                f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
                f'color:{colors["text_muted"]};margin-bottom:0.5rem">Best Match — {match_name}</div>',
                unsafe_allow_html=True)
            st.image(str(match_path), use_container_width=True)

        # Side-by-side composite
        st.markdown(
            f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;'
            f'color:{colors["text_muted"]};margin:1rem 0 0.5rem">Side-by-Side Comparison</div>',
            unsafe_allow_html=True)
        try:
            combo = side_by_side(q_path, match_path)
            st.image(combo, use_container_width=True)
        except Exception as e:
            st.caption(f"Comparison render failed: {e}")

        # Score strip
        st.markdown(f"""
        <div class="mstrip">
          <span class="ms-k">Matched to:</span>
          <span class="ms-v">{match_name}</span>
          <span class="ms-k" style="margin-left:auto">Similarity:</span>
          <span class="ms-v {pct_col(desc_sim*100)}">{desc_sim:.4f}</span>
          <span class="ms-k">Inliers:</span>
          <span class="ms-v {pct_col(geo_p)}">{geo_p:.1f}%</span>
          <span class="ms-k">Risk Level:</span>
          <span class="ms-v {c_cls}">{risk}</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="padding:3rem 2rem;border:1px dashed {colors['border']};border-radius:6px;text-align:center">
          <div style="font-size:1rem;color:{colors['accent_warning']};margin-bottom:0.5rem">
            No Match Found in Database
          </div>
          <div style="font-size:0.8rem;color:{colors['text_muted']};line-height:1.6">
            The query image did not meet the match threshold against any reference image.
            Try lowering the match threshold or adding more reference images.
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Tamper heatmap ────────────────────────────────────────────────────
    if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
        st.markdown('<div class="sh">Tamper Detection Heatmap</div>', unsafe_allow_html=True)
        
        # Generate heatmap
        heatmap = generate_deviation_heatmap(q_path, rep.best_match_path)
        
        if heatmap is not None:
            hc1, hc2 = st.columns([2, 1], gap="large")
            with hc1:
                st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                st.image(heatmap, use_container_width=True, caption="Deviation Heatmap (Red = High Deviation)")
                st.markdown('</div>', unsafe_allow_html=True)
            with hc2:
                st.markdown(f"""
                <div style="background:{colors['bg_secondary']};padding:1.5rem;border-radius:6px;border:1px solid {colors['border']}">
                  <div style="font-weight:600;margin-bottom:1rem;color:{colors['text_primary']}">Analysis</div>
                  <div class="dr"><span class="dk">Heatmap Generated</span><span class="dv">✓</span></div>
                  <div class="dr"><span class="dk">Method</span><span class="dv">Structural Deviation</span></div>
                  <div class="dr"><span class="dk">Color Scale</span><span class="dv">Blue (Low) → Red (High)</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                if rep.tamper and rep.tamper.suspicious_quadrants:
                    chips = "".join(
                        f'<span class="chip" style="background:{colors["bg_danger"]};color:{colors["accent_danger"]}">{z}</span>' 
                        for z in rep.tamper.suspicious_quadrants)
                    st.markdown(
                        f'<div style="margin-top:1rem"><div style="font-weight:600;margin-bottom:0.5rem;color:{colors["text_primary"]}">Flagged Zones</div>{chips}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="margin-top:1rem;padding:0.75rem;background:{colors["bg_success"]};border:1px solid {colors["accent_success"]};border-radius:4px;color:{colors["accent_success"]};font-size:0.8rem">No suspicious zones detected</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="padding:1rem;background:{colors["bg_warning"]};border:1px solid {colors["accent_warning"]};border-radius:4px;color:{colors["accent_warning"]}">Unable to generate heatmap</div>',
                unsafe_allow_html=True)

    # ── Detail columns ────────────────────────────────────────────────────
    LC, RC = st.columns([1.05, 1], gap="large")

    with LC:
        st.markdown('<div class="sh">Descriptor Analysis</div>', unsafe_allow_html=True)
        mn = Path(rep.best_match_path).name if rep.best_match_path else "No match found"
        st.markdown(f"""
        <div class="dr"><span class="dk">Best match</span><span class="dv">{mn}</span></div>
        <div class="dr"><span class="dk">Descriptor similarity</span>
             <span class="dv">{desc_sim:.4f}</span></div>
        <div class="dr"><span class="dk">Risk classification</span>
             <span class="dv {c_cls}" style="font-weight:600">{risk}</span></div>
        <div class="dr"><span class="dk">Session ID</span>
             <span class="dv" style="font-size:0.7rem">{rep.session_id}</span></div>
        """, unsafe_allow_html=True)

        if rep.geometric:
            g = rep.geometric
            st.markdown('<div class="sh">Geometric Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Inlier count</span>
                 <span class="dv">{g.inlier_count} keypoints</span></div>
            <div class="dr"><span class="dk">Inlier ratio</span>
                 <span class="dv">{g.inlier_ratio*100:.1f}%</span></div>
            <div class="dr"><span class="dk">Reprojection error</span>
                 <span class="dv">{g.reprojection_error:.2f} px</span></div>
            <div class="dr"><span class="dk">Homography stability</span>
                 <span class="dv">{g.homography_stability:.4f}</span></div>
            """, unsafe_allow_html=True)

        if rep.tamper:
            t = rep.tamper
            zv = ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None"
            st.markdown('<div class="sh">Tamper Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Unmatched keypoints</span>
                 <span class="dv">{t.unmatched_ratio*100:.1f}%</span></div>
            <div class="dr"><span class="dk">Structural deviation</span>
                 <span class="dv">{t.structural_deviation:.4f}</span></div>
            <div class="dr"><span class="dk">Suspicious zones</span>
                 <span class="dv">{zv}</span></div>
            """, unsafe_allow_html=True)

        if rep.multiscale:
            m = rep.multiscale
            st.markdown('<div class="sh">Multi-Scale Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Original</span><span class="dv">{m.original_score:.4f}</span></div>
            <div class="dr"><span class="dk">Downscale (0.5x)</span><span class="dv">{m.downscale_score:.4f}</span></div>
            <div class="dr"><span class="dk">Upscale (2x)</span><span class="dv">{m.upscale_score:.4f}</span></div>
            <div class="dr"><span class="dk">Consistency score</span>
                 <span class="dv">{m.consistency_score:.4f}</span></div>
            """, unsafe_allow_html=True)

    with RC:
        if rep.anti_spoof:
            a = rep.anti_spoof
            st.markdown('<div class="sh">Anti-Spoof Detection</div>', unsafe_allow_html=True)
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
                    style="width:{pct:.1f}%;background:{fill}"></div></div>
                  <span class="sb-val">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            if a.flags:
                chips = "".join(f'<span class="chip">{f}</span>' for f in a.flags)
                st.markdown(f'<div style="margin-top:0.7rem">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="margin-top:0.7rem;font-size:0.8rem;color:{colors["accent_success"]}">No spoof flags triggered</div>',
                    unsafe_allow_html=True)

        if rep.region_results:
            st.markdown('<div class="sh">Region Verification</div>', unsafe_allow_html=True)
            rows = ""
            for rr in rep.region_results:
                bc = {"VERIFIED":"bv","SUSPICIOUS":"bs","FAILED":"bf"}.get(rr.status,"bf")
                rows += (f"<tr><td>{rr.name.replace('_',' ').title()}</td>"
                         f"<td><span class='bdg {bc}'>{rr.status}</span></td>"
                         f"<td>{rr.confidence*100:.1f}%</td>"
                         f"<td>{rr.inlier_ratio*100:.1f}%</td>"
                         f"<td>{rr.matched_keypoints}</td></tr>")
            st.markdown(f"""
            <table class="rt">
              <thead><tr><th>Zone</th><th>Status</th><th>Confidence</th>
              <th>Inliers</th><th>Matches</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">Integrity Fingerprint</div>', unsafe_allow_html=True)
        fp = rep.fingerprint
        ts = datetime.datetime.fromtimestamp(fp.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="dr"><span class="dk">Image SHA-256</span>
             <span class="dv" style="font-size:0.7rem">{fp.image_sha256[:40]}...</span></div>
        <div class="dr"><span class="dk">Descriptor hash</span>
             <span class="dv" style="font-size:0.7rem">{fp.descriptor_hash[:40]}...</span></div>
        <div class="dr"><span class="dk">Timestamp</span>
             <span class="dv">{ts}</span></div>
        """, unsafe_allow_html=True)

    # ── Audit record ──────────────────────────────────────────────────────
    st.markdown('<div class="sh">Audit Record</div>', unsafe_allow_html=True)
    with st.expander("View Full Audit Log", expanded=False):
        if rep.audit_log_path and Path(rep.audit_log_path).exists():
            st.code(Path(rep.audit_log_path).read_text(), language="json")
        else:
            st.markdown(
                f'<div style="font-size:0.8rem;color:{colors["text_muted"]}">Log not available</div>',
                unsafe_allow_html=True)

    sig = rep.audit_signature
    if sig:
        st.markdown(f"""
        <div class="ab">
          <span class="ak">RSA-PSS-SHA256-4096</span><br>
          <span class="as">{sig.get('rsa_signature','')[:96]}...</span><br><br>
          <span class="ak">ECDSA-P384-SHA256</span><br>
          <span class="as">{sig.get('ec_signature','')[:96]}...</span><br><br>
          <span class="ak">Payload SHA-256</span>&nbsp;
          <span class="av">{sig.get('payload_sha256','')}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="font-size:0.8rem;color:{colors["text_muted"]};padding:0.5rem 0">'
            'Log signing inactive — add key pairs to ./keys/ to enable.</div>',
            unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Export</div>', unsafe_allow_html=True)
    ex1, ex2, ex3, _ = st.columns([1, 1, 1, 2], gap="small")

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
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("Run New Verification", use_container_width=False):
        del st.session_state["report"]
        del st.session_state["q_path"]
        del st.session_state["elapsed"]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:5rem 2rem;
                border:1px dashed {colors['border']};border-radius:6px;margin-top:0.5rem">
      <div style="font-weight:600;font-size:1rem;color:{colors['text_muted']};margin-bottom:0.75rem">
        No Verification in Progress
      </div>
      <div style="font-size:0.85rem;color:{colors['text_secondary']};text-align:center;
                  max-width:450px;line-height:1.8">
        Upload reference images in the sidebar, upload a query document above,
        then click <strong>Run Verification</strong> to execute the complete
        10-layer forensic analysis pipeline.
      </div>
    </div>""", unsafe_allow_html=True)
