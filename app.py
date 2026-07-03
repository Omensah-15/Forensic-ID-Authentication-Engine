"""
Forensic Document Verification — Demo Interface
=================================================


Run:
    streamlit run app.py

Requires script_v3.py in the same directory.
"""

from __future__ import annotations

import io, os, sys, json, time, shutil, datetime, tempfile, traceback
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forensic Document Verification",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2; CV2_OK = True
except ImportError:
    CV2_OK = False

# ─── Session dirs ──────────────────────────────────────────────────────────────
if "session_dir" not in st.session_state:
    st.session_state["session_dir"] = Path(tempfile.mkdtemp(prefix="fdv_"))
SESSION_DIR = st.session_state["session_dir"]
DB_SESSION  = SESSION_DIR / "database"
Q_DIR       = SESSION_DIR / "query"
DB_REPO     = ROOT / "database"
DB_SESSION.mkdir(parents=True, exist_ok=True)
Q_DIR.mkdir(parents=True, exist_ok=True)

for k, v in [("report", None), ("q_path", None), ("elapsed", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Palette ───────────────────────────────────────────────────────────────────
COL_BG        = "#f8fafc"
COL_SURFACE   = "#ffffff"
COL_BORDER    = "#e2e8f0"
COL_TEXT      = "#0f172a"
COL_TEXT_SOFT = "#475569"
COL_TEXT_MUTE = "#94a3b8"
COL_PRIMARY   = "#2563eb"
COL_SUCCESS   = "#16a34a"
COL_WARNING   = "#d97706"
COL_DANGER    = "#dc2626"

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:        #f8fafc;
    --surface:   #ffffff;
    --border:    #e2e8f0;
    --border-2:  #cbd5e1;
    --text:      #0f172a;
    --text-soft: #475569;
    --text-mute: #94a3b8;
    --primary:   #2563eb;
    --primary-soft: #eff6ff;
    --success:   #16a34a;
    --success-soft: #f0fdf4;
    --warning:   #d97706;
    --warning-soft: #fffbeb;
    --danger:    #dc2626;
    --danger-soft: #fef2f2;
    --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'JetBrains Mono', 'SF Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 4rem 2rem !important; max-width: 1500px; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.4rem 1.1rem !important; }

.stTextInput label, .stSlider label, .stCheckbox label,
.stSelectbox label, .stNumberInput label, .stFileUploader label {
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--text-soft) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--primary) !important; border-color: var(--primary) !important;
}
.stSlider > div > div > div > div { background: var(--primary) !important; }

[data-testid="stCheckbox"] label span:last-child {
    color: var(--text-soft) !important; font-family: var(--sans) !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    font-family: var(--sans) !important;
    font-size: 0.85rem !important; font-weight: 600 !important;
    border-radius: 8px !important; padding: 0.55rem 1.3rem !important;
    transition: all 0.15s !important; border: 1px solid var(--border-2) !important;
}
.stButton > button[kind="primary"] {
    background: var(--primary) !important; color: #ffffff !important;
    border: 1px solid var(--primary) !important;
    box-shadow: 0 1px 2px rgba(37,99,235,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important; border-color: #1d4ed8 !important;
}
.stButton > button:not([kind="primary"]) {
    background: var(--surface) !important; color: var(--text-soft) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--primary) !important; color: var(--primary) !important;
}
.stButton > button:disabled { opacity: 0.4 !important; }

.stDownloadButton > button {
    font-family: var(--sans) !important; font-size: 0.80rem !important;
    font-weight: 600 !important; border-radius: 8px !important;
    background: var(--surface) !important; color: var(--primary) !important;
    border: 1px solid var(--border-2) !important; padding: 0.5rem 0.9rem !important;
}
.stDownloadButton > button:hover { border-color: var(--primary) !important; }

[data-testid="stFileUploadDropzone"] {
    background: #fafbfc !important;
    border: 1.5px dashed var(--border-2) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--primary) !important; }
[data-testid="stFileUploadDropzone"] * { color: var(--text-mute) !important; }

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0.2rem !important; background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--sans) !important; font-size: 0.85rem !important;
    font-weight: 600 !important; color: var(--text-mute) !important;
    border: none !important; padding: 0.6rem 1.1rem !important;
    background: transparent !important; border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text-soft) !important; }

[data-testid="stProgress"] > div > div { background: var(--primary) !important; }
[data-testid="stProgress"] > div { background: var(--border) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--sans) !important; font-size: 0.82rem !important;
    font-weight: 600 !important; color: var(--text-soft) !important;
}

hr { border-color: var(--border) !important; margin: 1.3rem 0 !important; }
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Custom Components ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 1.4rem; background: var(--surface);
    border: 1px solid var(--border); border-radius: 14px; margin-bottom: 1.4rem;
}
.topbar-title { display: flex; flex-direction: column; gap: 2px; }
.topbar-h1 { font-family: var(--sans); font-size: 1.28rem; font-weight: 800; color: var(--text); letter-spacing: -0.01em; }
.topbar-sub { font-family: var(--sans); font-size: 0.80rem; font-weight: 500; color: var(--text-mute); }
.status-row { display: flex; align-items: center; gap: 1.3rem; }
.status-pill {
    display: flex; align-items: center; gap: 0.45rem;
    font-family: var(--sans); font-size: 0.76rem; font-weight: 600; color: var(--text-soft);
    padding: 0.32rem 0.7rem; background: var(--bg); border-radius: 999px; border: 1px solid var(--border);
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-ready { background: var(--success); }
.dot-wait  { background: var(--warning); }

.panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.3rem; margin-bottom: 1rem;
}
.panel-title {
    font-family: var(--sans); font-size: 0.95rem; font-weight: 700; color: var(--text);
    margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.5rem;
}
.panel-title-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); }

.sb-brand { padding-bottom: 1rem; margin-bottom: 1.1rem; border-bottom: 1px solid var(--border); }
.sb-brand-mark { font-family: var(--sans); font-size: 1.05rem; font-weight: 800; color: var(--text); }
.sb-brand-sub  { font-family: var(--sans); font-size: 0.72rem; font-weight: 500; color: var(--text-mute); margin-top: 2px; }
.sb-sect {
    font-family: var(--sans); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--text-mute);
    padding: 0.9rem 0 0.45rem 0; border-bottom: 1px solid var(--border); margin-bottom: 0.65rem;
}

.verdict-card {
    position: relative; padding: 1.5rem 1.7rem; border-radius: 14px;
    border: 1px solid var(--border); border-left: 5px solid var(--primary);
    background: var(--surface); margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;
}
.verdict-classify { font-family: var(--sans); font-size: 0.74rem; font-weight: 600; color: var(--text-mute); margin-bottom: 0.4rem; }
.verdict-title { font-family: var(--sans); font-size: 1.55rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.15; margin-bottom: 0.4rem; }
.verdict-body { font-family: var(--sans); font-size: 0.88rem; color: var(--text-soft); line-height: 1.55; max-width: 520px; }
.verdict-meta { font-family: var(--mono); font-size: 0.72rem; color: var(--text-mute); margin-top: 0.6rem; }

.kpi {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.95rem 1.05rem; border-top: 3px solid var(--border-2);
}
.kpi-eye { font-family: var(--sans); font-size: 0.72rem; font-weight: 600; color: var(--text-mute); margin-bottom: 0.35rem; }
.kpi-val { font-family: var(--sans); font-size: 1.65rem; font-weight: 800; line-height: 1; margin-bottom: 0.25rem; }
.kpi-sub { font-family: var(--sans); font-size: 0.72rem; color: var(--text-mute); }

.sec-rule {
    font-family: var(--sans); font-size: 0.82rem; font-weight: 700; color: var(--text);
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
    margin: 1.1rem 0 0.85rem 0;
}

.dr { display: flex; justify-content: space-between; align-items: baseline; padding: 0.42rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
.dr:last-child { border-bottom: none; }
.drk { font-family: var(--sans); font-size: 0.80rem; color: var(--text-mute); font-weight: 500; }
.drv { font-family: var(--mono); font-size: 0.82rem; color: var(--text); font-weight: 600; }

.sbar { display: flex; align-items: center; gap: 0.9rem; margin: 0.32rem 0 0.65rem; }
.sbar-lbl { font-family: var(--sans); font-size: 0.80rem; color: var(--text-soft); width: 170px; flex-shrink: 0; font-weight: 500; }
.sbar-track { flex: 1; height: 8px; background: #eef1f5; border-radius: 5px; position: relative; overflow: hidden; }
.sbar-fill { height: 100%; border-radius: 5px; transition: width 0.5s; }
.sbar-val { font-family: var(--mono); font-size: 0.78rem; color: var(--text-soft); width: 50px; text-align: right; flex-shrink: 0; font-weight: 600; }

.badge { display: inline-block; font-family: var(--sans); font-size: 0.70rem; font-weight: 700; padding: 3px 9px; border-radius: 6px; }
.badge-ok   { color: var(--success); background: var(--success-soft); border: 1px solid #bbf7d0; }
.badge-warn { color: var(--warning); background: var(--warning-soft); border: 1px solid #fde68a; }
.badge-fail { color: var(--danger);  background: var(--danger-soft);  border: 1px solid #fecaca; }

.flag { display: inline-block; font-family: var(--sans); font-size: 0.74rem; font-weight: 600; padding: 4px 10px; margin: 3px 4px 3px 0; border-radius: 999px; color: var(--danger); background: var(--danger-soft); border: 1px solid #fecaca; }

.mblk-lbl { font-family: var(--sans); font-size: 0.72rem; font-weight: 700; color: var(--text-mute); margin-bottom: 4px; }
.mblk { font-family: var(--mono); font-size: 0.75rem; color: var(--text-soft); background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.9rem; word-break: break-all; line-height: 1.7; margin: 0.3rem 0 0.9rem; }

.rtbl { width: 100%; border-collapse: collapse; }
.rtbl th { font-family: var(--sans); font-size: 0.72rem; font-weight: 700; color: var(--text-mute); padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--border); text-align: left; }
.rtbl td { font-family: var(--sans); font-size: 0.82rem; padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--border); color: var(--text-soft); }
.rtbl tr:last-child td { border-bottom: none; }
.rtbl tr:hover td { background: #f8fafc; }

.pstep { display: flex; align-items: center; gap: 0.85rem; padding: 0.48rem 0; border-bottom: 1px solid var(--border); font-family: var(--sans); font-size: 0.84rem; }
.pstep:last-child { border-bottom: none; }
.pstep-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.pstep-tag { font-family: var(--mono); font-size: 0.70rem; font-weight: 700; color: var(--primary); min-width: 54px; }

.imgpanel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.imgpanel-head { padding: 0.55rem 0.85rem; border-bottom: 1px solid var(--border); font-family: var(--sans); font-size: 0.76rem; font-weight: 700; color: var(--text-soft); background: #fafbfc; }

.db-file { display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0; border-bottom: 1px solid var(--border); font-family: var(--sans); font-size: 0.78rem; color: var(--text-soft); }
.db-file:last-child { border-bottom: none; }
.db-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--primary); flex-shrink: 0; }

.empty-st { padding: 4.5rem 2rem; border: 1.5px dashed var(--border-2); border-radius: 16px; text-align: center; margin-top: 0.8rem; background: var(--surface); }
.empty-title { font-family: var(--sans); font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 0.7rem; }
.empty-body { font-family: var(--sans); font-size: 0.88rem; color: var(--text-mute); line-height: 1.7; max-width: 440px; margin: 0 auto; }

.no-match { padding: 2.8rem 2rem; border: 1.5px dashed #fde68a; border-radius: 14px; text-align: center; background: var(--warning-soft); }

.chain-ok   { font-family: var(--sans); font-size: 0.85rem; font-weight: 600; color: var(--success); }
.chain-fail { font-family: var(--sans); font-size: 0.85rem; font-weight: 600; color: var(--danger); }
.w-ok   { font-family: var(--sans); font-size: 0.78rem; font-weight: 600; color: var(--success); }
.w-fail { font-family: var(--sans); font-size: 0.78rem; font-weight: 600; color: var(--danger); }

.footer { display: flex; justify-content: space-between; align-items: center; padding-top: 0.9rem; font-family: var(--sans); font-size: 0.74rem; color: var(--text-mute); }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def risk_color(r):
    return {"LOW": COL_SUCCESS, "MEDIUM": COL_WARNING, "HIGH": COL_DANGER}.get(r, COL_TEXT_MUTE)

def score_col(v, invert=False):
    x = 1 - v if invert else v
    if x >= 0.70: return COL_SUCCESS
    if x >= 0.40: return COL_WARNING
    return COL_DANGER

def sbar(label, value, invert=False):
    col = score_col(value, invert)
    pct = value * 100
    st.markdown(f"""<div class="sbar">
        <span class="sbar-lbl">{label}</span>
        <div class="sbar-track"><div class="sbar-fill" style="width:{pct:.1f}%;background:{col};"></div></div>
        <span class="sbar-val">{pct:.1f}%</span>
    </div>""", unsafe_allow_html=True)

def dr(k, v, vc=""):
    cs = f"color:{vc};" if vc else ""
    st.markdown(f"""<div class="dr">
        <span class="drk">{k}</span>
        <span class="drv" style="{cs}">{v}</span>
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="sec-rule">{label}</div>', unsafe_allow_html=True)

def get_db():
    si = sorted([p for p in DB_SESSION.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]) if DB_SESSION.exists() else []
    ri = sorted([p for p in DB_REPO.iterdir()    if p.suffix.lower() in {".jpg",".jpeg",".png"}]) if DB_REPO.exists() else []
    return (DB_SESSION, si) if si else (DB_REPO, ri) if ri else (DB_SESSION, [])

def verdict_text(r, fp):
    if r == "LOW":    return "Document Authenticated",  f"All forensic layers confirm structural authenticity. Fraud probability {fp:.1f}%."
    if r == "MEDIUM": return "Manual Review Recommended", f"Ambiguous signals detected across verification layers. Fraud probability {fp:.1f}%."
    return "Fraud Indicators Detected", f"Multiple layers flagged structural anomalies. Fraud probability {fp:.1f}%. Do not authenticate."

def gen_heatmap(q_path, ref_path):
    if not CV2_OK: return None
    try:
        q = cv2.imread(str(q_path), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        if q is None or r is None: return None
        if q.shape != r.shape: r = cv2.resize(r, (q.shape[1], q.shape[0]))
        diff = cv2.GaussianBlur(cv2.absdiff(q, r), (9, 9), 0)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_TURBO)
        base = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(base, 0.35, heat, 0.65, 0)
    except Exception: return None

def make_gauge(value_pct, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"size": 32, "color": COL_TEXT}},
        title={"text": title, "font": {"size": 13, "color": COL_TEXT_MUTE}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COL_BORDER, "tickfont": {"size": 10, "color": COL_TEXT_MUTE}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#eef1f5",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "#eef1f5"}],
        },
    ))
    fig.update_layout(height=190, margin=dict(l=25, r=25, t=45, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig

def make_signal_bar(labels, values, colors, height_per_row=42):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"size": 12, "color": COL_TEXT_SOFT, "family": "Inter"},
    ))
    fig.update_layout(
        height=height_per_row * max(len(labels), 1) + 50,
        margin=dict(l=10, r=45, t=10, b=10),
        xaxis=dict(range=[0, 112], showgrid=True, gridcolor=COL_BORDER, ticksuffix="%",
                   tickfont={"size": 11, "color": COL_TEXT_MUTE}),
        yaxis=dict(autorange="reversed", tickfont={"size": 12, "color": COL_TEXT_SOFT}),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        bargap=0.35,
    )
    return fig


# ─── Engine loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine(weights, conf, nms, ms, reg, asp):
    from script_v3 import ForensicVerificationEngine, LogSigner, CRYPTO_AVAILABLE, FraudScoreCalibrator
    signer = None
    if CRYPTO_AVAILABLE:
        rp, ep = ROOT/"keys"/"rsa_private.pem", ROOT/"keys"/"ec_private.pem"
        if rp.exists() and ep.exists():
            try: signer = LogSigner(rp.read_bytes(), ep.read_bytes())
            except Exception: pass
    calibrator = None
    cp = ROOT / "calibrator.json"
    if cp.exists():
        try: calibrator = FraudScoreCalibrator.load(str(cp))
        except Exception: pass
    return ForensicVerificationEngine(
        weights_path=weights, conf_thresh=conf, nms_dist=nms, cuda=False,
        audit_log_dir=str(ROOT/"audit_logs"), chain_store_dir=str(ROOT/"audit_store"),
        run_multiscale=ms, run_region_verification=reg,
        run_anti_spoof=asp, base_size=(640, 480),
        signer=signer, calibrator=calibrator, cache_size=512, max_workers=4,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div class="sb-brand">
        <div class="sb-brand-mark">Forensic Verification</div>
        <div class="sb-brand-sub">Document Authenticity Engine</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sect">Engine</div>', unsafe_allow_html=True)
    weights_input = st.text_input("Weights path", value="superpoint_v1.pth")
    weights_path  = str(ROOT / weights_input)
    weights_ok    = Path(weights_path).exists()
    st.markdown(f'<div class="{"w-ok" if weights_ok else "w-fail"}">{"Weights located" if weights_ok else "Weights not found"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sect">Detection</div>', unsafe_allow_html=True)
    conf_thresh     = st.slider("Keypoint confidence", 0.001, 0.050, 0.003, 0.001, format="%.3f")
    nms_dist        = st.slider("NMS distance", 1, 8, 3)
    match_threshold = st.slider("Match threshold", 0.40, 0.99, 0.70, 0.01, format="%.2f")
    max_keypoints   = st.select_slider("Max keypoints", options=[250, 500, 750, 1000, 1500, 2000], value=1000)

    st.markdown('<div class="sb-sect">Modules</div>', unsafe_allow_html=True)
    run_multiscale = st.checkbox("Multi-scale verification", value=True)
    run_region     = st.checkbox("Region verification",      value=True)
    run_anti_spoof = st.checkbox("Anti-spoof detection",     value=True)
    align_template = st.checkbox("Template alignment",       value=False)

    st.markdown('<div class="sb-sect">Reference Database</div>', unsafe_allow_html=True)
    db_upload = st.file_uploader("Upload reference images", type=["jpg","jpeg","png"],
                                  accept_multiple_files=True, label_visibility="collapsed")
    if db_upload:
        for f in db_upload:
            dest = DB_SESSION / f.name
            if not dest.exists(): dest.write_bytes(f.getbuffer())

    db_dir, db_imgs = get_db()
    source_lbl = "session" if db_dir == DB_SESSION else "database/"
    if db_imgs:
        html = "".join(f'<div class="db-file"><div class="db-dot"></div>{p.name}</div>' for p in db_imgs[:20])
        if len(db_imgs) > 20:
            html += f'<div style="font-family:var(--sans);font-size:0.72rem;color:var(--text-mute);padding:0.3rem 0">+ {len(db_imgs)-20} more</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:var(--sans);font-size:0.76rem;font-weight:600;color:var(--primary);margin-top:0.5rem;">{len(db_imgs)} reference image{"s" if len(db_imgs)>1 else ""} ({source_lbl})</div>', unsafe_allow_html=True)
        if db_dir == DB_SESSION and st.button("Clear database", use_container_width=True):
            shutil.rmtree(DB_SESSION, ignore_errors=True); DB_SESSION.mkdir(parents=True, exist_ok=True); st.rerun()
    else:
        st.markdown('<div style="font-family:var(--sans);font-size:0.80rem;color:var(--warning);line-height:1.7;">No reference images. Upload above or add to database/ folder.</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-top:1.8rem;padding-top:0.8rem;border-top:1px solid var(--border);font-family:var(--mono);font-size:0.70rem;color:var(--text-mute);">{datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
_, db_imgs_now = get_db()
sys_ready = weights_ok and bool(db_imgs_now)
dot_cls   = "dot-ready" if sys_ready else "dot-wait"
sys_lbl   = "System ready" if sys_ready else "Awaiting configuration"

st.markdown(f"""<div class="topbar">
    <div class="topbar-title">
        <span class="topbar-h1">Forensic Document Verification</span>
        <span class="topbar-sub">SuperPoint neural pipeline &nbsp;·&nbsp; 15 verification layers</span>
    </div>
    <div class="status-row">
        <span class="status-pill"><div class="status-dot {dot_cls}"></div>{sys_lbl}</span>
        <span class="status-pill">{datetime.datetime.now().strftime("%H:%M:%S")}</span>
    </div>
</div>""", unsafe_allow_html=True)

# ─── Upload row ────────────────────────────────────────────────────────────────
col_up, col_pipe = st.columns([2, 1], gap="large")

with col_up:
    st.markdown('<div class="panel"><div class="panel-title"><div class="panel-title-dot"></div>Query Document Upload</div>', unsafe_allow_html=True)
    query_file = st.file_uploader("Upload query document", type=["jpg","jpeg","png","bmp","tiff","webp"], label_visibility="collapsed")
    st.markdown('<div style="font-family:var(--sans);font-size:0.74rem;color:var(--text-mute);margin-top:0.4rem;">Supported formats: JPG, PNG, BMP, TIFF, WEBP</div>', unsafe_allow_html=True)
    if query_file:
        raw = query_file.read(); query_file.seek(0)
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.markdown('<div class="imgpanel"><div class="imgpanel-head">Query Preview</div>', unsafe_allow_html=True)
            st.image(raw, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipe:
    st.markdown('<div class="panel"><div class="panel-title"><div class="panel-title-dot"></div>Verification Pipeline</div>', unsafe_allow_html=True)
    layers = [("01","Descriptor Matching"),("02","Geometric Consistency"),("03","Tamper Localisation"),
              ("04","Multi-Scale Analysis"),("05","Region Verification"),("06","Integrity Fingerprint"),
              ("07","Fraud Score"),("08","Anti-Spoof"),("09","Adaptive Thresholds"),
              ("10","Signed Audit Log"),("11","Hash Chain"),("12","Template Alignment"),
              ("13","Descriptor Cache"),("14","Batch Pipeline"),("15","Score Calibration")]
    rows = "".join(f'<div class="dr"><span class="drk" style="font-family:var(--mono);color:var(--primary);font-weight:600;">{n}</span><span class="drv" style="font-size:0.80rem;font-weight:500;color:var(--text-soft);">{name}</span></div>' for n, name in layers)
    st.markdown(f'<div style="max-height:340px;overflow-y:auto;">{rows}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Run button ────────────────────────────────────────────────────────────────
db_dir, db_imgs = get_db()
can_run = bool(query_file and db_imgs and weights_ok)
bc, hc = st.columns([1, 6], gap="small")
with bc:
    run_btn = st.button("Run Analysis", type="primary", disabled=not can_run, use_container_width=True)
with hc:
    if not weights_ok:
        st.markdown(f'<div style="font-family:var(--sans);font-size:0.82rem;color:var(--danger);padding-top:0.6rem;">Weights not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not db_imgs:
        st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--warning);padding-top:0.6rem;">No reference images loaded.</div>', unsafe_allow_html=True)
    elif not query_file:
        st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--text-mute);padding-top:0.6rem;">Upload a query document to begin.</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

# ─── Pipeline execution ────────────────────────────────────────────────────────
STEPS = [
    ("INIT",   "Loading verification engine"),
    ("L1",     "Extracting keypoints & descriptors"),
    ("L1",     "Scanning reference database"),
    ("L2",     "Geometric consistency check"),
    ("L3",     "Tamper localisation"),
    ("L4",     "Multi-scale verification"),
    ("L5",     "Region verification"),
    ("L6-7",   "Integrity fingerprint & fraud score"),
    ("L8",     "Anti-spoof analysis"),
    ("L9",     "Adaptive threshold update"),
    ("L10-11", "Audit log & hash chain"),
]

if run_btn and can_run:
    q_path = Q_DIR / query_file.name
    q_path.write_bytes(query_file.getbuffer())
    db_dir_path, _ = get_db()

    prog = st.progress(0)
    step_box = st.empty()

    def render_steps(active):
        html = ""
        for i, (tag, lbl) in enumerate(STEPS):
            if i < active:
                dot   = "background:var(--success);"
                tc    = "color:var(--text-mute);"
                tag_c = "color:var(--text-mute);"
            elif i == active:
                dot   = "background:var(--primary);"
                tc    = "color:var(--text);"
                tag_c = "color:var(--primary);"
            else:
                dot   = "background:var(--border-2);"
                tc    = "color:var(--text-mute);"
                tag_c = "color:var(--text-mute);"
            html += (f'<div class="pstep" style="{tc}">'
                     f'<div class="pstep-dot" style="{dot}"></div>'
                     f'<span class="pstep-tag" style="{tag_c}">[{tag}]</span>'
                     f'{lbl}</div>')
        step_box.markdown(f'<div class="panel" style="max-width:520px;">{html}</div>', unsafe_allow_html=True)

    render_steps(0); prog.progress(5)
    try:
        engine = load_engine(weights_path, conf_thresh, nms_dist, run_multiscale, run_region, run_anti_spoof)
    except Exception:
        prog.empty(); step_box.empty()
        st.error("Engine failed to load."); st.code(traceback.format_exc(), language="python"); st.stop()

    render_steps(2); prog.progress(20)
    t0 = time.time()
    try:
        report = engine.verify(
            query_path=str(q_path), database_dir=str(db_dir_path),
            match_threshold=match_threshold, max_keypoints=max_keypoints,
            visualize=False, align_template=align_template,
        )
    except Exception:
        prog.empty(); step_box.empty()
        st.error("Verification failed."); st.code(traceback.format_exc(), language="python"); st.stop()

    elapsed = time.time() - t0
    render_steps(len(STEPS)); prog.progress(100)
    time.sleep(0.3); prog.empty(); step_box.empty()
    st.session_state.update(report=report, q_path=q_path, elapsed=elapsed)
    st.rerun()


# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state["report"] is not None:
    rep     = st.session_state["report"]
    q_path  = st.session_state["q_path"]
    elapsed = st.session_state["elapsed"]

    risk    = rep.risk_level.value if hasattr(rep.risk_level, "value") else str(rep.risk_level)
    fraud_p = rep.fraud_probability * 100
    auth_p  = rep.authenticity_score * 100
    geo_p   = rep.geometric.inlier_ratio * 100 if rep.geometric else 0.0
    spoof_p = rep.anti_spoof.overall_spoof_probability * 100 if rep.anti_spoof else 0.0

    rc = risk_color(risk)
    vtitle, vsub = verdict_text(risk, fraud_p)

    st.markdown(f"""<div class="verdict-card" style="border-left-color:{rc};">
        <div>
            <div class="verdict-classify">Classification &nbsp;·&nbsp; Session {rep.session_id} &nbsp;·&nbsp; {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}</div>
            <div class="verdict-title" style="color:{rc};">{vtitle}</div>
            <div class="verdict-body">{vsub}</div>
            <div class="verdict-meta">Processing time: {elapsed:.3f}s &nbsp;·&nbsp; File: {rep.fingerprint.file_size_bytes:,} bytes &nbsp;·&nbsp; Risk: {risk}</div>
        </div>
        <div style="min-width:190px;">
    """, unsafe_allow_html=True)
    st.plotly_chart(make_gauge(fraud_p, "Fraud Probability", rc), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div></div>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    def kpi(col, eye, val, sub, color=""):
        with col:
            st.markdown(f'<div class="kpi" style="border-top-color:{color if color else "var(--border-2)"};">'
                        f'<div class="kpi-eye">{eye}</div>'
                        f'<div class="kpi-val" style="color:{color if color else "var(--text)"};">{val}</div>'
                        f'<div class="kpi-sub">{sub}</div>'
                        f'</div>', unsafe_allow_html=True)
    kpi(k1, "Authenticity",      f"{auth_p:.1f}%",  "Composite weighted",  score_col(auth_p/100))
    kpi(k2, "Fraud Probability", f"{fraud_p:.1f}%", "Calibrated estimate", score_col(fraud_p/100, invert=True))
    kpi(k3, "Geometric Inliers", f"{geo_p:.1f}%",   "RANSAC homography",   score_col(geo_p/100))
    kpi(k4, "Spoof Probability", f"{spoof_p:.1f}%", "Anti-spoof analysis", score_col(spoof_p/100, invert=True))
    kpi(k5, "Risk Level",         risk,              "Classification",       rc)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["Visual Match", "Signals", "Regions", "Anti-Spoof", "Tamper Map", "Audit"])

    # Tab 1: Visual Match
    with tabs[0]:
        if rep.best_match_path and Path(rep.best_match_path).exists():
            mpath = Path(rep.best_match_path)
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown('<div class="imgpanel"><div class="imgpanel-head">Query Document</div>', unsafe_allow_html=True)
                st.image(str(q_path), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="imgpanel"><div class="imgpanel-head">Best Match — {mpath.name}</div>', unsafe_allow_html=True)
                st.image(str(mpath), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            ds = rep.descriptor_similarity
            st.markdown(f"""<div class="panel" style="margin-top:0.9rem;display:flex;align-items:center;gap:2rem;
                font-family:var(--sans);font-size:0.82rem;">
                <span style="color:var(--text-mute);font-weight:600;">MATCHED TO</span>
                <span style="color:var(--text);font-weight:600;">{mpath.name}</span>
                <span style="color:var(--text-mute);margin-left:auto;font-weight:600;">SIMILARITY</span>
                <span style="color:{score_col(ds)};font-weight:700;">{ds:.4f}</span>
                <span style="color:var(--text-mute);font-weight:600;">INLIERS</span>
                <span style="color:{score_col(geo_p/100)};font-weight:700;">{geo_p:.1f}%</span>
                <span style="color:var(--text-mute);font-weight:600;">RISK</span>
                <span style="color:{rc};font-weight:700;">{risk}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="no-match">
                <div style="font-family:var(--sans);font-size:1.05rem;font-weight:700;color:var(--warning);margin-bottom:0.6rem;">No Match Found</div>
                <div style="font-family:var(--sans);font-size:0.86rem;color:var(--text-soft);line-height:1.7;max-width:460px;margin:0 auto;">
                    The query document did not meet the match threshold against any reference image in the database.
                </div>
            </div>""", unsafe_allow_html=True)

    # Tab 2: Signals
    with tabs[1]:
        sec("Core Verification Signals")
        core_labels = ["Authenticity Score", "Fraud Probability (inv.)", "Descriptor Similarity"]
        core_values = [auth_p, 100 - fraud_p, rep.descriptor_similarity * 100]
        if rep.geometric:
            core_labels += ["Geometric Inlier Ratio", "Homography Stability"]
            core_values += [rep.geometric.inlier_ratio * 100, rep.geometric.homography_stability * 100]
        core_colors = [score_col(v/100) for v in core_values]
        st.plotly_chart(make_signal_bar(core_labels, core_values, core_colors), use_container_width=True, config={"displayModeBar": False})

        s1, s2 = st.columns(2, gap="large")
        with s1:
            sec("Geometric Detail")
            if rep.geometric:
                g = rep.geometric
                dr("Inlier count",        str(g.inlier_count))
                dr("Inlier ratio",        f"{g.inlier_ratio*100:.1f}%", score_col(g.inlier_ratio))
                dr("Reprojection error",  f"{g.reprojection_error:.2f} px" if g.reprojection_error < float('inf') else "N/A")
                dr("H stability",         f"{g.homography_stability:.4f}")
                dr("Condition number",    f"{g.condition_number:.2f}" if g.condition_number < float('inf') else "Degenerate")
            else:
                st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--text-mute);">Not available.</div>', unsafe_allow_html=True)

            sec("Tamper Metrics")
            if rep.tamper:
                t = rep.tamper
                dr("Unmatched ratio",      f"{t.unmatched_ratio*100:.1f}%")
                dr("Structural deviation", f"{t.structural_deviation:.4f}")
                dr("Suspicious zones",
                   ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None",
                   COL_DANGER if t.suspicious_quadrants else COL_SUCCESS)

        with s2:
            sec("Multi-Scale Consistency")
            if rep.multiscale:
                m = rep.multiscale
                sbar("Original resolution", m.original_score,  invert=False)
                sbar("Downscale (0.5x)",    m.downscale_score, invert=False)
                sbar("Upscale (2.0x)",      m.upscale_score,   invert=False)
                dr("Consistency score", f"{m.consistency_score:.4f}")
                dr("Scale variance",    f"{m.scale_variance:.6f}")
                dr("Stability", "Stable" if m.consistency_score < 0.05 else "Unstable",
                   COL_SUCCESS if m.consistency_score < 0.05 else COL_WARNING)
            else:
                st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--text-mute);">Multi-scale not enabled.</div>', unsafe_allow_html=True)

            if rep.tamper and rep.tamper.quadrant_scores:
                sec("Quadrant Deviation Scores")
                qs = rep.tamper.quadrant_scores
                q_labels = [k.replace("-", " ").title() for k in qs.keys()]
                q_max = max(qs.values()) or 1e-6
                q_values = [min(v / q_max, 1.0) * 100 for v in qs.values()]
                q_colors = [score_col(v/100, invert=True) for v in q_values]
                st.plotly_chart(make_signal_bar(q_labels, q_values, q_colors), use_container_width=True, config={"displayModeBar": False})

    # Tab 3: Regions
    with tabs[2]:
        if not rep.region_results:
            st.info("Region verification was not enabled or produced no results.")
        else:
            sec("Per-Zone Analysis")
            rows = ""
            for rr in rep.region_results:
                bcls = "badge-ok" if rr.status == "VERIFIED" else "badge-warn" if rr.status == "SUSPICIOUS" else "badge-fail"
                rows += (f"<tr><td>{rr.name.replace('_',' ').upper()}</td>"
                         f"<td><span class='badge {bcls}'>{rr.status}</span></td>"
                         f"<td>{rr.confidence*100:.1f}%</td>"
                         f"<td>{rr.inlier_ratio*100:.1f}%</td>"
                         f"<td>{rr.avg_descriptor_sim:.4f}</td>"
                         f"<td>{rr.matched_keypoints}</td></tr>")
            st.markdown(f'<table class="rtbl"><thead><tr><th>Zone</th><th>Status</th><th>Confidence</th><th>Inlier Ratio</th><th>Desc Sim</th><th>Matches</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            sec("Zone Confidence")
            zone_status_color = {"VERIFIED": COL_SUCCESS, "SUSPICIOUS": COL_WARNING, "FAILED": COL_DANGER}
            z_labels = [rr.name.replace("_", " ").upper() for rr in rep.region_results]
            z_values = [rr.confidence * 100 for rr in rep.region_results]
            z_colors = [zone_status_color.get(rr.status, COL_TEXT_MUTE) for rr in rep.region_results]
            st.plotly_chart(make_signal_bar(z_labels, z_values, z_colors), use_container_width=True, config={"displayModeBar": False})

    # Tab 4: Anti-Spoof
    with tabs[3]:
        if rep.anti_spoof is None:
            st.info("Anti-spoof analysis was not enabled.")
        else:
            a = rep.anti_spoof
            sec("Signal Scores")
            as_labels = ["Moire Pattern", "Photocopy", "Screen Replay", "Print-Scan Artefact", "Compression Artefact"]
            as_values = [a.moire_score*100, a.photocopy_score*100, a.screen_replay_score*100, a.print_scan_score*100, a.compression_score*100]
            as_colors = [score_col(v/100, invert=True) for v in as_values]
            st.plotly_chart(make_signal_bar(as_labels, as_values, as_colors), use_container_width=True, config={"displayModeBar": False})

            as1, as2 = st.columns(2, gap="large")
            with as1:
                sec("Overall Spoof Probability")
                st.plotly_chart(make_gauge(a.overall_spoof_probability*100, "Spoof Probability",
                                            score_col(a.overall_spoof_probability, invert=True)),
                                 use_container_width=True, config={"displayModeBar": False})
            with as2:
                sec("Triggered Flags")
                if a.flags:
                    st.markdown("".join(f'<span class="flag">{f}</span>' for f in a.flags), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-family:var(--sans);font-size:0.85rem;color:var(--success);font-weight:600;">No spoof flags triggered</div>', unsafe_allow_html=True)
                sec("Signal Reference")
                for name, desc in [
                    ("Moire",         "Ink-dot / camera-sensor interference from photographing a print"),
                    ("Photocopy",     "Histogram flattening and shadow clipping from copy processes"),
                    ("Screen Replay", "Horizontal scan-line peaks from re-photographing a display"),
                    ("Print-Scan",    "Halftone rosettes and DCT block artefacts from print-scan"),
                    ("Compression",   "8-pixel block boundary discontinuities from repeated JPEG save"),
                ]:
                    st.markdown(f"""<div class="dr">
                        <span class="drk">{name}</span>
                        <span style="font-family:var(--sans);font-size:0.78rem;color:var(--text-soft);max-width:260px;text-align:right;">{desc}</span>
                    </div>""", unsafe_allow_html=True)

    # Tab 5: Tamper Map
    with tabs[4]:
        if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
            hm = gen_heatmap(q_path, rep.best_match_path)
            if hm is not None:
                hm1, hm2 = st.columns([2, 1], gap="large")
                with hm1:
                    st.markdown('<div class="imgpanel"><div class="imgpanel-head">Deviation Heatmap — high deviation shown brighter</div>', unsafe_allow_html=True)
                    st.image(hm, use_container_width=True, channels="BGR")
                    st.markdown('</div>', unsafe_allow_html=True)
                with hm2:
                    sec("Tamper Summary")
                    if rep.tamper:
                        t = rep.tamper
                        dr("Unmatched ratio",      f"{t.unmatched_ratio*100:.1f}%")
                        dr("Structural deviation", f"{t.structural_deviation:.4f}")
                        dr("Colour scale",         "Low → High")
                        if t.suspicious_quadrants:
                            st.markdown("<br>", unsafe_allow_html=True)
                            sec("Flagged Zones")
                            st.markdown("".join(f'<span class="flag">{z}</span>' for z in t.suspicious_quadrants), unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--success);font-weight:600;margin-top:0.6rem;">No suspicious zones</div>', unsafe_allow_html=True)
            else:
                st.info("Heatmap generation failed.")
        else:
            st.info("No match available for tamper heatmap.")

    # Tab 6: Audit
    with tabs[5]:
        au1, au2 = st.columns(2, gap="large")
        with au1:
            sec("Integrity Fingerprint")
            fp = rep.fingerprint
            ts = datetime.datetime.fromtimestamp(fp.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.markdown('<div class="mblk-lbl">Image SHA-256</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mblk">{fp.image_sha256}</div>', unsafe_allow_html=True)
            st.markdown('<div class="mblk-lbl">Descriptor Hash</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="mblk">{fp.descriptor_hash}</div>', unsafe_allow_html=True)
            sec("Metadata")
            dr("Session ID",  rep.session_id)
            dr("Timestamp",   ts)
            dr("File size",   f"{fp.file_size_bytes:,} bytes")
            dr("Risk level",  risk, rc)
            dr("Processing",  f"{rep.processing_time_sec:.3f}s")
            if rep.db_stats:
                sec("Database Stats")
                dr("Documents evaluated", str(rep.db_stats.total_documents))
                dr("Cached descriptors",  str(rep.db_stats.cached_documents))
                dr("Cache hit rate",      f"{rep.db_stats.cache_hit_rate*100:.1f}%")
                dr("DB scan time",        f"{rep.db_stats.index_build_time:.2f}s")

        with au2:
            sec("Cryptographic Signatures")
            sig = rep.audit_signature
            if sig:
                st.markdown('<div class="mblk-lbl">RSA-PSS-SHA256-4096</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mblk">{sig.get("rsa_signature","")[:80]}...</div>', unsafe_allow_html=True)
                st.markdown('<div class="mblk-lbl">ECDSA-P384-SHA256</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mblk">{sig.get("ec_signature","")[:80]}...</div>', unsafe_allow_html=True)
                st.markdown('<div class="mblk-lbl">Payload SHA-256</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="mblk">{sig.get("payload_sha256","")}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--text-mute);">Log signing inactive. Add PEM keys to ./keys/ to enable.</div>', unsafe_allow_html=True)
            sec("Audit Log")
            with st.expander("View Full Log JSON"):
                if rep.audit_log_path and Path(rep.audit_log_path).exists():
                    st.code(Path(rep.audit_log_path).read_text(), language="json")
                else:
                    st.markdown('<div style="font-family:var(--sans);font-size:0.82rem;color:var(--text-mute);">Log not available.</div>', unsafe_allow_html=True)
            try:
                from script_v3 import AuditChainStore
                cp = ROOT / "audit_store"
                if cp.exists():
                    sec("Chain Integrity")
                    if st.button("Verify Audit Chain"):
                        store = AuditChainStore(str(cp))
                        valid, violations = store.verify_chain()
                        if valid:
                            st.markdown('<div class="chain-ok">Chain valid — all entries intact</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="chain-fail">Chain compromised</div>', unsafe_allow_html=True)
                            for v in violations: st.error(v)
            except Exception: pass

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Export")
    ex1, ex2, ex3, ex4 = st.columns(4, gap="small")
    try:
        from script_v3 import build_log_payload
        with ex1:
            st.download_button("Report JSON",
                data=json.dumps(build_log_payload(rep), indent=2).encode(),
                file_name=f"fdv_report_{rep.session_id}.json",
                mime="application/json", use_container_width=True)
    except Exception: pass
    if rep.audit_log_path and Path(rep.audit_log_path).exists():
        with ex2:
            st.download_button("Audit Log",
                data=Path(rep.audit_log_path).read_bytes(),
                file_name=f"audit_{rep.session_id}.json",
                mime="application/json", use_container_width=True)
    if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
        hm_ex = gen_heatmap(q_path, rep.best_match_path)
        if hm_ex is not None:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(hm_ex, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
            with ex3:
                st.download_button("Heatmap PNG",
                    data=buf.getvalue(),
                    file_name=f"heatmap_{rep.session_id}.png",
                    mime="image/png", use_container_width=True)
    with ex4:
        if st.button("New Verification", use_container_width=True):
            st.session_state.update(report=None, q_path=None, elapsed=None); st.rerun()


# ─── Empty state ───────────────────────────────────────────────────────────────
elif st.session_state["report"] is None and not run_btn:
    st.markdown("""<div class="empty-st">
        <div class="empty-title">Awaiting Document Submission</div>
        <div class="empty-body">
            Upload reference images in the sidebar, upload a query document above,
            and run the analysis to execute the complete 15-layer forensic
            verification pipeline.
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="footer">
    <span>Forensic Document Verification &nbsp;·&nbsp; v3.0</span>
    <span>SuperPoint Neural Architecture &nbsp;·&nbsp; 15 Layers &nbsp;·&nbsp; RSA-PSS-4096 + ECDSA-P384</span>
</div>""", unsafe_allow_html=True)
