"""
Forensic ID Authentication Engine — Demo Interface
===================================================
Cold War Intelligence Terminal aesthetic.

Run:
    streamlit run demo_app.py

Requires script_v3.py in the same directory.
"""

from __future__ import annotations

import io, os, sys, json, time, shutil, datetime, tempfile, traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIAE // Forensic Analysis Terminal",
    page_icon="🔬",
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
    st.session_state["session_dir"] = Path(tempfile.mkdtemp(prefix="fiae_"))
SESSION_DIR = st.session_state["session_dir"]
DB_SESSION  = SESSION_DIR / "database"
Q_DIR       = SESSION_DIR / "query"
DB_REPO     = ROOT / "database"
DB_SESSION.mkdir(parents=True, exist_ok=True)
Q_DIR.mkdir(parents=True, exist_ok=True)

for k, v in [("report", None), ("q_path", None), ("elapsed", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Courier+Prime:ital,wght@0,400;0,700;1,400&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --ink:        #0a0c0f;
    --deep:       #0e1117;
    --surface:    #141820;
    --raised:     #1c2230;
    --border:     #1e2d1e;
    --border2:    #2a3d2a;
    --green:      #00ff41;
    --green2:     #00cc33;
    --green3:     #009922;
    --green-dim:  rgba(0,255,65,0.08);
    --green-glow: rgba(0,255,65,0.15);
    --amber:      #ffb000;
    --red:        #ff2244;
    --text-a:     #c8d8c8;
    --text-b:     #7a947a;
    --text-c:     #3d4d3d;
    --mono:       'Courier Prime', monospace;
    --head:       'Rajdhani', sans-serif;
    --serif:      'Crimson Text', Georgia, serif;
}

html, body, [class*="css"] {
    font-family: var(--mono) !important;
    background: var(--ink) !important;
    color: var(--text-a) !important;
}
.stApp { background: var(--ink) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1.4rem 4rem 1.4rem !important; max-width: 1600px; }

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(0,255,65,0.012) 3px, rgba(0,255,65,0.012) 4px
    );
    pointer-events: none; z-index: 9998;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.5) 100%);
    pointer-events: none; z-index: 9997;
}

[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }

.stTextInput label, .stSlider label, .stCheckbox label,
.stSelectbox label, .stNumberInput label, .stFileUploader label {
    font-family: var(--head) !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: var(--text-b) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    caret-color: var(--green) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--green3) !important;
    box-shadow: 0 0 0 1px var(--green3), 0 0 12px var(--green-dim) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--green) !important; border-color: var(--green) !important;
}
.stSlider > div > div > div > div { background: var(--green3) !important; }

[data-testid="stCheckbox"] span {
    border-color: var(--border2) !important; border-radius: 0 !important;
    background: var(--surface) !important;
}
[data-testid="stCheckbox"] input:checked + span {
    background: var(--green3) !important; border-color: var(--green3) !important;
}
[data-testid="stCheckbox"] label span:last-child {
    color: var(--text-b) !important; font-family: var(--head) !important;
    font-size: 0.72rem !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stButton > button {
    font-family: var(--head) !important;
    font-size: 0.72rem !important; font-weight: 700 !important;
    letter-spacing: 0.18em !important; text-transform: uppercase !important;
    border-radius: 0 !important; padding: 0.55rem 1.4rem !important;
    transition: all 0.12s !important;
}
.stButton > button[kind="primary"] {
    background: transparent !important; color: var(--green) !important;
    border: 1px solid var(--green3) !important;
    box-shadow: inset 0 0 20px var(--green-dim), 0 0 8px var(--green-dim) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--green-glow) !important;
    box-shadow: inset 0 0 30px var(--green-glow), 0 0 20px var(--green-glow) !important;
    color: var(--green) !important;
}
.stButton > button:not([kind="primary"]) {
    background: transparent !important; color: var(--text-b) !important;
    border: 1px solid var(--border2) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--green3) !important; color: var(--green) !important;
}
.stButton > button:disabled { opacity: 0.3 !important; }

.stDownloadButton > button {
    font-family: var(--head) !important; font-size: 0.68rem !important;
    font-weight: 600 !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; border-radius: 0 !important;
    background: transparent !important; color: var(--green2) !important;
    border: 1px solid var(--border2) !important; padding: 0.4rem 0.9rem !important;
}
.stDownloadButton > button:hover {
    border-color: var(--green3) !important;
    box-shadow: 0 0 8px var(--green-dim) !important;
}

[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--green3) !important; }
[data-testid="stFileUploadDropzone"] * { color: var(--text-b) !important; }

[data-testid="stSelectSlider"] div[role="slider"] { background: var(--green3) !important; }

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border2) !important;
    gap: 0 !important; background: var(--deep) !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--head) !important; font-size: 0.65rem !important;
    font-weight: 600 !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; color: var(--text-c) !important;
    border: none !important; padding: 0.55rem 1.2rem !important;
    background: transparent !important; border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--green) !important; border-bottom: 2px solid var(--green3) !important;
    background: var(--green-dim) !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text-b) !important; }

[data-testid="stProgress"] > div > div { background: var(--green3) !important; }
[data-testid="stProgress"] > div { background: var(--border) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important; background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--head) !important; font-size: 0.68rem !important;
    letter-spacing: 0.10em !important; text-transform: uppercase !important;
    color: var(--text-b) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 0 !important; padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--head) !important; font-size: 0.58rem !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
    color: var(--text-c) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.4rem !important; color: var(--text-a) !important;
}

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stAlert"] { border-radius: 0 !important; }

/* ── Custom Components ── */
.term-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 0 0.7rem 0;
    border-bottom: 1px solid var(--border2); margin-bottom: 1.2rem;
}
.term-logo { display: flex; align-items: baseline; gap: 1rem; }
.term-logo-id {
    font-family: var(--head); font-size: 1.3rem; font-weight: 700;
    letter-spacing: 0.12em; color: var(--green);
    text-shadow: 0 0 20px var(--green-glow);
}
.term-logo-sub {
    font-family: var(--head); font-size: 0.62rem; font-weight: 400;
    letter-spacing: 0.20em; text-transform: uppercase; color: var(--text-c);
}
.term-status-row {
    display: flex; align-items: center; gap: 1.4rem;
    font-family: var(--head); font-size: 0.62rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-c);
}
.term-led { display: inline-flex; align-items: center; gap: 0.4rem; }
.led-dot { width: 6px; height: 6px; border-radius: 50%; }
.led-green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: led-pulse 2s infinite; }
.led-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); animation: led-pulse 1.4s infinite; }
.led-red   { background: var(--red);   box-shadow: 0 0 6px var(--red);   animation: led-pulse 0.8s infinite; }
@keyframes led-pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

.sb-brand { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.2rem; }
.sb-brand-mark {
    font-family: var(--head); font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.14em; color: var(--green); text-shadow: 0 0 12px var(--green-dim);
}
.sb-brand-sub {
    font-family: var(--head); font-size: 0.58rem;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-c); margin-top: 2px;
}
.sb-sect {
    font-family: var(--head); font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-c);
    padding: 0.8rem 0 0.3rem 0; border-bottom: 1px solid var(--border); margin-bottom: 0.6rem;
}

.verdict-wrap {
    position: relative; padding: 1.4rem 1.8rem;
    border: 1px solid; margin-bottom: 1.4rem; overflow: hidden;
}
.verdict-wrap::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        -45deg, transparent, transparent 20px,
        rgba(255,255,255,0.007) 20px, rgba(255,255,255,0.007) 21px
    );
    pointer-events: none;
}
.verdict-classify {
    font-family: var(--head); font-size: 0.58rem;
    letter-spacing: 0.22em; text-transform: uppercase; color: var(--text-c); margin-bottom: 0.5rem;
}
.verdict-title {
    font-family: var(--head); font-size: 2.4rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; line-height: 1;
    margin-bottom: 0.5rem; text-shadow: 0 0 30px currentColor;
}
.verdict-body {
    font-family: var(--serif); font-size: 0.90rem;
    color: var(--text-b); line-height: 1.6; font-style: italic;
}
.verdict-meta {
    font-family: var(--mono); font-size: 0.65rem;
    color: var(--text-c); margin-top: 0.7rem; letter-spacing: 0.04em;
}

.kpi {
    background: var(--surface); border: 1px solid var(--border2);
    padding: 0.9rem 1.1rem 0.8rem; position: relative;
}
.kpi::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: currentColor; opacity: 0.25;
}
.kpi-eye {
    font-family: var(--head); font-size: 0.55rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-c); margin-bottom: 0.35rem;
}
.kpi-val {
    font-family: var(--mono); font-size: 1.9rem; font-weight: 700;
    line-height: 1; margin-bottom: 0.2rem;
}
.kpi-sub { font-family: var(--head); font-size: 0.60rem; letter-spacing: 0.08em; color: var(--text-c); }

.sec-rule {
    font-family: var(--head); font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.20em; text-transform: uppercase; color: var(--green3);
    padding-bottom: 0.35rem; border-bottom: 1px solid var(--border2);
    margin: 1.1rem 0 0.8rem 0; display: flex; align-items: center; gap: 0.5rem;
}
.sec-rule::before { content: '//'; color: var(--text-c); font-weight: 400; }

.dr {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.38rem 0; border-bottom: 1px solid var(--border); font-size: 0.80rem;
}
.dr:last-child { border-bottom: none; }
.drk {
    font-family: var(--head); font-size: 0.64rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-c);
}
.drv { font-family: var(--mono); font-size: 0.76rem; color: var(--text-a); }

.sbar { display: flex; align-items: center; gap: 0.8rem; margin: 0.3rem 0 0.6rem; }
.sbar-lbl {
    font-family: var(--head); font-size: 0.64rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-b); width: 150px; flex-shrink: 0;
}
.sbar-track { flex: 1; height: 3px; background: var(--border2); position: relative; }
.sbar-fill { height: 100%; transition: width 0.6s; }
.sbar-val { font-family: var(--mono); font-size: 0.68rem; color: var(--text-b); width: 46px; text-align: right; flex-shrink: 0; }

.badge {
    display: inline-block; font-family: var(--head); font-size: 0.58rem;
    font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase;
    padding: 2px 7px; border-radius: 0;
}
.badge-ok   { color: var(--green);  background: var(--green-dim);         border: 1px solid var(--green3); }
.badge-warn { color: var(--amber);  background: rgba(255,176,0,0.08);     border: 1px solid rgba(255,176,0,0.3); }
.badge-fail { color: var(--red);    background: rgba(255,34,68,0.08);     border: 1px solid rgba(255,34,68,0.3); }

.flag {
    display: inline-block; font-family: var(--mono); font-size: 0.60rem;
    letter-spacing: 0.06em; padding: 2px 7px; margin: 2px 3px 2px 0;
    color: var(--red); background: rgba(255,34,68,0.08); border: 1px solid rgba(255,34,68,0.3);
}

.mblk {
    font-family: var(--mono); font-size: 0.66rem; color: var(--green2);
    background: var(--deep); border: 1px solid var(--border);
    padding: 0.7rem 0.9rem; word-break: break-all; line-height: 1.8; margin: 0.3rem 0 0.7rem;
}
.mblk-lbl {
    font-family: var(--head); font-size: 0.56rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-c); margin-bottom: 3px;
}

.rtbl { width: 100%; border-collapse: collapse; }
.rtbl th {
    font-family: var(--head); font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-c);
    padding: 0.4rem 0.7rem; border-bottom: 1px solid var(--border2); text-align: left;
}
.rtbl td {
    font-family: var(--mono); font-size: 0.70rem; padding: 0.45rem 0.7rem;
    border-bottom: 1px solid var(--border); color: var(--text-b);
}
.rtbl tr:last-child td { border-bottom: none; }
.rtbl tr:hover td { background: var(--green-dim); }

.pstep {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.38rem 0; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 0.70rem;
}
.pstep:last-child { border-bottom: none; }
.pstep-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.pstep-tag {
    font-family: var(--head); font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.08em; color: var(--green3); min-width: 70px;
}

.imgpanel { background: var(--deep); border: 1px solid var(--border2); }
.imgpanel-head {
    padding: 0.38rem 0.7rem; border-bottom: 1px solid var(--border);
    font-family: var(--head); font-size: 0.58rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-c);
    display: flex; align-items: center; gap: 0.4rem;
}
.imgpanel-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--green3); }

.db-file {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.28rem 0; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 0.66rem; color: var(--text-b);
}
.db-file:last-child { border-bottom: none; }
.db-dot { width: 3px; height: 3px; border-radius: 50%; background: var(--green3); flex-shrink: 0; }

.empty-st {
    padding: 5rem 2rem; border: 1px dashed var(--border2);
    text-align: center; margin-top: 0.8rem;
}
.empty-title {
    font-family: var(--head); font-size: 1rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-c); margin-bottom: 0.8rem;
}
.empty-body {
    font-family: var(--serif); font-size: 0.88rem; color: var(--text-c);
    font-style: italic; line-height: 1.8; max-width: 440px; margin: 0 auto;
}
.empty-cursor {
    display: inline-block; width: 8px; height: 1rem;
    background: var(--green3); margin-left: 3px;
    animation: blink 1s step-end infinite; vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

.no-match {
    padding: 3rem 2rem; border: 1px dashed rgba(255,176,0,0.3);
    text-align: center; background: rgba(255,176,0,0.03);
}

.chain-ok   { font-family: var(--mono); font-size: 0.76rem; color: var(--green); }
.chain-fail { font-family: var(--mono); font-size: 0.76rem; color: var(--red); }
.w-ok   { font-family: var(--head); font-size: 0.64rem; letter-spacing: 0.08em; color: var(--green2); }
.w-fail { font-family: var(--head); font-size: 0.64rem; letter-spacing: 0.08em; color: var(--red); }

.footer {
    display: flex; justify-content: space-between; align-items: center;
    padding-top: 0.8rem; font-family: var(--head); font-size: 0.58rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-c);
}

.panel-head {
    font-family: var(--head); font-size: 0.60rem; font-weight: 600;
    letter-spacing: 0.20em; text-transform: uppercase; color: var(--text-c);
    margin-bottom: 0.7rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
}
.panel-head-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green3); }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def risk_color(r):
    return {"LOW": "#00ff41", "MEDIUM": "#ffb000", "HIGH": "#ff2244"}.get(r, "#7a947a")

def score_col(v, invert=False):
    x = 1 - v if invert else v
    if x >= 0.70: return "#00ff41"
    if x >= 0.40: return "#ffb000"
    return "#ff2244"

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
    if r == "LOW":    return "Document Authenticated",    f"All forensic layers confirm structural authenticity. Fraud probability {fp:.1f}%."
    if r == "MEDIUM": return "Manual Review Required",    f"Ambiguous signals detected across verification layers. Fraud probability {fp:.1f}%."
    return "Fraud Indicators Detected",                  f"Multiple layers flagged structural anomalies. Fraud probability {fp:.1f}%. Do not authenticate."

def gen_heatmap(q_path, ref_path):
    if not CV2_OK: return None
    try:
        q = cv2.imread(str(q_path), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        if q is None or r is None: return None
        if q.shape != r.shape: r = cv2.resize(r, (q.shape[1], q.shape[0]))
        diff = cv2.GaussianBlur(cv2.absdiff(q, r), (9, 9), 0)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        base = cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(base, 0.30, heat, 0.70, 0)
    except Exception: return None


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
        <div class="sb-brand-mark">FIAE</div>
        <div class="sb-brand-sub">Forensic ID Authentication Engine</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sect">Engine</div>', unsafe_allow_html=True)
    weights_input = st.text_input("Weights path", value="superpoint_v1.pth")
    weights_path  = str(ROOT / weights_input)
    weights_ok    = Path(weights_path).exists()
    st.markdown(f'<div class="{"w-ok" if weights_ok else "w-fail"}">{"✓ Weights located" if weights_ok else "✗ Weights not found"}</div>', unsafe_allow_html=True)

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
            html += f'<div style="font-family:var(--head);font-size:0.60rem;color:var(--text-c);padding:0.3rem 0">+ {len(db_imgs)-20} more</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f'<div style="font-family:var(--head);font-size:0.64rem;letter-spacing:0.08em;color:var(--green2);margin-top:0.4rem;">{len(db_imgs)} reference image{"s" if len(db_imgs)>1 else ""} ({source_lbl})</div>', unsafe_allow_html=True)
        if db_dir == DB_SESSION and st.button("Clear database", use_container_width=True):
            shutil.rmtree(DB_SESSION, ignore_errors=True); DB_SESSION.mkdir(parents=True, exist_ok=True); st.rerun()
    else:
        st.markdown('<div style="font-family:var(--head);font-size:0.68rem;letter-spacing:0.06em;color:var(--amber);line-height:1.7;">No reference images. Upload above or add to database/ folder.</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-top:1.8rem;padding-top:0.7rem;border-top:1px solid var(--border);font-family:var(--mono);font-size:0.60rem;color:var(--text-c);">{datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────
_, db_imgs_now = get_db()
sys_ready = weights_ok and bool(db_imgs_now)
led_cls   = "led-green" if sys_ready else "led-amber"
sys_lbl   = "SYSTEM READY" if sys_ready else "AWAITING CONFIGURATION"

st.markdown(f"""<div class="term-bar">
    <div class="term-logo">
        <span class="term-logo-id">FIAE</span>
        <span class="term-logo-sub">Forensic Identity Authentication Engine &nbsp;//&nbsp; v3.0</span>
    </div>
    <div class="term-status-row">
        <span class="term-led"><div class="led-dot {led_cls}"></div>{sys_lbl}</span>
        <span>SuperPoint Neural Pipeline</span>
        <span>15 Verification Layers</span>
        <span style="color:var(--green3);">{datetime.datetime.now().strftime("%H:%M:%S")}</span>
    </div>
</div>""", unsafe_allow_html=True)

# ─── Upload row ────────────────────────────────────────────────────────────────
col_up, col_pipe = st.columns([2, 1], gap="large")

with col_up:
    st.markdown('<div class="panel-head"><div class="panel-head-dot"></div>Query Document Upload</div>', unsafe_allow_html=True)
    query_file = st.file_uploader("Upload query document", type=["jpg","jpeg","png","bmp","tiff","webp"], label_visibility="collapsed")
    st.markdown('<div style="font-family:var(--head);font-size:0.60rem;letter-spacing:0.10em;color:var(--text-c);margin-top:0.3rem;">JPG · PNG · BMP · TIFF · WEBP</div>', unsafe_allow_html=True)
    if query_file:
        raw = query_file.read(); query_file.seek(0)
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.markdown('<div class="imgpanel"><div class="imgpanel-head"><div class="imgpanel-dot"></div>QUERY PREVIEW</div>', unsafe_allow_html=True)
            st.image(raw, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

with col_pipe:
    st.markdown('<div class="panel-head"><div class="panel-head-dot"></div>Verification Pipeline</div>', unsafe_allow_html=True)
    layers = [("01","Descriptor Matching"),("02","Geometric Consistency"),("03","Tamper Localisation"),
              ("04","Multi-Scale Analysis"),("05","Region Verification"),("06","Integrity Fingerprint"),
              ("07","Fraud Score"),("08","Anti-Spoof"),("09","Adaptive Thresholds"),
              ("10","Signed Audit Log"),("11","Hash Chain"),("12","Template Alignment"),
              ("13","Descriptor Cache"),("14","Batch Pipeline"),("15","Score Calibration")]
    rows = "".join(f'<div class="dr"><span class="drk">{n}</span><span class="drv" style="font-size:0.70rem;">{name}</span></div>' for n, name in layers)
    st.markdown(f'<div style="background:var(--surface);border:1px solid var(--border2);padding:0.5rem 0.8rem;">{rows}</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─── Run button ────────────────────────────────────────────────────────────────
db_dir, db_imgs = get_db()
can_run = bool(query_file and db_imgs and weights_ok)
bc, hc = st.columns([1, 6], gap="small")
with bc:
    run_btn = st.button("⟩ Execute Analysis", type="primary", disabled=not can_run, use_container_width=True)
with hc:
    if not weights_ok:
        st.markdown(f'<div style="font-family:var(--mono);font-size:0.70rem;color:var(--red);padding-top:0.6rem;">Weights not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not db_imgs:
        st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--amber);padding-top:0.6rem;">No reference images loaded.</div>', unsafe_allow_html=True)
    elif not query_file:
        st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--text-c);padding-top:0.6rem;">Upload a query document to begin.</div>', unsafe_allow_html=True)

st.markdown("---")

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
                dot   = "background:var(--green3);"
                tc    = "color:var(--text-c);"
                tag_c = "color:var(--text-c);"
            elif i == active:
                dot   = "background:var(--green);box-shadow:0 0 8px var(--green);animation:led-pulse 0.6s infinite;"
                tc    = "color:var(--text-a);"
                tag_c = "color:var(--green);"
            else:
                dot   = "background:var(--border2);"
                tc    = "color:var(--text-c);"
                tag_c = "color:var(--text-c);"
            html += (f'<div class="pstep" style="{tc}">'
                     f'<div class="pstep-dot" style="{dot}"></div>'
                     f'<span class="pstep-tag" style="{tag_c}">[{tag}]</span>'
                     f'{lbl}</div>')
        step_box.markdown(f'<div style="background:var(--surface);border:1px solid var(--border2);padding:0.6rem 1rem;max-width:500px;">{html}</div>', unsafe_allow_html=True)

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

    bg_map = {"LOW": "rgba(0,255,65,0.04)", "MEDIUM": "rgba(255,176,0,0.04)", "HIGH": "rgba(255,34,68,0.04)"}
    bstyle = f"border-color:{rc};background:{bg_map.get(risk,'rgba(0,0,0,0)')};"
    st.markdown(f"""<div class="verdict-wrap" style="{bstyle}">
        <div class="verdict-classify">Classification // Session {rep.session_id} // {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}</div>
        <div class="verdict-title" style="color:{rc};">{vtitle}</div>
        <div class="verdict-body">{vsub}</div>
        <div class="verdict-meta">Processing time: {elapsed:.3f}s &nbsp;//&nbsp; File: {rep.fingerprint.file_size_bytes:,} bytes &nbsp;//&nbsp; Risk: {risk}</div>
    </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    def kpi(col, eye, val, sub, color=""):
        with col:
            st.markdown(f'<div class="kpi" style="color:{color if color else "var(--text-a)"};">'
                        f'<div class="kpi-eye">{eye}</div>'
                        f'<div class="kpi-val" style="color:{color if color else "var(--text-a)"};">{val}</div>'
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
                st.markdown('<div class="imgpanel"><div class="imgpanel-head"><div class="imgpanel-dot"></div>QUERY DOCUMENT</div>', unsafe_allow_html=True)
                st.image(str(q_path), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="imgpanel"><div class="imgpanel-head"><div class="imgpanel-dot"></div>BEST MATCH — {mpath.name}</div>', unsafe_allow_html=True)
                st.image(str(mpath), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            ds = rep.descriptor_similarity
            st.markdown(f"""<div style="display:flex;align-items:center;gap:2rem;padding:0.7rem 1.1rem;
                margin-top:0.5rem;background:var(--surface);border:1px solid var(--border2);
                font-family:var(--mono);font-size:0.70rem;">
                <span style="color:var(--text-c);">MATCHED TO</span>
                <span style="color:var(--text-a);">{mpath.name}</span>
                <span style="color:var(--text-c);margin-left:auto;">SIM</span>
                <span style="color:{score_col(ds)};">{ds:.4f}</span>
                <span style="color:var(--text-c);">INLIERS</span>
                <span style="color:{score_col(geo_p/100)};">{geo_p:.1f}%</span>
                <span style="color:var(--text-c);">RISK</span>
                <span style="color:{rc};font-weight:700;">{risk}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="no-match">
                <div style="font-family:var(--head);font-size:1rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--amber);margin-bottom:0.6rem;">No Match Found</div>
                <div style="font-family:var(--serif);font-size:0.88rem;color:var(--text-b);font-style:italic;line-height:1.7;max-width:460px;margin:0 auto;">
                    The query document did not meet the match threshold against any reference image in the database.
                </div>
            </div>""", unsafe_allow_html=True)

    # Tab 2: Signals
    with tabs[1]:
        s1, s2 = st.columns(2, gap="large")
        with s1:
            sec("Core Verification Signals")
            sbar("Authenticity Score",    rep.authenticity_score,    invert=False)
            sbar("Fraud Probability",     rep.fraud_probability,     invert=True)
            sbar("Descriptor Similarity", rep.descriptor_similarity, invert=False)
            if rep.geometric:
                sbar("Geometric Inlier Ratio",  rep.geometric.inlier_ratio,         invert=False)
                sbar("Homography Stability",    rep.geometric.homography_stability,  invert=False)
            if rep.multiscale:
                sbar("Scale Consistency", max(0.0, 1.0 - rep.multiscale.consistency_score * 5), invert=False)
            sec("Geometric Detail")
            if rep.geometric:
                g = rep.geometric
                dr("Inlier count",        str(g.inlier_count))
                dr("Inlier ratio",        f"{g.inlier_ratio*100:.1f}%", score_col(g.inlier_ratio))
                dr("Reprojection error",  f"{g.reprojection_error:.2f} px" if g.reprojection_error < float('inf') else "N/A")
                dr("H stability",         f"{g.homography_stability:.4f}")
                dr("Condition number",    f"{g.condition_number:.2f}" if g.condition_number < float('inf') else "Degenerate")
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
                   "#00ff41" if m.consistency_score < 0.05 else "#ffb000")
            else:
                st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--text-c);">Multi-scale not enabled.</div>', unsafe_allow_html=True)
            sec("Tamper Metrics")
            if rep.tamper:
                t = rep.tamper
                dr("Unmatched ratio",      f"{t.unmatched_ratio*100:.1f}%")
                dr("Structural deviation", f"{t.structural_deviation:.4f}")
                dr("Suspicious zones",
                   ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None",
                   "#ff2244" if t.suspicious_quadrants else "#00ff41")
                if t.quadrant_scores:
                    sec("Quadrant Scores")
                    mx = max(t.quadrant_scores.values()) or 1e-6
                    for qn, qs in t.quadrant_scores.items():
                        sbar(qn.replace("-"," ").title(), min(qs/mx, 1.0), invert=True)

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
            for rr in rep.region_results:
                col = "#00ff41" if rr.status == "VERIFIED" else "#ffb000" if rr.status == "SUSPICIOUS" else "#ff2244"
                pct = rr.confidence * 100
                st.markdown(f"""<div class="sbar">
                    <span class="sbar-lbl">{rr.name.replace("_"," ").upper()}</span>
                    <div class="sbar-track"><div class="sbar-fill" style="width:{pct:.1f}%;background:{col};"></div></div>
                    <span class="sbar-val">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)

    # Tab 4: Anti-Spoof
    with tabs[3]:
        if rep.anti_spoof is None:
            st.info("Anti-spoof analysis was not enabled.")
        else:
            a = rep.anti_spoof
            as1, as2 = st.columns(2, gap="large")
            with as1:
                sec("Signal Scores")
                sbar("Moire Pattern",        a.moire_score,               invert=True)
                sbar("Photocopy",            a.photocopy_score,           invert=True)
                sbar("Screen Replay",        a.screen_replay_score,       invert=True)
                sbar("Print-Scan Artefact",  a.print_scan_score,          invert=True)
                sbar("Compression Artefact", a.compression_score,         invert=True)
                sbar("Overall Spoof Prob.",  a.overall_spoof_probability, invert=True)
            with as2:
                sec("Triggered Flags")
                if a.flags:
                    st.markdown("".join(f'<span class="flag">{f}</span>' for f in a.flags), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-family:var(--mono);font-size:0.72rem;color:var(--green);">No spoof flags triggered</div>', unsafe_allow_html=True)
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
                        <span style="font-family:var(--serif);font-size:0.80rem;color:var(--text-b);font-style:italic;max-width:260px;text-align:right;">{desc}</span>
                    </div>""", unsafe_allow_html=True)

    # Tab 5: Tamper Map
    with tabs[4]:
        if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
            hm = gen_heatmap(q_path, rep.best_match_path)
            if hm is not None:
                hm1, hm2 = st.columns([2, 1], gap="large")
                with hm1:
                    st.markdown('<div class="imgpanel"><div class="imgpanel-head"><div class="imgpanel-dot"></div>DEVIATION HEATMAP — HIGH DEVIATION = BRIGHT</div>', unsafe_allow_html=True)
                    st.image(hm, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with hm2:
                    sec("Tamper Summary")
                    if rep.tamper:
                        t = rep.tamper
                        dr("Unmatched ratio",      f"{t.unmatched_ratio*100:.1f}%")
                        dr("Structural deviation", f"{t.structural_deviation:.4f}")
                        dr("Colour scale",         "Dark → Bright")
                        if t.suspicious_quadrants:
                            st.markdown("<br>", unsafe_allow_html=True)
                            sec("Flagged Zones")
                            st.markdown("".join(f'<span class="flag">{z}</span>' for z in t.suspicious_quadrants), unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--green);margin-top:0.6rem;">No suspicious zones</div>', unsafe_allow_html=True)
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
                st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--text-c);">Log signing inactive. Add PEM keys to ./keys/ to enable.</div>', unsafe_allow_html=True)
            sec("Audit Log")
            with st.expander("View Full Log JSON"):
                if rep.audit_log_path and Path(rep.audit_log_path).exists():
                    st.code(Path(rep.audit_log_path).read_text(), language="json")
                else:
                    st.markdown('<div style="font-family:var(--mono);font-size:0.70rem;color:var(--text-c);">Log not available.</div>', unsafe_allow_html=True)
            try:
                from script_v3 import AuditChainStore
                cp = ROOT / "audit_store"
                if cp.exists():
                    sec("Chain Integrity")
                    if st.button("Verify Audit Chain"):
                        store = AuditChainStore(str(cp))
                        valid, violations = store.verify_chain()
                        if valid:
                            st.markdown('<div class="chain-ok">✓ CHAIN VALID — All entries intact</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="chain-fail">✗ CHAIN COMPROMISED</div>', unsafe_allow_html=True)
                            for v in violations: st.error(v)
            except Exception: pass

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Export")
    ex1, ex2, ex3, ex4 = st.columns(4, gap="small")
    try:
        from script_v3 import build_log_payload
        with ex1:
            st.download_button("↓ Report JSON",
                data=json.dumps(build_log_payload(rep), indent=2).encode(),
                file_name=f"fiae_report_{rep.session_id}.json",
                mime="application/json", use_container_width=True)
    except Exception: pass
    if rep.audit_log_path and Path(rep.audit_log_path).exists():
        with ex2:
            st.download_button("↓ Audit Log",
                data=Path(rep.audit_log_path).read_bytes(),
                file_name=f"audit_{rep.session_id}.json",
                mime="application/json", use_container_width=True)
    if CV2_OK and rep.best_match_path and Path(rep.best_match_path).exists():
        hm_ex = gen_heatmap(q_path, rep.best_match_path)
        if hm_ex is not None:
            buf = io.BytesIO()
            Image.fromarray(cv2.cvtColor(hm_ex, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
            with ex3:
                st.download_button("↓ Heatmap PNG",
                    data=buf.getvalue(),
                    file_name=f"heatmap_{rep.session_id}.png",
                    mime="image/png", use_container_width=True)
    with ex4:
        if st.button("⟩ New Verification", use_container_width=True):
            st.session_state.update(report=None, q_path=None, elapsed=None); st.rerun()


# ─── Empty state ───────────────────────────────────────────────────────────────
elif st.session_state["report"] is None and not run_btn:
    st.markdown("""<div class="empty-st">
        <div class="empty-title">Awaiting Document Submission<span class="empty-cursor"></span></div>
        <div class="empty-body">
            Upload reference images in the sidebar, upload a query document above,
            and execute the analysis to run the complete 15-layer forensic
            verification pipeline.
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="footer">
    <span>Forensic ID Authentication Engine &nbsp;//&nbsp; v3.0</span>
    <span>SuperPoint Neural Architecture &nbsp;·&nbsp; 15 Layers &nbsp;·&nbsp; RSA-PSS-4096 + ECDSA-P384</span>
</div>""", unsafe_allow_html=True)
