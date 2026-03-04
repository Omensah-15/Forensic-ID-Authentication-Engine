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
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #080b0f;
    color: #b8c4d0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 5rem; max-width: 1440px; }

[data-testid="stSidebar"] {
    background: #05080c;
    border-right: 1px solid #0f161e;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* masthead */
.mh { display:flex; align-items:center; padding-bottom:1.6rem;
      border-bottom:1px solid #0f161e; margin-bottom:2rem; }
.mh-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.3rem;
            color:#dde6f0; letter-spacing:-0.02em; line-height:1.2; }
.mh-sub   { font-size:0.62rem; letter-spacing:0.14em; color:#253545;
            text-transform:uppercase; margin-top:3px; }
.mh-pill  { margin-left:auto; font-size:0.6rem; letter-spacing:0.12em;
            text-transform:uppercase; padding:3px 10px;
            border:1px solid #1a2d3d; color:#3a8fd4;
            background:rgba(58,143,212,0.05); white-space:nowrap; }

/* verdict */
.v-ok  { background:linear-gradient(135deg,#04160c,#061810);
         border:1px solid #0c3a1e; border-left:3px solid #22c55e;
         padding:1.5rem 1.7rem; margin-bottom:1.5rem; }
.v-med { background:linear-gradient(135deg,#160f03,#1a1306);
         border:1px solid #382b0e; border-left:3px solid #f59e0b;
         padding:1.5rem 1.7rem; margin-bottom:1.5rem; }
.v-bad { background:linear-gradient(135deg,#130404,#170606);
         border:1px solid #3a0e0e; border-left:3px solid #ef4444;
         padding:1.5rem 1.7rem; margin-bottom:1.5rem; }
.v-lbl   { font-size:0.62rem; letter-spacing:0.18em; text-transform:uppercase;
           opacity:0.45; margin-bottom:0.45rem; }
.v-ok  .v-lbl  { color:#22c55e; }
.v-med .v-lbl  { color:#f59e0b; }
.v-bad .v-lbl  { color:#ef4444; }
.v-h-ok  { font-family:'Syne',sans-serif; font-weight:800; font-size:1.8rem; color:#22c55e; }
.v-h-med { font-family:'Syne',sans-serif; font-weight:800; font-size:1.8rem; color:#f59e0b; }
.v-h-bad { font-family:'Syne',sans-serif; font-weight:800; font-size:1.8rem; color:#ef4444; }
.v-meta  { font-size:0.7rem; color:#304858; margin-top:0.4rem; }

/* score tiles */
.tiles { display:grid; grid-template-columns:repeat(4,1fr);
         gap:1px; background:#0f161e; border:1px solid #0f161e;
         margin-bottom:1.5rem; }
.tile  { background:#080b0f; padding:1.1rem 1.3rem; }
.tile-lbl { font-size:0.58rem; letter-spacing:0.14em; text-transform:uppercase;
            color:#253040; margin-bottom:0.5rem; }
.tile-val { font-family:'Syne',sans-serif; font-weight:700;
            font-size:1.65rem; line-height:1; }
.tile-sub { font-size:0.6rem; color:#253040; margin-top:0.3rem; }

/* colours */
.cg { color:#22c55e; }  .ca { color:#f59e0b; }
.cr { color:#ef4444; }  .cb { color:#3a8fd4; }  .cd { color:#253040; }

/* section header */
.sh { font-size:0.58rem; letter-spacing:0.16em; text-transform:uppercase;
      color:#1a2d3a; padding:0.4rem 0; border-bottom:1px solid #0f161e;
      margin:1.5rem 0 0.9rem; }

/* data rows */
.dr { display:flex; justify-content:space-between; align-items:baseline;
      padding:0.5rem 0; border-bottom:1px solid #0b1016; font-size:0.73rem; }
.dr:last-child { border-bottom:none; }
.dk { color:#253040; }
.dv { color:#6a8fa8; font-weight:500; }

/* spoof bars */
.sb-row { display:flex; align-items:center; gap:0.9rem;
          padding:0.4rem 0; font-size:0.7rem; }
.sb-lbl { color:#253040; width:120px; flex-shrink:0; }
.sb-bg  { flex:1; height:3px; background:#0b1016; position:relative; }
.sb-fill { position:absolute; left:0; top:0; bottom:0; }
.sb-val { color:#4a6a80; width:40px; text-align:right; font-size:0.65rem; }

/* region table */
.rt { width:100%; border-collapse:collapse; font-size:0.7rem; }
.rt th { font-size:0.56rem; letter-spacing:0.12em; text-transform:uppercase;
         color:#1a2d3a; padding:0.45rem 0.7rem; border-bottom:1px solid #0f161e;
         text-align:left; font-weight:500; }
.rt td { padding:0.55rem 0.7rem; border-bottom:1px solid #0b1016; color:#4a6878; }
.rt tr:last-child td { border-bottom:none; }
.bv { color:#22c55e; background:rgba(34,197,94,.07);
      border:1px solid rgba(34,197,94,.18); }
.bs { color:#f59e0b; background:rgba(245,158,11,.07);
      border:1px solid rgba(245,158,11,.18); }
.bf { color:#ef4444; background:rgba(239,68,68,.07);
      border:1px solid rgba(239,68,68,.18); }
.bdg { display:inline-block; font-size:0.56rem; letter-spacing:.09em;
       text-transform:uppercase; padding:2px 6px; font-weight:500; }

/* chips */
.chip { display:inline-block; font-size:0.58rem; letter-spacing:.09em;
        text-transform:uppercase; padding:2px 8px; margin:2px 3px 2px 0;
        color:#f59e0b; background:rgba(245,158,11,.06);
        border:1px solid rgba(245,158,11,.18); }

/* audit block */
.ab { background:#040710; border:1px solid #0b1520; padding:1rem 1.2rem;
      font-size:0.63rem; line-height:1.9; word-break:break-all; }
.ak { color:#162030; } .av { color:#1e4050; } .as { color:#0f2820; font-size:0.58rem; }

/* step list */
.step { display:flex; align-items:center; gap:0.7rem;
        padding:0.35rem 0; font-size:0.7rem; }
.dot-done { width:5px;height:5px;border-radius:50%;background:#22c55e;flex-shrink:0; }
.dot-live { width:5px;height:5px;border-radius:50%;background:#3a8fd4;flex-shrink:0;
            animation:blink 1s infinite; }
.dot-wait { width:5px;height:5px;border-radius:50%;background:#0f161e;
            border:1px solid #1a2d3d;flex-shrink:0; }
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}

/* match strip */
.mstrip { display:flex; gap:2rem; padding:0.7rem 1rem;
          background:#050810; border:1px solid #0b1520;
          margin-top:0.5rem; font-size:0.68rem; flex-wrap:wrap; }
.ms-k { color:#1a2d3a; }
.ms-v { font-weight:500; }

/* db file list */
.dbl { display:flex;align-items:center;gap:0.5rem;padding:.28rem 0;
       font-size:0.65rem;color:#2e4558;border-bottom:1px solid #0a0f14; }
.dbl:last-child { border-bottom:none; }
.dbl-dot { width:3px;height:3px;border-radius:50%;background:#1a3040;flex-shrink:0; }

/* upload zone tweaks */
[data-testid="stFileUploadDropzone"] {
    background:#04070b !important;
    border:1px dashed #172030 !important;
    border-radius:0 !important; }
[data-testid="stFileUploadDropzone"]:hover {
    border-color:#3a8fd4 !important; }

/* buttons */
.stButton>button {
    font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:.1em;
    text-transform:uppercase; background:transparent; border:1px solid #1a2d3a;
    color:#3a8fd4; padding:.5rem 1.2rem; transition:all .2s; border-radius:0; }
.stButton>button:hover {
    background:rgba(58,143,212,.07); border-color:#3a8fd4; color:#6ab8f0; }
.stButton>button[kind="primary"] {
    background:#091c2e; border-color:#3a8fd4; color:#6ab8f0; }

.stTextInput>div>div, .stSelectbox>div>div {
    background:#04070b !important; border:1px solid #0f161e !important;
    border-radius:0 !important; font-size:0.7rem !important; }
.stCheckbox>label { font-size:0.7rem; color:#304858; }
.stExpander { border:1px solid #0f161e !important; border-radius:0 !important;
              background:#04070b !important; }
details summary p { font-size:0.68rem !important; color:#253040 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct_col(v: float) -> str:
    return "cg" if v >= 75 else ("ca" if v >= 45 else "cr")

def bar_col(s: float) -> str:
    return "#22c55e" if s < .25 else ("#f59e0b" if s < .55 else "#ef4444")

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
            dot, tc, sc = "dot-done", "cg", "color:#22c55e"
        elif i == active:
            dot, tc, sc = "dot-live", "cb", "color:#3a8fd4"
        else:
            dot, tc, sc = "dot-wait", "cd", "color:#1a2d3a"
        rows += (f'<div class="step"><div class="{dot}"></div>'
                 f'<span class="{tc}" style="width:60px;flex-shrink:0">{tag}</span>'
                 f'<span style="{sc}">{lbl}</span></div>')
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
    gap   = 4
    total = qi.width + gap + mi.width
    out   = Image.new("RGB", (total, h), (8, 11, 15))
    out.paste(qi, (0, 0))
    out.paste(mi, (qi.width + gap, 0))
    d = ImageDraw.Draw(out)
    d.line([(qi.width + 1, 0), (qi.width + 1, h - 1)], fill=(20, 40, 55), width=2)
    return out


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
        f'<div style="font-size:.56rem;letter-spacing:.14em;text-transform:uppercase;'
        f'color:#162030;padding:.8rem 0 .35rem">{txt}</div>',
        unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:1rem;border-bottom:1px solid #0f161e">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;
                    color:#dde6f0">FIAE</div>
        <div style="font-size:.58rem;letter-spacing:.14em;color:#162030;
                    text-transform:uppercase;margin-top:2px">
            Forensic ID Authentication Engine
        </div>
    </div>""", unsafe_allow_html=True)

    sb_label("Engine")
    weights_input = st.text_input("Weights file", value="superpoint_v1.pth",
                                   label_visibility="collapsed",
                                   help="superpoint_v1.pth relative to app.py")
    st.markdown(
        f'<div style="font-size:.6rem;color:#162030;margin-top:-.4rem;margin-bottom:.4rem">'
        f'superpoint_v1.pth</div>', unsafe_allow_html=True)

    sb_label("Detection Parameters")
    conf_thresh     = st.slider("Keypoint confidence", .001, .050, .003, .001, format="%.3f")
    nms_dist        = st.slider("NMS distance (px)", 1, 8, 3)
    match_threshold = st.slider("Match threshold", .40, .99, .70, .01, format="%.2f")
    max_keypoints   = st.select_slider("Max keypoints",
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
        extra = (f'<div style="font-size:.6rem;color:#162030;padding:.25rem 0">'
                 f'+ {len(db_imgs)-25} more</div>') if len(db_imgs) > 25 else ""
        st.markdown(
            f'<div style="margin-bottom:.3rem">{items}{extra}</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:.63rem;color:#22c55e;margin-top:.3rem">'
            f'{len(db_imgs)} image{"s" if len(db_imgs)>1 else ""} loaded '
            f'<span style="color:#1a3040">({source_label})</span></div>',
            unsafe_allow_html=True)

        # Button to clear session uploads
        if db_dir == DB_SESSION and st.button("Clear uploaded database", use_container_width=True):
            shutil.rmtree(DB_SESSION, ignore_errors=True)
            DB_SESSION.mkdir(parents=True, exist_ok=True)
            st.rerun()
    else:
        st.markdown(
            '<div style="font-size:.63rem;color:#5a2020;line-height:1.75">'
            'No reference images loaded. Upload images above, or commit them '
            'to a <code style="color:#1e3a50">database/</code> folder in your repo.</div>',
            unsafe_allow_html=True)

    st.markdown(
        '<div style="margin-top:1.5rem;padding-top:.9rem;border-top:1px solid #0a0f14;'
        'font-size:.54rem;color:#0c1820;letter-spacing:.08em">v2.0 — forensic edition</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mh">
  <div>
    <div class="mh-title">Forensic ID Authentication Engine</div>
    <div class="mh-sub">SuperPoint architecture &mdash; 10-layer verification pipeline</div>
  </div>
  <div class="mh-pill">Production Build</div>
</div>
""", unsafe_allow_html=True)

# ── Query upload ──────────────────────────────────────────────────────────────
upload_col, layer_col = st.columns([1.8, 1], gap="large")

with upload_col:
    st.markdown(
        '<div style="font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;'
        'color:#1a2d3a;margin-bottom:.45rem">Query Document</div>',
        unsafe_allow_html=True)
    query_file = st.file_uploader(
        "query", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="query_upload")

with layer_col:
    st.markdown("""
    <div style="padding-top:.15rem">
      <div class="dr"><span class="dk">Layer 1</span><span class="dv">Descriptor matching</span></div>
      <div class="dr"><span class="dk">Layer 2</span><span class="dv">RANSAC homography</span></div>
      <div class="dr"><span class="dk">Layer 3</span><span class="dv">Tamper localisation</span></div>
      <div class="dr"><span class="dk">Layer 4</span><span class="dv">Multi-scale consistency</span></div>
      <div class="dr"><span class="dk">Layer 5</span><span class="dv">Region verification</span></div>
      <div class="dr"><span class="dk">Layer 6</span><span class="dv">Integrity fingerprint</span></div>
      <div class="dr"><span class="dk">Layer 7</span><span class="dv">Fraud scoring</span></div>
      <div class="dr"><span class="dk">Layer 8</span><span class="dv">Anti-spoof analysis</span></div>
      <div class="dr"><span class="dk">Layer 9</span><span class="dv">Signed audit log</span></div>
      <div class="dr"><span class="dk">Layer 10</span><span class="dv">Hash-chained storage</span></div>
    </div>""", unsafe_allow_html=True)

# Query preview
if query_file:
    raw = query_file.read(); query_file.seek(0)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            '<div style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;'
            'color:#1a2d3a;margin:.7rem 0 .35rem">Query Preview</div>',
            unsafe_allow_html=True)
        st.image(raw, use_container_width=True)

st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

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
            f'<div style="font-size:.68rem;color:#7a2525;padding-top:.5rem">'
            f'Weights not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not db_imgs:
        st.markdown(
            '<div style="font-size:.68rem;color:#5a2a10;padding-top:.5rem">'
            'No reference images — upload images in the sidebar database section.</div>',
            unsafe_allow_html=True)
    elif not query_file:
        st.markdown(
            '<div style="font-size:.68rem;color:#1e3040;padding-top:.5rem">'
            'Upload a query document above to begin.</div>',
            unsafe_allow_html=True)

st.markdown(
    '<div style="height:1px;background:#0f161e;margin:1.2rem 0 1.8rem"></div>',
    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
STEPS = [
    ("Loading engine",           "Init"),
    ("Extracting keypoints",     "Layer 1"),
    ("Searching database",       "Layer 1"),
    ("Geometric consistency",    "Layer 2"),
    ("Tamper localisation",      "Layer 3"),
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
      <div class="v-lbl">Verification Result &nbsp;|&nbsp; Session {rep.session_id}</div>
      <div class="{h_cls}">{v_title}</div>
      <div class="v-meta">{v_sub} &mdash; processed in {elapsed:.2f}s</div>
    </div>""", unsafe_allow_html=True)

    # Score tiles
    geo_p   = rep.geometric.inlier_ratio * 100 if rep.geometric else 0.0
    spoof_p = rep.anti_spoof.overall_spoof_probability * 100 if rep.anti_spoof else 0.0

    st.markdown(f"""
    <div class="tiles">
      <div class="tile">
        <div class="tile-lbl">Authenticity Score</div>
        <div class="tile-val {pct_col(auth_p)}">{auth_p:.1f}<span style="font-size:.95rem">%</span></div>
        <div class="tile-sub">Composite weighted score</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Fraud Probability</div>
        <div class="tile-val {pct_col(100-fraud_p)}">{fraud_p:.1f}<span style="font-size:.95rem">%</span></div>
        <div class="tile-sub">Calibrated estimate</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Geometric Inliers</div>
        <div class="tile-val {pct_col(geo_p)}">{geo_p:.1f}<span style="font-size:.95rem">%</span></div>
        <div class="tile-sub">RANSAC homography</div>
      </div>
      <div class="tile">
        <div class="tile-lbl">Spoof Probability</div>
        <div class="tile-val {pct_col(100-spoof_p)}">{spoof_p:.1f}<span style="font-size:.95rem">%</span></div>
        <div class="tile-sub">Anti-spoof analysis</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── MATCH VISUALISATION — front and centre ────────────────────────────
    st.markdown('<div class="sh">Match Visualisation</div>', unsafe_allow_html=True)

    if rep.best_match_path and Path(rep.best_match_path).exists():
        match_path = Path(rep.best_match_path)
        match_name = match_path.name

        col_q, col_m = st.columns(2, gap="large")
        with col_q:
            st.markdown(
                '<div style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;'
                'color:#1a2d3a;margin-bottom:.4rem">Query Document</div>',
                unsafe_allow_html=True)
            st.image(str(q_path), use_container_width=True)

        with col_m:
            st.markdown(
                f'<div style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;'
                f'color:#1a2d3a;margin-bottom:.4rem">Best Match &mdash; {match_name}</div>',
                unsafe_allow_html=True)
            st.image(str(match_path), use_container_width=True)

        # Side-by-side composite
        st.markdown(
            '<div style="font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;'
            'color:#1a2d3a;margin:.9rem 0 .4rem">Side-by-Side Comparison</div>',
            unsafe_allow_html=True)
        try:
            combo = side_by_side(q_path, match_path)
            st.image(combo, use_container_width=True)
        except Exception as e:
            st.caption(f"Comparison render failed: {e}")

        # Score strip
        st.markdown(f"""
        <div class="mstrip">
          <span class="ms-k">Matched to</span>
          <span class="ms-v" style="color:#4a7090">{match_name}</span>
          <span class="ms-k" style="margin-left:auto">Similarity</span>
          <span class="ms-v {pct_col(desc_sim*100)}">{desc_sim:.4f}</span>
          <span class="ms-k">Inliers</span>
          <span class="ms-v {pct_col(geo_p)}">{geo_p:.1f}%</span>
          <span class="ms-k">Risk</span>
          <span class="ms-v {c_cls}">{risk}</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding:2.5rem 1.5rem;border:1px dashed #1a2030;text-align:center">
          <div style="font-size:.78rem;color:#4a1818;margin-bottom:.4rem">
            No match found in database
          </div>
          <div style="font-size:.65rem;color:#1e1a1a;line-height:1.8">
            The query image did not meet the match threshold against any reference image.<br>
            Try lowering the match threshold in the sidebar, or add more reference images.
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Tamper heatmap ────────────────────────────────────────────────────
    if (rep.tamper and rep.tamper.deviation_heatmap is not None
            and rep.tamper.deviation_heatmap.size > 0 and CV2_OK):
        st.markdown('<div class="sh">Tamper Deviation Heatmap</div>', unsafe_allow_html=True)
        hm_rgb = cv2.cvtColor(rep.tamper.deviation_heatmap, cv2.COLOR_BGR2RGB)
        hc1, hc2 = st.columns([2, 1], gap="large")
        with hc1:
            st.image(hm_rgb, use_container_width=True)
        with hc2:
            sq = rep.tamper.suspicious_quadrants
            if sq:
                chips = "".join(
                    f'<span class="chip" style="color:#ef4444;'
                    f'border-color:rgba(239,68,68,.25)">{z}</span>' for z in sq)
                st.markdown(
                    f'<div style="padding-top:.9rem"><div style="font-size:.6rem;'
                    f'color:#253040;margin-bottom:.4rem">Flagged Zones</div>{chips}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="padding-top:.9rem;font-size:.67rem;color:#0c2818">'
                    'No suspicious zones. Deviation within tolerance.</div>',
                    unsafe_allow_html=True)

    # ── Detail columns ────────────────────────────────────────────────────
    LC, RC = st.columns([1.05, 1], gap="large")

    with LC:
        st.markdown('<div class="sh">Descriptor Analysis</div>', unsafe_allow_html=True)
        mn = Path(rep.best_match_path).name if rep.best_match_path else "No match found"
        st.markdown(f"""
        <div class="dr"><span class="dk">Best match</span><span class="dv">{mn}</span></div>
        <div class="dr"><span class="dk">Descriptor similarity</span>
             <span class="dv {pct_col(desc_sim*100)}">{desc_sim:.4f}</span></div>
        <div class="dr"><span class="dk">Risk classification</span>
             <span class="dv {c_cls}">{risk}</span></div>
        <div class="dr"><span class="dk">Session ID</span>
             <span class="dv" style="font-size:.63rem">{rep.session_id}</span></div>
        """, unsafe_allow_html=True)

        if rep.geometric:
            g = rep.geometric
            rc = "cg" if g.reprojection_error < 2 else ("ca" if g.reprojection_error < 6 else "cr")
            st.markdown('<div class="sh">Geometric Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Inlier count</span>
                 <span class="dv">{g.inlier_count} keypoints</span></div>
            <div class="dr"><span class="dk">Inlier ratio</span>
                 <span class="dv {pct_col(g.inlier_ratio*100)}">{g.inlier_ratio*100:.1f}%</span></div>
            <div class="dr"><span class="dk">Reprojection error</span>
                 <span class="dv {rc}">{g.reprojection_error:.2f} px</span></div>
            <div class="dr"><span class="dk">Homography stability</span>
                 <span class="dv {pct_col(g.homography_stability*100)}">{g.homography_stability:.4f}</span></div>
            """, unsafe_allow_html=True)

        if rep.tamper:
            t = rep.tamper
            zc = "ca" if t.suspicious_quadrants else "cg"
            zv = ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None"
            st.markdown('<div class="sh">Tamper Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Unmatched keypoints</span>
                 <span class="dv {pct_col(100-t.unmatched_ratio*100)}">{t.unmatched_ratio*100:.1f}%</span></div>
            <div class="dr"><span class="dk">Structural deviation</span>
                 <span class="dv">{t.structural_deviation:.4f}</span></div>
            <div class="dr"><span class="dk">Suspicious zones</span>
                 <span class="dv {zc}">{zv}</span></div>
            """, unsafe_allow_html=True)

        if rep.multiscale:
            m = rep.multiscale
            st_c = "cg" if m.consistency_score < .05 else "ca"
            st_l = "stable" if m.consistency_score < .05 else "unstable"
            st.markdown('<div class="sh">Multi-Scale Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dr"><span class="dk">Original</span><span class="dv">{m.original_score:.4f}</span></div>
            <div class="dr"><span class="dk">Downscale (0.5x)</span><span class="dv">{m.downscale_score:.4f}</span></div>
            <div class="dr"><span class="dk">Upscale (2x)</span><span class="dv">{m.upscale_score:.4f}</span></div>
            <div class="dr"><span class="dk">Consistency score</span>
                 <span class="dv {st_c}">{m.consistency_score:.4f}
                 <span style="font-size:.58rem;opacity:.5"> {st_l}</span></span></div>
            """, unsafe_allow_html=True)

    with RC:
        if rep.anti_spoof:
            a = rep.anti_spoof
            st.markdown('<div class="sh">Anti-Spoof Detection</div>', unsafe_allow_html=True)
            for lbl, sc in [
                ("Moire pattern",  a.moire_score),
                ("Photocopy",      a.photocopy_score),
                ("Screen replay",  a.screen_replay_score),
                ("Print / scan",   a.print_scan_score),
                ("Overall spoof",  a.overall_spoof_probability),
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
                st.markdown(f'<div style="margin-top:.7rem">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="margin-top:.7rem;font-size:.65rem;color:#0c2818">'
                    'No spoof flags triggered.</div>', unsafe_allow_html=True)

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
              <thead><tr><th>Zone</th><th>Status</th><th>Conf.</th>
              <th>Inliers</th><th>Matches</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">Integrity Fingerprint</div>', unsafe_allow_html=True)
        fp = rep.fingerprint
        ts = datetime.datetime.fromtimestamp(fp.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="dr"><span class="dk">Image SHA-256</span>
             <span class="dv" style="font-size:.6rem">{fp.image_sha256[:40]}...</span></div>
        <div class="dr"><span class="dk">Descriptor hash</span>
             <span class="dv" style="font-size:.6rem">{fp.descriptor_hash[:40]}...</span></div>
        <div class="dr"><span class="dk">Timestamp</span>
             <span class="dv">{ts}</span></div>
        """, unsafe_allow_html=True)

    # ── Audit record ──────────────────────────────────────────────────────
    st.markdown('<div class="sh">Audit Record</div>', unsafe_allow_html=True)
    with st.expander("View full audit log JSON", expanded=False):
        if rep.audit_log_path and Path(rep.audit_log_path).exists():
            st.code(Path(rep.audit_log_path).read_text(), language="json")
        else:
            st.markdown(
                '<div style="font-size:.66rem;color:#1a2d3a">Log not available.</div>',
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
            '<div style="font-size:.65rem;color:#0d1820;padding:.35rem 0">'
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

    if (rep.tamper and rep.tamper.deviation_heatmap is not None
            and rep.tamper.deviation_heatmap.size > 0 and CV2_OK):
        hm_rgb = cv2.cvtColor(rep.tamper.deviation_heatmap, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(hm_rgb).save(buf, format="PNG")
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
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:4.5rem 2rem;
                border:1px dashed #0c1620;margin-top:.5rem">
      <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;
                  color:#16242e;margin-bottom:.65rem">
        No verification in progress
      </div>
      <div style="font-size:.66rem;color:#0e1c26;text-align:center;
                  max-width:400px;line-height:2">
        Upload reference images in the sidebar, drop a query document above,
        then press <strong style="color:#1a3040">Run Verification</strong>
        to execute the full 10-layer forensic pipeline.
      </div>
    </div>""", unsafe_allow_html=True)
