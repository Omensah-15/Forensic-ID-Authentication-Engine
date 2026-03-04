"""
Forensic ID Authentication Engine UI
"""

import os
import sys
import time
import json
import datetime
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Forensic ID Authentication Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #080b0f;
    color: #b8c4d0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.4rem 2.8rem 5rem; max-width: 1440px; }

[data-testid="stSidebar"] { background: #05080c; border-right: 1px solid #12181f; }
[data-testid="stSidebar"] .block-container { padding: 1.8rem 1.4rem; }

.masthead { display:flex; align-items:center; gap:1.1rem;
            padding-bottom:1.8rem; border-bottom:1px solid #12181f; margin-bottom:2.2rem; }
.masthead-wordmark { font-family:'Syne',sans-serif; font-weight:800; font-size:1.35rem;
                     letter-spacing:-0.02em; color:#e2e8f0; line-height:1.15; }
.masthead-sub { font-size:0.68rem; letter-spacing:0.14em; color:#2e4050;
                text-transform:uppercase; margin-top:2px; }
.masthead-badge { margin-left:auto; font-size:0.62rem; letter-spacing:0.12em;
                  text-transform:uppercase; padding:3px 10px;
                  border:1px solid #1e2d3d; color:#3a8fd4; background:rgba(58,143,212,0.06); }

.verdict-authentic { background:linear-gradient(135deg,#051a0f 0%,#071e14 100%);
    border:1px solid #0f4025; border-left:3px solid #22c55e; padding:1.6rem 1.8rem; margin-bottom:1.6rem; }
.verdict-warning { background:linear-gradient(135deg,#1a1205 0%,#1e1507 100%);
    border:1px solid #3d2f0f; border-left:3px solid #f59e0b; padding:1.6rem 1.8rem; margin-bottom:1.6rem; }
.verdict-fraud { background:linear-gradient(135deg,#160505 0%,#1a0707 100%);
    border:1px solid #3d1010; border-left:3px solid #ef4444; padding:1.6rem 1.8rem; margin-bottom:1.6rem; }
.verdict-label { font-family:'Syne',sans-serif; font-weight:700; font-size:0.65rem;
                 letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.5rem; opacity:0.5; }
.verdict-title-authentic { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; color:#22c55e; }
.verdict-title-warning   { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; color:#f59e0b; }
.verdict-title-fraud     { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem; color:#ef4444; }
.verdict-meta { font-size:0.72rem; color:#3a5060; margin-top:0.4rem; letter-spacing:0.06em; }

.score-grid { display:grid; grid-template-columns:repeat(4,1fr);
              gap:1px; background:#12181f; border:1px solid #12181f; margin-bottom:1.6rem; }
.score-tile { background:#080b0f; padding:1.2rem 1.4rem; }
.score-tile-label { font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase;
                    color:#2a3a48; margin-bottom:0.55rem; }
.score-tile-value { font-family:'Syne',sans-serif; font-weight:700; font-size:1.75rem; line-height:1; }
.score-tile-sub   { font-size:0.62rem; color:#2a3a48; margin-top:0.35rem; }

.col-green { color:#22c55e; }
.col-amber { color:#f59e0b; }
.col-red   { color:#ef4444; }
.col-blue  { color:#3a8fd4; }
.col-dim   { color:#2a3a48; }

.section-header { font-size:0.6rem; letter-spacing:0.16em; text-transform:uppercase;
                  color:#1e3040; padding:0.5rem 0; border-bottom:1px solid #12181f;
                  margin:1.6rem 0 1rem; }

.data-row { display:flex; justify-content:space-between; align-items:baseline;
            padding:0.55rem 0; border-bottom:1px solid #0c1218; font-size:0.75rem; }
.data-row:last-child { border-bottom:none; }
.data-key { color:#2a3a48; }
.data-val { color:#7a96aa; font-weight:500; }

.region-table { width:100%; border-collapse:collapse; font-size:0.72rem; }
.region-table th { font-size:0.58rem; letter-spacing:0.12em; text-transform:uppercase;
                   color:#1e3040; padding:0.5rem 0.75rem; border-bottom:1px solid #12181f;
                   text-align:left; font-weight:500; }
.region-table td { padding:0.6rem 0.75rem; border-bottom:1px solid #0c1218; color:#5a7688; }
.region-table tr:last-child td { border-bottom:none; }

.badge { display:inline-block; font-size:0.58rem; letter-spacing:0.1em;
         text-transform:uppercase; padding:2px 7px; font-weight:500; }
.badge-verified   { color:#22c55e; background:rgba(34,197,94,0.08);  border:1px solid rgba(34,197,94,0.2); }
.badge-suspicious { color:#f59e0b; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); }
.badge-failed     { color:#ef4444; background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.2); }

.spoof-row { display:flex; align-items:center; gap:1rem; padding:0.45rem 0; font-size:0.72rem; }
.spoof-label { color:#2a3a48; width:130px; flex-shrink:0; }
.spoof-bar-bg { flex:1; height:4px; background:#0c1218; position:relative; }
.spoof-bar-fill { position:absolute; left:0; top:0; bottom:0; }
.spoof-val { color:#5a7688; width:42px; text-align:right; font-size:0.68rem; }

.flag-chip { display:inline-block; font-size:0.6rem; letter-spacing:0.1em;
             text-transform:uppercase; padding:3px 9px; margin:3px 4px 3px 0;
             color:#f59e0b; background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.2); }

.audit-block { background:#050810; border:1px solid #0d1520; padding:1.1rem 1.3rem;
               font-size:0.65rem; color:#1e3040; line-height:1.9; word-break:break-all; }
.audit-key { color:#1a2d3a; }
.audit-val { color:#2a5060; }
.audit-sig { color:#1a3028; font-size:0.6rem; }

.step-row { display:flex; align-items:center; gap:0.75rem; padding:0.4rem 0;
            font-size:0.72rem; color:#2a3a48; }
.step-dot-done    { width:6px;height:6px;border-radius:50%;background:#22c55e;flex-shrink:0; }
.step-dot-active  { width:6px;height:6px;border-radius:50%;background:#3a8fd4;flex-shrink:0;
                    animation:pulse 1s infinite; }
.step-dot-pending { width:6px;height:6px;border-radius:50%;background:#12181f;
                    border:1px solid #1e2d3d;flex-shrink:0; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.sidebar-section { font-size:0.58rem; letter-spacing:0.14em; text-transform:uppercase;
                   color:#1a2d3a; padding:0.9rem 0 0.4rem; }
.upload-label { font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase;
                color:#1e3040; margin-bottom:0.5rem; }
.vis-label { font-size:0.6rem; letter-spacing:0.12em; text-transform:uppercase;
             color:#1e3040; margin-bottom:0.5rem; }

.stButton > button { font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.1em;
    text-transform:uppercase; background:transparent; border:1px solid #1e3040;
    color:#3a8fd4; padding:0.55rem 1.4rem; transition:all 0.2s; border-radius:0; }
.stButton > button:hover { background:rgba(58,143,212,0.07); border-color:#3a8fd4; color:#6ab8f0; }
.stButton > button[kind="primary"] { background:#0a1e30; border-color:#3a8fd4; color:#6ab8f0; }

[data-testid="stFileUploadDropzone"] {
    background:#05080c !important; border:1px dashed #1a2d3d !important; border-radius:0 !important; }
[data-testid="stFileUploadDropzone"]:hover {
    border-color:#3a8fd4 !important; background:rgba(58,143,212,0.03) !important; }

.stSelectbox > div > div, .stTextInput > div > div {
    background:#05080c !important; border:1px solid #12181f !important;
    border-radius:0 !important; font-size:0.72rem !important; }
.stCheckbox > label { font-size:0.72rem; color:#3a5060; }
.stExpander { border:1px solid #12181f !important; border-radius:0 !important; background:#05080c !important; }
details summary p { font-size:0.7rem !important; color:#2a4050 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_uploaded(uploaded_file, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / uploaded_file.name
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out


def color_for_risk(risk_str):
    if risk_str == "LOW":
        return "col-green", "verdict-authentic", "verdict-title-authentic"
    if risk_str == "MEDIUM":
        return "col-amber", "verdict-warning", "verdict-title-warning"
    return "col-red", "verdict-fraud", "verdict-title-fraud"


def verdict_text(risk_str, fraud_pct):
    if risk_str == "LOW":
        return ("Document Authenticated",
                f"Structural analysis confirms authenticity across all layers. "
                f"Fraud probability {fraud_pct:.1f}%.")
    if risk_str == "MEDIUM":
        return ("Requires Manual Review",
                f"Ambiguous signals detected on one or more forensic layers. "
                f"Fraud probability {fraud_pct:.1f}%. Manual inspection advised.")
    return ("Potential Fraud Detected",
            f"Multiple forensic layers flagged structural anomalies. "
            f"Fraud probability {fraud_pct:.1f}%. Do not authenticate.")


def pct_color(v):
    if v >= 75: return "col-green"
    if v >= 45: return "col-amber"
    return "col-red"


def bar_color(score):
    if score < 0.25: return "#22c55e"
    if score < 0.55: return "#f59e0b"
    return "#ef4444"


def render_steps(active_idx, steps):
    rows = ""
    for i, (label, tag) in enumerate(steps):
        if i < active_idx:
            dot = "step-dot-done"; c1 = "col-green"; c2 = "color:#22c55e"
        elif i == active_idx:
            dot = "step-dot-active"; c1 = "col-blue"; c2 = "color:#3a8fd4"
        else:
            dot = "step-dot-pending"; c1 = "col-dim"; c2 = "color:#1e2d3d"
        rows += (f'<div class="step-row"><div class="{dot}"></div>'
                 f'<span class="{c1}" style="width:60px;flex-shrink:0">{tag}</span>'
                 f'<span style="{c2}">{label}</span></div>')
    return f'<div style="padding:0.8rem 0;">{rows}</div>'


# ── Engine loader (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_engine(weights_path, conf_thresh, nms_dist,
                run_multiscale, run_region, run_anti_spoof):
    sys.path.insert(0, str(Path(__file__).parent))
    from script_v2 import (ForensicVerificationEngine, LogSigner,
                            FraudScoreCalibrator, CRYPTO_AVAILABLE)

    signer = None
    key_dir = Path("./keys")
    if CRYPTO_AVAILABLE:
        rp = key_dir / "rsa_private.pem"
        ep = key_dir / "ec_private.pem"
        if rp.exists() and ep.exists():
            try:
                signer = LogSigner(rp.read_bytes(), ep.read_bytes())
            except Exception:
                pass

    calibrator = None
    if Path("./calibrator.json").exists():
        try:
            calibrator = FraudScoreCalibrator.load("./calibrator.json")
        except Exception:
            pass

    return ForensicVerificationEngine(
        weights_path=weights_path,
        conf_thresh=conf_thresh,
        nms_dist=nms_dist,
        cuda=False,
        audit_log_dir="./audit_logs",
        chain_store_dir="./audit_store",
        run_multiscale=run_multiscale,
        run_region_verification=run_region,
        run_anti_spoof=run_anti_spoof,
        base_size=(640, 480),
        signer=signer,
        calibrator=calibrator,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:1.2rem;border-bottom:1px solid #12181f;">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;
                    color:#e2e8f0;letter-spacing:-0.01em;">FIAE</div>
        <div style="font-size:0.6rem;letter-spacing:0.14em;color:#1a2d3a;
                    text-transform:uppercase;margin-top:3px;">
            Forensic ID Authentication Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Engine</div>', unsafe_allow_html=True)
    weights_path = st.text_input("Weights file", value="superpoint_v1.pth",
                                  help="Path to superpoint_v1.pth")

    st.markdown('<div class="sidebar-section">Detection Parameters</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Keypoint confidence",
                             min_value=0.001, max_value=0.050,
                             value=0.003, step=0.001, format="%.3f")
    nms_dist    = st.slider("NMS distance (px)", 1, 8, 3)
    match_threshold = st.slider("Match threshold", 0.40, 0.99, 0.70, 0.01, format="%.2f")
    max_keypoints   = st.select_slider("Max keypoints",
                                        options=[250, 500, 750, 1000, 1500, 2000],
                                        value=1000)

    st.markdown('<div class="sidebar-section">Analysis Modules</div>', unsafe_allow_html=True)
    run_multiscale = st.checkbox("Multi-scale verification",  value=True)
    run_region     = st.checkbox("Region-based verification", value=True)
    run_anti_spoof = st.checkbox("Anti-spoof detection",      value=True)
    align_template = st.checkbox("Template alignment",        value=False)

    st.markdown('<div class="sidebar-section">Reference Database</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem;color:#1a2d3a;margin-bottom:0.6rem;line-height:1.7;">'
        'Upload one or more reference images. The engine will search all of them.</div>',
        unsafe_allow_html=True
    )
    db_files = st.file_uploader(
        "Reference images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    n_refs = len(db_files) if db_files else 0
    if n_refs:
        st.markdown(
            f'<div style="font-size:0.65rem;color:#22c55e;margin-top:0.3rem;">'
            f'{n_refs} reference image{"s" if n_refs>1 else ""} loaded</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="margin-top:1.6rem;padding-top:1rem;border-top:1px solid #0d1520;'
        'font-size:0.56rem;color:#0f1e28;letter-spacing:0.08em;">v2.0 — forensic edition</div>',
        unsafe_allow_html=True
    )


# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div>
        <div class="masthead-wordmark">Forensic ID Authentication Engine</div>
        <div class="masthead-sub">Structural document verification — SuperPoint architecture — 10-layer pipeline</div>
    </div>
    <div class="masthead-badge">Production Build</div>
</div>
""", unsafe_allow_html=True)


# ── Upload + preview ──────────────────────────────────────────────────────────
up_col, info_col = st.columns([1.8, 1], gap="large")

with up_col:
    st.markdown('<div class="upload-label">Query Document</div>', unsafe_allow_html=True)
    query_file = st.file_uploader(
        "query", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed", key="query_upload"
    )

with info_col:
    st.markdown("""
    <div style="padding-top:0.2rem;">
    <div class="data-row"><span class="data-key">Layer 1</span><span class="data-val">Descriptor matching</span></div>
    <div class="data-row"><span class="data-key">Layer 2</span><span class="data-val">RANSAC homography</span></div>
    <div class="data-row"><span class="data-key">Layer 3</span><span class="data-val">Tamper localisation</span></div>
    <div class="data-row"><span class="data-key">Layer 4</span><span class="data-val">Multi-scale consistency</span></div>
    <div class="data-row"><span class="data-key">Layer 5</span><span class="data-val">Region verification</span></div>
    <div class="data-row"><span class="data-key">Layer 6</span><span class="data-val">Integrity fingerprint</span></div>
    <div class="data-row"><span class="data-key">Layer 7</span><span class="data-val">Fraud scoring</span></div>
    <div class="data-row"><span class="data-key">Layer 8</span><span class="data-val">Anti-spoof analysis</span></div>
    <div class="data-row"><span class="data-key">Layer 9</span><span class="data-val">Signed audit log</span></div>
    <div class="data-row"><span class="data-key">Layer 10</span><span class="data-val">Hash-chained storage</span></div>
    </div>
    """, unsafe_allow_html=True)

# Preview row
if query_file or db_files:
    n_prev  = min(1 + len(db_files), 7)
    p_cols  = st.columns(n_prev, gap="small")
    idx     = 0
    if query_file:
        with p_cols[0]:
            st.markdown('<div class="vis-label">Query</div>', unsafe_allow_html=True)
            b = query_file.read(); query_file.seek(0)
            st.image(b, use_container_width=True)
        idx = 1
    for i, dbf in enumerate(db_files[:6]):
        with p_cols[idx + i]:
            st.markdown(f'<div class="vis-label">Ref {i+1}</div>', unsafe_allow_html=True)
            b = dbf.read(); dbf.seek(0)
            st.image(b, use_container_width=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Run button ────────────────────────────────────────────────────────────────
weights_ok  = Path(weights_path).exists()
can_run     = bool(query_file and db_files and weights_ok)

btn_c, hint_c = st.columns([1, 5], gap="small")
with btn_c:
    run_btn = st.button("Run Verification", type="primary",
                         disabled=not can_run, use_container_width=True)
with hint_c:
    if not weights_ok:
        st.markdown(f'<div style="font-size:0.68rem;color:#5a2020;padding-top:0.55rem;">'
                    f'Weights not found: {weights_path}</div>', unsafe_allow_html=True)
    elif not query_file:
        st.markdown('<div style="font-size:0.68rem;color:#1e3040;padding-top:0.55rem;">'
                    'Upload a query image above.</div>', unsafe_allow_html=True)
    elif not db_files:
        st.markdown('<div style="font-size:0.68rem;color:#1e3040;padding-top:0.55rem;">'
                    'Upload reference images in the sidebar.</div>', unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:#12181f;margin:1.4rem 0 2rem;"></div>',
            unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and can_run:
    tmp_root  = Path(tempfile.mkdtemp(prefix="fiae_"))
    q_dir     = tmp_root / "query"
    db_dir    = tmp_root / "database"

    q_path = save_uploaded(query_file, q_dir)
    for dbf in db_files:
        save_uploaded(dbf, db_dir)

    STEPS = [
        ("Descriptor matching",       "Layer 1"),
        ("Geometric consistency",      "Layer 2"),
        ("Tamper localisation",        "Layer 3"),
        ("Multi-scale verification",   "Layer 4"),
        ("Region verification",        "Layer 5"),
        ("Integrity fingerprint",      "Layer 6"),
        ("Fraud scoring",              "Layer 7"),
        ("Anti-spoof analysis",        "Layer 8"),
        ("Signing audit log",          "Layer 9"),
        ("Writing audit chain",        "Layer 10"),
    ]

    prog = st.empty()
    prog.markdown(render_steps(0, STEPS), unsafe_allow_html=True)

    # Load engine
    try:
        engine = load_engine(weights_path, conf_thresh, nms_dist,
                              run_multiscale, run_region, run_anti_spoof)
    except Exception as e:
        st.error(f"Engine load error: {e}")
        shutil.rmtree(tmp_root, ignore_errors=True)
        st.stop()

    prog.markdown(render_steps(2, STEPS), unsafe_allow_html=True)

    start = time.time()
    try:
        report = engine.verify(
            query_path=str(q_path),
            database_dir=str(db_dir),
            match_threshold=match_threshold,
            max_keypoints=max_keypoints,
            visualize=False,
            align_template=align_template,
        )
    except Exception as e:
        st.error(f"Verification error: {e}")
        shutil.rmtree(tmp_root, ignore_errors=True)
        st.stop()

    elapsed = time.time() - start
    prog.markdown(render_steps(len(STEPS), STEPS), unsafe_allow_html=True)
    time.sleep(0.4)
    prog.empty()

    st.session_state.update({"report": report, "q_path": q_path,
                              "db_dir": db_dir, "elapsed": elapsed})


# ── Results ───────────────────────────────────────────────────────────────────
if "report" in st.session_state:
    report  = st.session_state["report"]
    q_path  = st.session_state["q_path"]
    elapsed = st.session_state["elapsed"]

    risk_str  = report.risk_level.value
    fraud_pct = report.fraud_probability * 100
    auth_pct  = report.authenticity_score * 100
    desc_sim  = report.descriptor_similarity

    col_cls, banner_cls, title_cls = color_for_risk(risk_str)
    v_title, v_sub = verdict_text(risk_str, fraud_pct)

    # Verdict banner
    st.markdown(f"""
    <div class="{banner_cls}">
        <div class="verdict-label">Verification Result &nbsp;|&nbsp; Session {report.session_id}</div>
        <div class="{title_cls}">{v_title}</div>
        <div class="verdict-meta">{v_sub} &nbsp;&mdash;&nbsp; processed in {elapsed:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

    # Score tiles
    geo_inlier = report.geometric.inlier_ratio * 100 if report.geometric else 0.0
    spoof_pct  = (report.anti_spoof.overall_spoof_probability * 100
                  if report.anti_spoof else 0.0)

    st.markdown(f"""
    <div class="score-grid">
        <div class="score-tile">
            <div class="score-tile-label">Authenticity Score</div>
            <div class="score-tile-value {pct_color(auth_pct)}">{auth_pct:.1f}<span style="font-size:1rem">%</span></div>
            <div class="score-tile-sub">Composite weighted score</div>
        </div>
        <div class="score-tile">
            <div class="score-tile-label">Fraud Probability</div>
            <div class="score-tile-value {pct_color(100-fraud_pct)}">{fraud_pct:.1f}<span style="font-size:1rem">%</span></div>
            <div class="score-tile-sub">Calibrated estimate</div>
        </div>
        <div class="score-tile">
            <div class="score-tile-label">Geometric Inliers</div>
            <div class="score-tile-value {pct_color(geo_inlier)}">{geo_inlier:.1f}<span style="font-size:1rem">%</span></div>
            <div class="score-tile-sub">RANSAC homography</div>
        </div>
        <div class="score-tile">
            <div class="score-tile-label">Spoof Probability</div>
            <div class="score-tile-value {pct_color(100-spoof_pct)}">{spoof_pct:.1f}<span style="font-size:1rem">%</span></div>
            <div class="score-tile-sub">Anti-spoof analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detail columns
    L, R = st.columns([1.05, 1], gap="large")

    with L:
        # Descriptor
        st.markdown('<div class="section-header">Descriptor Analysis</div>', unsafe_allow_html=True)
        match_name = Path(report.best_match_path).name if report.best_match_path else "No match found"
        st.markdown(f"""
        <div class="data-row"><span class="data-key">Best match</span>
             <span class="data-val">{match_name}</span></div>
        <div class="data-row"><span class="data-key">Descriptor similarity</span>
             <span class="data-val {pct_color(desc_sim*100)}">{desc_sim:.4f}</span></div>
        <div class="data-row"><span class="data-key">Risk classification</span>
             <span class="data-val {col_cls}">{risk_str}</span></div>
        <div class="data-row"><span class="data-key">Session ID</span>
             <span class="data-val" style="font-size:0.68rem;">{report.session_id}</span></div>
        """, unsafe_allow_html=True)

        # Geometric
        if report.geometric:
            g = report.geometric
            re_col = ("col-green" if g.reprojection_error < 2 else
                      ("col-amber" if g.reprojection_error < 6 else "col-red"))
            st.markdown('<div class="section-header">Geometric Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Inlier count</span>
                 <span class="data-val">{g.inlier_count} keypoints</span></div>
            <div class="data-row"><span class="data-key">Inlier ratio</span>
                 <span class="data-val {pct_color(g.inlier_ratio*100)}">{g.inlier_ratio*100:.1f}%</span></div>
            <div class="data-row"><span class="data-key">Reprojection error</span>
                 <span class="data-val {re_col}">{g.reprojection_error:.2f} px</span></div>
            <div class="data-row"><span class="data-key">Homography stability</span>
                 <span class="data-val {pct_color(g.homography_stability*100)}">{g.homography_stability:.4f}</span></div>
            """, unsafe_allow_html=True)

        # Tamper
        if report.tamper:
            t = report.tamper
            zones = ", ".join(t.suspicious_quadrants) if t.suspicious_quadrants else "None"
            zc    = "col-amber" if t.suspicious_quadrants else "col-green"
            st.markdown('<div class="section-header">Tamper Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Unmatched keypoints</span>
                 <span class="data-val {pct_color(100-t.unmatched_ratio*100)}">{t.unmatched_ratio*100:.1f}%</span></div>
            <div class="data-row"><span class="data-key">Structural deviation</span>
                 <span class="data-val {pct_color(100-t.structural_deviation*100)}">{t.structural_deviation:.4f}</span></div>
            <div class="data-row"><span class="data-key">Suspicious zones</span>
                 <span class="data-val {zc}">{zones}</span></div>
            """, unsafe_allow_html=True)

        # Multi-scale
        if report.multiscale:
            ms     = report.multiscale
            stable = ms.consistency_score < 0.05
            st.markdown('<div class="section-header">Multi-Scale Consistency</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="data-row"><span class="data-key">Original resolution</span>
                 <span class="data-val">{ms.original_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Downscale (0.5x)</span>
                 <span class="data-val">{ms.downscale_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Upscale (2x)</span>
                 <span class="data-val">{ms.upscale_score:.4f}</span></div>
            <div class="data-row"><span class="data-key">Consistency score</span>
                 <span class="data-val {'col-green' if stable else 'col-amber'}">{ms.consistency_score:.4f}
                 &nbsp;<span style="font-size:0.6rem;opacity:0.5;">{'stable' if stable else 'unstable'}</span></span></div>
            """, unsafe_allow_html=True)

    with R:
        # Anti-spoof
        if report.anti_spoof:
            a = report.anti_spoof
            st.markdown('<div class="section-header">Anti-Spoof Detection</div>', unsafe_allow_html=True)
            for label, score in [
                ("Moire pattern",   a.moire_score),
                ("Photocopy",       a.photocopy_score),
                ("Screen replay",   a.screen_replay_score),
                ("Print / scan",    a.print_scan_score),
                ("Overall spoof",   a.overall_spoof_probability),
            ]:
                pct  = score * 100
                fill = bar_color(score)
                st.markdown(f"""
                <div class="spoof-row">
                    <span class="spoof-label">{label}</span>
                    <div class="spoof-bar-bg">
                        <div class="spoof-bar-fill" style="width:{pct:.1f}%;background:{fill};"></div>
                    </div>
                    <span class="spoof-val">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)

            if a.flags:
                chips = "".join(f'<span class="flag-chip">{f}</span>' for f in a.flags)
                st.markdown(f'<div style="margin-top:0.8rem;">{chips}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="margin-top:0.8rem;font-size:0.67rem;color:#0f3020;">'
                    'No spoof flags triggered.</div>', unsafe_allow_html=True)

        # Region verification
        if report.region_results:
            st.markdown('<div class="section-header">Region Verification</div>', unsafe_allow_html=True)
            rows = ""
            for rr in report.region_results:
                bc = {"VERIFIED":"badge-verified","SUSPICIOUS":"badge-suspicious",
                      "FAILED":"badge-failed"}.get(rr.status, "badge-failed")
                rows += (f"<tr><td>{rr.name.replace('_',' ').title()}</td>"
                         f"<td><span class='badge {bc}'>{rr.status}</span></td>"
                         f"<td>{rr.confidence*100:.1f}%</td>"
                         f"<td>{rr.inlier_ratio*100:.1f}%</td>"
                         f"<td>{rr.matched_keypoints}</td></tr>")
            st.markdown(f"""
            <table class="region-table">
                <thead><tr><th>Zone</th><th>Status</th><th>Confidence</th>
                <th>Inliers</th><th>Matches</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)

        # Fingerprint
        st.markdown('<div class="section-header">Integrity Fingerprint</div>', unsafe_allow_html=True)
        fp     = report.fingerprint
        ts_str = datetime.datetime.fromtimestamp(fp.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="data-row"><span class="data-key">Image SHA-256</span>
             <span class="data-val" style="font-size:0.62rem;">{fp.image_sha256[:40]}...</span></div>
        <div class="data-row"><span class="data-key">Descriptor hash</span>
             <span class="data-val" style="font-size:0.62rem;">{fp.descriptor_hash[:40]}...</span></div>
        <div class="data-row"><span class="data-key">Timestamp</span>
             <span class="data-val">{ts_str}</span></div>
        """, unsafe_allow_html=True)

    # ── Match visualisation ───────────────────────────────────────────────
    if report.best_match_path and Path(report.best_match_path).exists():
        st.markdown('<div class="section-header">Match Visualisation</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2, gap="large")
        with v1:
            st.markdown('<div class="vis-label">Query Document</div>', unsafe_allow_html=True)
            st.image(str(q_path), use_container_width=True)
        with v2:
            st.markdown('<div class="vis-label">Best Database Match</div>', unsafe_allow_html=True)
            st.image(report.best_match_path, use_container_width=True)

    # Tamper heatmap
    if (report.tamper and report.tamper.deviation_heatmap is not None
            and report.tamper.deviation_heatmap.size > 0):
        st.markdown('<div class="section-header">Tamper Deviation Heatmap</div>', unsafe_allow_html=True)
        hm_rgb = cv2.cvtColor(report.tamper.deviation_heatmap, cv2.COLOR_BGR2RGB)
        hm_col, cap_col = st.columns([2, 1], gap="large")
        with hm_col:
            st.image(hm_rgb, use_container_width=True)
        with cap_col:
            suspicious = report.tamper.suspicious_quadrants
            if suspicious:
                chips = "".join(
                    f'<span class="flag-chip" style="color:#ef4444;border-color:rgba(239,68,68,0.3);">'
                    f'{z}</span>' for z in suspicious
                )
                st.markdown(
                    f'<div style="padding-top:1rem;">'
                    f'<div class="data-key" style="font-size:0.62rem;margin-bottom:0.5rem;">'
                    f'Flagged Zones</div>{chips}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="padding-top:1rem;font-size:0.68rem;color:#0f3020;">'
                    'No suspicious zones detected. Structural deviation is within tolerance.</div>',
                    unsafe_allow_html=True
                )

    # ── Audit record ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Audit Record</div>', unsafe_allow_html=True)

    with st.expander("View full audit log JSON", expanded=False):
        if report.audit_log_path and Path(report.audit_log_path).exists():
            with open(report.audit_log_path) as f:
                st.code(f.read(), language="json")
        else:
            st.markdown('<div style="font-size:0.68rem;color:#1e3040;">Log file not available.</div>',
                        unsafe_allow_html=True)

    sig = report.audit_signature
    if sig:
        st.markdown(f"""
        <div class="audit-block">
            <span class="audit-key">RSA-PSS-SHA256-4096</span><br>
            <span class="audit-sig">{sig.get('rsa_signature','')[:96]}...</span><br><br>
            <span class="audit-key">ECDSA-P384-SHA256</span><br>
            <span class="audit-sig">{sig.get('ec_signature','')[:96]}...</span><br><br>
            <span class="audit-key">Payload SHA-256</span>&nbsp;
            <span class="audit-val">{sig.get('payload_sha256','')}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.67rem;color:#0f1e28;padding:0.4rem 0;">'
            'Log signing inactive. Place key pairs in ./keys/ to enable RSA-PSS and ECDSA signatures.</div>',
            unsafe_allow_html=True
        )

    # ── Export ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    ex1, ex2, ex3, _ = st.columns([1, 1, 1, 2], gap="small")

    # Report JSON
    try:
        from script_v2 import build_log_payload
        payload_bytes = json.dumps(build_log_payload(report), indent=2).encode()
        with ex1:
            st.download_button("Download Report", data=payload_bytes,
                file_name=f"fiae_report_{report.session_id}.json",
                mime="application/json", use_container_width=True)
    except Exception:
        pass

    # Audit log
    if report.audit_log_path and Path(report.audit_log_path).exists():
        with ex2:
            st.download_button("Download Audit Log",
                data=Path(report.audit_log_path).read_bytes(),
                file_name=f"audit_{report.session_id}.json",
                mime="application/json", use_container_width=True)

    # Heatmap PNG
    if (report.tamper and report.tamper.deviation_heatmap is not None
            and report.tamper.deviation_heatmap.size > 0):
        import io
        hm_img = cv2.cvtColor(report.tamper.deviation_heatmap, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.fromarray(hm_img).save(buf, format="PNG")
        with ex3:
            st.download_button("Download Heatmap",
                data=buf.getvalue(),
                file_name=f"heatmap_{report.session_id}.png",
                mime="image/png", use_container_width=True)

# ── Empty state ───────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:5rem 2rem;border:1px dashed #0d1520;margin-top:0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                    color:#1a2a38;letter-spacing:-0.01em;margin-bottom:0.6rem;">
            No verification in progress
        </div>
        <div style="font-size:0.68rem;color:#101e28;text-align:center;
                    max-width:360px;line-height:1.9;">
            Upload a query document above and reference images in the sidebar,
            then press Run Verification to execute the full 10-layer forensic pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)
