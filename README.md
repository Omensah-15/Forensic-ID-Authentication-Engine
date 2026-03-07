# Forensic ID Authentication Engine

A production-grade forensic image verification system built on the SuperPoint deep learning architecture. The engine performs multi-layer structural analysis of identity documents and security patterns to detect counterfeits, localise tampering, and produce cryptographically signed audit records.

The system extracts keypoints from query images, validates geometric consistency against a trusted reference database, and applies anti-spoof analysis, multi-scale verification, and region-based zone inspection to produce a statistically calibrated fraud probability with a structured chain-of-custody audit log.

---

## Architecture Overview

The engine executes fifteen verification layers sequentially on every query:

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Descriptor Matching | SuperPoint reciprocal cosine similarity |
| 2 | Geometric Consistency | RANSAC homography, inlier ratio, reprojection error |
| 3 | Tamper Localisation | Per-quadrant deviation heatmap |
| 4 | Multi-Scale Robustness | Three-resolution consistency check |
| 5 | Region Verification | Per-zone ID document analysis |
| 6 | Integrity Fingerprint | SHA-256 image and descriptor hash |
| 7 | Fraud Scoring | Weighted composite with isotonic calibration |
| 8 | Anti-Spoof Analysis | Moire, photocopy, screen replay, print-scan, compression |
| 9 | Adaptive Thresholds | EMA-based self-updating detection thresholds |
| 10 | Signed Audit Log | RSA-PSS 4096 and ECDSA P-384 dual signatures |
| 11 | Hash-Chained Storage | Append-only tamper-evident audit chain |
| 12 | Template Alignment | ORB-RANSAC homographic pre-alignment |
| 13 | Descriptor Cache | LRU cache with file-mtime invalidation |
| 14 | Batch Pipeline | Concurrent multi-document verification |
| 15 | Score Calibration | Isotonic regression and ROC-optimal thresholds |

---

## Features

- **Geometric Consistency Validation** — RANSAC homography estimation rejects texture-copy counterfeits that lack coherent planar geometry
- **Tamper Localisation** — Spatial deviation heatmap identifies the specific image region where modification occurred
- **Multi-Scale Verification** — Matching at three resolutions detects resampling artefacts introduced during forgery production
- **Region-Based Zone Inspection** — Photo, hologram, serial number, and logo zones are verified independently
- **Anti-Spoof Detection** — Frequency-domain and statistical analysis detects Moire patterns, photocopies, screen replay attacks, print-scan artefacts, and JPEG compression artefacts without a secondary neural network
- **Adaptive Thresholds** — Exponential moving average drift detection keeps false-positive and false-negative rates stable across deployment environments
- **Calibrated Risk Scoring** — ROC curve analysis and isotonic regression produce statistically justified LOW, MEDIUM, and HIGH risk classifications
- **Cryptographic Audit Trail** — Every verification event is signed with RSA-PSS and ECDSA and appended to a hash-chained append-only log
- **Descriptor Cache** — LRU cache with file-modification-time invalidation eliminates redundant neural forward passes on stable reference databases
- **No Retraining Required** — Uses the pre-trained SuperPoint model directly; no labelled fraud dataset needed

---

## Requirements

```
torch>=1.12.0
torchvision>=0.13.0
opencv-python-headless>=4.5.0
numpy>=1.22.0
scipy>=1.8.0
streamlit>=1.28.0
Pillow>=9.0.0
cryptography>=41.0.0
scikit-learn>=1.2.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

**1. Obtain SuperPoint weights**

Download the pre-trained weights file `superpoint_v1.pth` and place it in the project root.

**2. Prepare a reference database**

Create a `database/` directory in the project root and add genuine reference document images (JPG or PNG).

```
project/
    demo_app.py
    script_v3.py
    superpoint_v1.pth
    requirements.txt
    database/
        reference_001.jpg
        reference_002.jpg
```

**3. (Optional) Generate signing keys**

```python
from script_v3 import LogSigner
LogSigner.generate_keys("./keys")
```

**4. Launch the interface**

```bash
streamlit run demo_app.py
```

---

## Programmatic Usage

```python
from script_v3 import ForensicVerificationEngine

engine = ForensicVerificationEngine(
    weights_path="superpoint_v1.pth",
    conf_thresh=0.005,
    nms_dist=4,
    cuda=False,
    run_multiscale=True,
    run_region_verification=True,
    run_anti_spoof=True,
    audit_log_dir="./audit_logs",
    chain_store_dir="./audit_store",
)

report = engine.verify(
    query_path="query.png",
    database_dir="./database",
    match_threshold=0.70,
    max_keypoints=1000,
)

print(report.risk_level.value)        # LOW | MEDIUM | HIGH
print(report.fraud_probability)       # 0.0 – 1.0
print(report.authenticity_score)      # 0.0 – 1.0
print(report.audit_log_path)          # path to signed JSON log
```

**Batch verification:**

```python
reports = engine.verify_batch(
    query_paths=["doc1.png", "doc2.png", "doc3.png"],
    database_dir="./database",
)
```

---

## Fraud Score Weights

The composite authenticity score combines six signals:

| Signal | Weight |
|--------|--------|
| Descriptor similarity | 30% |
| Geometric inlier ratio | 22% |
| Reprojection error | 13% |
| Tamper deviation | 15% |
| Multi-scale consistency | 10% |
| Anti-spoof penalty | 10% |

---

## Audit Chain Verification

Every verification event is appended to a hash-chained JSONL file. To verify the integrity of the full chain:

```python
from script_v3 import AuditChainStore

store = AuditChainStore("./audit_store")
valid, violations = store.verify_chain()

if valid:
    print("Chain intact")
else:
    for v in violations:
        print(v)
```

---

## Project Structure

```
project/
    script_v3.py          Core verification engine (15 layers)
    demo_app.py           Streamlit interface
    requirements.txt      Python dependencies
    database/             Reference document images
    audit_logs/           Per-session signed JSON audit logs
    audit_store/          Hash-chained append-only chain file
    keys/                 RSA and ECDSA key pairs (generated once)
    calibrator.json       Fitted score calibrator (optional)
    adaptive_thresholds.json  Persisted threshold state (optional)
```

---

## Limitations

- Operates on grayscale image data only; colour channels are not used in matching
- Requires a pre-built reference database of genuine documents
- Anti-spoof signals are signal-processing heuristics and may be circumvented by an adversary with knowledge of the detection methods
- No built-in regulatory compliance; not a substitute for certified identity verification systems

---

## 👨‍💻 Author

**Mensah Obed**  
[heavenzlebron7@gmail.com](mailto:heavenzlebron7@gmail.com)  
[linkedin.com/in/obed-mensah-87001237b](https://www.linkedin.com/in/obed-mensah-87001237b)
