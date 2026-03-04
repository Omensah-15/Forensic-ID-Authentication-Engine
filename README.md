# Forensic ID Authentication Engine

A production-grade forensic image verification system built on the SuperPoint deep learning architecture. The engine performs multi-layer structural analysis of identity documents and security patterns to detect counterfeits, localise tampering, and produce cryptographically signed audit records.

The system extracts nano-level keypoints from query images, validates geometric consistency against a trusted reference database, and applies anti-spoof analysis, multi-scale verification, and region-based zone inspection to produce a statistically calibrated fraud probability with a structured chain-of-custody audit log.

---

## Architecture Overview

The engine is composed of ten verification layers that execute sequentially on every query:

| Layer | Component | Purpose |
|---|---|---|
| 1 | Descriptor Matching | SuperPoint reciprocal cosine similarity |
| 2 | Geometric Consistency | RANSAC homography, inlier ratio, reprojection error |
| 3 | Tamper Localisation | Per-quadrant deviation heatmap |
| 4 | Multi-Scale Robustness | Three-resolution consistency check |
| 5 | Region Verification | Per-zone ID document analysis |
| 6 | Integrity Fingerprint | SHA-256 image and descriptor hash |
| 7 | Fraud Scoring | Weighted composite with isotonic calibration |
| 8 | Anti-Spoof Analysis | Moire, photocopy, screen replay, print-scan |
| 9 | Signed Audit Log | RSA-PSS 4096 and ECDSA P-384 dual signatures |
| 10 | Hash-Chained Storage | Append-only tamper-evident audit chain |

---

## Features

- **Sub-millimeter Accuracy** — Detects micro-patterns and structural features at the keypoint level across the full image surface
- **Geometric Consistency Validation** — RANSAC homography estimation rejects texture-copy counterfeits that lack coherent planar geometry
- **Tamper Localisation** — Spatial deviation heatmap identifies the specific image region where modification occurred
- **Multi-Scale Verification** — Matching at three resolutions detects resampling artefacts introduced during forgery production
- **Region-Based ID Zone Inspection** — Photo, hologram, serial number, and logo zones are verified independently
- **Anti-Spoof Detection** — Frequency-domain and statistical analysis detects Moire patterns, photocopies, screen replay attacks, and print-scan artefacts without a secondary neural network
- **Calibrated Risk Scoring** — ROC curve analysis and isotonic regression produce statistically justified LOW / MEDIUM / HIGH risk classifications
- **Cryptographic Audit Trail** — Every verification event is signed with RSA-PSS and ECDSA and appended to a hash-chained append-only log
- **No Retraining Required** — Uses the pre-trained SuperPoint model directly

---

## Demo Output

See [`notebook.ipynb`](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/aafee20046f3ec26850acc9f83f4d285b56b78ba/notebook.ipynb) for a full walkthrough.

**Query image — `query_030.png`**

![Query image](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/e2bba3fdf564d8adfc6a72358ddf824b31dc17e7/asset/query_030.png)

**Reference image — `pattern_014.png`**

![Reference image](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/5055cf5cd07158ef9015961ed9189ebf63c6a9f4/asset/pattern_014.png)

**Verification result**

![Result](https://github.com/Omensah-15/Image-Based-Anti-Counterfeit-Detection-with-SuperPoint/blob/9ef8d762412fc8830cf38fc21126b41aec2eed90/asset/Screenshot%202025-08-21%20162010.png)

The result displays the SuperPoint match visualisation between `query_030.png` and `pattern_014.png`. Coloured lines connect reciprocal keypoint matches across both images. The density and spatial distribution of matches across the full image surface, combined with a similarity score of 1.000, confirms a structurally authentic match. A genuine counterfeit would show sparse, geometrically inconsistent matches and a collapsed inlier ratio under RANSAC.

---
   
## 👨‍💻 Author

**Developed by Mensah Obed**
[Email](mailto:heavenzlebron7@gmail.com) 
[LinkedIn](https://www.linkedin.com/in/obed-mensah-87001237b)
