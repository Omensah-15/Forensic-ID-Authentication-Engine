"""
SuperPoint Forensic Verification Engine — Enterprise Edition
=============================================================
Adds four production modules on top of the base forensic pipeline:

  Module A  Digital signature of audit logs (RSA-PSS + ECDSA)
  Module B  Secure, hash-chained append-only audit storage
  Module C  Anti-spoof detection (Moiré, photocopy, screen-replay, print-scan)
  Module D  Confidence calibration via ROC / isotonic regression
"""

import os, hashlib, time, json, uuid, logging, math, struct
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Optional crypto imports — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("cryptography package not installed — log signing disabled. "
                    "Install with: pip install cryptography")

# ---------------------------------------------------------------------------
# Optional calibration imports
# ---------------------------------------------------------------------------
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed — calibration disabled. "
                    "Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


###############################################################################
# SECTION 1 — DATA STRUCTURES
###############################################################################

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


@dataclass
class RegionResult:
    name: str
    status: str
    confidence: float
    inlier_ratio: float
    matched_keypoints: int


@dataclass
class GeometricResult:
    inlier_count: int
    inlier_ratio: float
    reprojection_error: float
    homography_stability: float
    homography: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TamperResult:
    unmatched_ratio: float
    structural_deviation: float
    suspicious_quadrants: List[str]
    deviation_heatmap: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class MultiScaleResult:
    original_score: float
    downscale_score: float
    upscale_score: float
    consistency_score: float


@dataclass
class IntegrityFingerprint:
    image_sha256: str
    descriptor_hash: str
    timestamp: float
    session_id: str


@dataclass
class AntiSpoofResult:
    """Aggregated anti-spoof signals."""
    moire_score: float          # 0 = clean, 1 = strong Moiré
    photocopy_score: float      # 0 = likely original, 1 = likely copy
    screen_replay_score: float  # 0 = physical, 1 = likely screen display
    print_scan_score: float     # 0 = digital, 1 = print-scan artefact
    overall_spoof_probability: float
    flags: List[str]            # human-readable triggered flags


@dataclass
class ForensicReport:
    session_id: str
    query_path: str
    best_match_path: Optional[str]
    risk_level: RiskLevel
    fraud_probability: float
    authenticity_score: float
    descriptor_similarity: float
    geometric: Optional[GeometricResult]
    tamper: Optional[TamperResult]
    multiscale: Optional[MultiScaleResult]
    anti_spoof: Optional[AntiSpoofResult]
    region_results: List[RegionResult]
    fingerprint: IntegrityFingerprint
    audit_log_path: Optional[str] = None
    audit_signature: Optional[Dict] = None


###############################################################################
# SECTION 2 — NEURAL NETWORK
###############################################################################

class SuperPointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1,  c1, 3, 1, 1)
        self.conv1b = torch.nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = torch.nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = torch.nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = torch.nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = torch.nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = torch.nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = torch.nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = torch.nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = torch.nn.Conv2d(c5, d1, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.conv1a(x)); x = self.relu(self.conv1b(x)); x = self.pool(x)
        x = self.relu(self.conv2a(x)); x = self.relu(self.conv2b(x)); x = self.pool(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x))
        semi = self.convPb(self.relu(self.convPa(x)))
        desc = self.convDb(self.relu(self.convDa(x)))
        desc = desc.div(torch.unsqueeze(torch.norm(desc, p=2, dim=1), 1))
        return semi, desc


###############################################################################
# SECTION 3 — IMAGE SCANNER
###############################################################################

class ImageScanner:
    """SuperPoint feature extractor — vectorised NMS, optional GPU half-precision."""

    def __init__(self, weights_path: str, conf_thresh: float = 0.015,
                 nms_dist: int = 4, border_remove: int = 4, cuda: bool = False):
        self.conf_thresh   = conf_thresh
        self.nms_dist      = nms_dist
        self.cell          = 8
        self.border_remove = border_remove
        self.cuda          = cuda
        self.device        = torch.device("cuda" if cuda else "cpu")
        self.net           = SuperPointNet()
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.net = self.net.to(self.device).eval()

    def load_image(self, path: str, target_size=None) -> np.ndarray:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise IOError(f"Cannot read: {path}")
        if target_size:
            g = cv2.resize(g, target_size, interpolation=cv2.INTER_AREA)
        return g.astype("float32") / 255.0

    def scan_image(self, path: str, target_size=None,
                   max_keypoints: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        img  = self.load_image(path, target_size)
        H, W = img.shape
        inp  = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
        if self.cuda:
            inp = inp.half(); self.net = self.net.half()

        with torch.no_grad():
            semi, coarse = self.net(inp)

        semi   = semi.cpu().numpy().squeeze()
        coarse = coarse.cpu().numpy().squeeze()
        dense  = np.exp(semi - np.max(semi))
        dense /= np.sum(dense, axis=0) + 1e-8
        nodust = dense[:-1]
        Hc, Wc = semi.shape[1], semi.shape[2]
        heatmap = (nodust.transpose(1, 2, 0).reshape(Hc, Wc, 8, 8)
                         .transpose(0, 2, 1, 3).reshape(Hc * 8, Wc * 8))

        ys, xs = np.where(heatmap >= self.conf_thresh)
        if not len(xs):
            return np.zeros((3, 0)), np.zeros((256, 0))

        pts = np.array([xs, ys, heatmap[ys, xs]], dtype=float)
        grid = np.zeros((H, W))
        grid[pts[1].astype(int), pts[0].astype(int)] = pts[2]
        mf   = maximum_filter(grid, size=self.nms_dist * 2 + 1)
        ky, kx = np.where((grid == mf) & (grid >= self.conf_thresh))
        pts  = np.array([kx, ky, grid[ky, kx]], dtype=float)

        if pts.shape[1] > max_keypoints:
            pts = pts[:, np.argsort(-pts[2])[:max_keypoints]]

        b   = self.border_remove
        pts = pts[:, ~((pts[0] < b) | (pts[0] >= W - b) |
                       (pts[1] < b) | (pts[1] >= H - b))]
        if not pts.shape[1]:
            return pts, np.zeros((256, 0))

        samp = pts[:2].copy().T
        samp[:, 0] = (samp[:, 0] / (W - 1)) * 2 - 1
        samp[:, 1] = (samp[:, 1] / (H - 1)) * 2 - 1
        st = torch.from_numpy(samp).float().unsqueeze(0).unsqueeze(0).to(self.device)
        ct = torch.from_numpy(coarse).float().unsqueeze(0).to(self.device)
        if self.cuda:
            ct = ct.half(); st = st.half()

        desc = torch.nn.functional.grid_sample(
            ct, st, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze().cpu().numpy()
        if desc.ndim == 1:
            desc = desc.reshape(-1, 1)
        desc /= np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8
        return pts, desc


###############################################################################
# SECTION 4 — MODULE A: DIGITAL LOG SIGNING
###############################################################################

class LogSigner:
    """
    Signs serialised audit log payloads using RSA-PSS and ECDSA (P-384).
    Both signatures are embedded in the log file so verification can use
    either algorithm depending on the relying party's trust store.

    Key generation
    --------------
    Call LogSigner.generate_keys(output_dir) once during deployment to
    produce PEM-encoded key pairs.  Store private keys in a hardware security
    module or secrets manager; distribute only the public keys.

    Signature scheme
    ----------------
    The canonical log payload is serialised to UTF-8 JSON with sorted keys and
    no trailing whitespace. The SHA-256 digest of that byte string is signed
    separately by both algorithms. Signatures are hex-encoded and appended as
    a 'signatures' block; the payload itself is never modified.
    """

    RSA_KEY_SIZE  = 4096
    EC_CURVE      = ec.SECP384R1 if CRYPTO_AVAILABLE else None

    def __init__(self, rsa_private_key_pem: bytes,
                 ec_private_key_pem: bytes):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required for log signing.")
        self._rsa_key = serialization.load_pem_private_key(
            rsa_private_key_pem, password=None, backend=default_backend()
        )
        self._ec_key = serialization.load_pem_private_key(
            ec_private_key_pem, password=None, backend=default_backend()
        )

    # ------------------------------------------------------------------
    @staticmethod
    def generate_keys(output_dir: str = "./keys") -> Dict[str, str]:
        """
        Generate and persist RSA-4096 and ECDSA P-384 key pairs.
        Returns a dict of file paths.  Call once at deployment time.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required.")
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # RSA
        rsa_priv = rsa.generate_private_key(
            public_exponent=65537, key_size=LogSigner.RSA_KEY_SIZE,
            backend=default_backend()
        )
        for name, key, enc in [
            ("rsa_private.pem", rsa_priv,
             serialization.Encoding.PEM,),
        ]:
            pass  # structured below

        rsa_priv_pem = rsa_priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        )
        rsa_pub_pem  = rsa_priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # ECDSA
        ec_priv = ec.generate_private_key(ec.SECP384R1(), default_backend())
        ec_priv_pem = ec_priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        )
        ec_pub_pem  = ec_priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        )

        for fname, data in [("rsa_private.pem", rsa_priv_pem),
                             ("rsa_public.pem",  rsa_pub_pem),
                             ("ec_private.pem",  ec_priv_pem),
                             ("ec_public.pem",   ec_pub_pem)]:
            p = os.path.join(output_dir, fname)
            with open(p, "wb") as f:
                f.write(data)
            paths[fname] = p

        logger.info("Key pairs written to %s", output_dir)
        return paths

    # ------------------------------------------------------------------
    def sign_payload(self, payload_bytes: bytes) -> Dict[str, str]:
        """
        Sign the canonical payload bytes.
        Returns a dict with both hex-encoded signatures and the payload digest.
        """
        digest = hashlib.sha256(payload_bytes).digest()

        rsa_sig = self._rsa_key.sign(
            digest,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH),
            hashes.Prehashed(hashes.SHA256())
        )
        ec_sig = self._ec_key.sign(digest, ec.ECDSA(hashes.Prehashed(hashes.SHA256())))

        return {
            "algorithm_rsa": "RSA-PSS-SHA256-4096",
            "algorithm_ec":  "ECDSA-P384-SHA256",
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "rsa_signature":  rsa_sig.hex(),
            "ec_signature":   ec_sig.hex(),
            "signed_at_utc":  time.time(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def verify_signatures(payload_bytes: bytes,
                          signature_block: Dict,
                          rsa_public_key_pem: bytes,
                          ec_public_key_pem: bytes) -> Dict[str, bool]:
        """
        Verify both signatures against a payload byte string.
        Returns {'rsa_valid': bool, 'ec_valid': bool}.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required.")

        digest   = hashlib.sha256(payload_bytes).digest()
        rsa_pub  = serialization.load_pem_public_key(rsa_public_key_pem,
                                                      backend=default_backend())
        ec_pub   = serialization.load_pem_public_key(ec_public_key_pem,
                                                      backend=default_backend())

        results = {"rsa_valid": False, "ec_valid": False}

        try:
            rsa_pub.verify(
                bytes.fromhex(signature_block["rsa_signature"]),
                digest,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH),
                hashes.Prehashed(hashes.SHA256())
            )
            results["rsa_valid"] = True
        except Exception:
            pass

        try:
            ec_pub.verify(
                bytes.fromhex(signature_block["ec_signature"]),
                digest,
                ec.ECDSA(hashes.Prehashed(hashes.SHA256()))
            )
            results["ec_valid"] = True
        except Exception:
            pass

        return results


###############################################################################
# SECTION 5 — MODULE B: SECURE HASH-CHAINED AUDIT STORAGE
###############################################################################

class AuditChainStore:
    """
    Append-only, hash-chained audit log store.

    Storage layout
    --------------
    Each entry is a JSON object written as a single line to
    <store_dir>/audit_chain.jsonl  (newline-delimited JSON).

    Chain integrity
    ---------------
    Every entry carries the SHA-256 hash of the previous entry's raw line,
    forming a linked chain analogous to a blockchain.  Tampering with any
    historical record breaks every subsequent link, which is detectable by
    verify_chain().

    Immutability enforcement
    ------------------------
    The JSONL file is opened in append-only mode ('a').  On POSIX systems
    the store directory can additionally be made append-only at the OS level
    (chattr +a on Linux) for stronger enforcement.  The in-process guard
    prevents any write path other than append().

    Index file
    ----------
    <store_dir>/index.json maps session_id -> byte offset in the chain file
    for O(1) retrieval without scanning the full chain.
    """

    CHAIN_FILENAME = "audit_chain.jsonl"
    INDEX_FILENAME = "chain_index.json"
    GENESIS_HASH   = "0" * 64   # sentinel previous-hash for the first entry

    def __init__(self, store_dir: str = "./audit_store"):
        self.store_dir  = Path(store_dir)
        self.chain_path = self.store_dir / self.CHAIN_FILENAME
        self.index_path = self.store_dir / self.INDEX_FILENAME
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    # ------------------------------------------------------------------
    def _load_index(self):
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._index: Dict[str, int] = json.load(f)
        else:
            self._index = {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f)

    # ------------------------------------------------------------------
    def _last_hash(self) -> str:
        """Return the SHA-256 hash of the last line in the chain file."""
        if not self.chain_path.exists():
            return self.GENESIS_HASH
        last_line = b""
        with open(self.chain_path, "rb") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        return hashlib.sha256(last_line).hexdigest() if last_line else self.GENESIS_HASH

    # ------------------------------------------------------------------
    def append(self, payload: dict, signature_block: Optional[dict] = None) -> str:
        """
        Append a new audit entry to the chain.
        Returns the entry hash (SHA-256 of the written line).
        """
        entry_id    = payload.get("session_id", str(uuid.uuid4())[:12])
        prev_hash   = self._last_hash()
        entry_index = len(self._index)

        record = {
            "entry_index":  entry_index,
            "prev_hash":    prev_hash,
            "payload":      payload,
            "signatures":   signature_block or {},
            "written_utc":  time.time(),
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        line_bytes = line.encode("utf-8")
        entry_hash = hashlib.sha256(line_bytes).hexdigest()

        byte_offset = (self.chain_path.stat().st_size
                       if self.chain_path.exists() else 0)

        with open(self.chain_path, "a", encoding="utf-8") as f:
            f.write(line)

        self._index[entry_id] = byte_offset
        self._save_index()
        logger.info("Chain entry %d written (hash: %s...)", entry_index, entry_hash[:16])
        return entry_hash

    # ------------------------------------------------------------------
    def retrieve(self, session_id: str) -> Optional[dict]:
        """Retrieve a single entry by session_id using the index."""
        offset = self._index.get(session_id)
        if offset is None:
            return None
        with open(self.chain_path, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline())

    # ------------------------------------------------------------------
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """
        Walk the full chain and verify every prev_hash link.
        Returns (chain_valid: bool, list_of_violation_messages).
        """
        violations  = []
        prev_hash   = self.GENESIS_HASH
        prev_line   = None

        with open(self.chain_path, "rb") as f:
            for lineno, raw_line in enumerate(f, start=1):
                raw_line = raw_line.rstrip(b"\n")
                if not raw_line:
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    violations.append(f"Line {lineno}: JSON parse error — {e}")
                    continue

                declared_prev = record.get("prev_hash", "")
                if lineno > 1:
                    if declared_prev != prev_hash:
                        violations.append(
                            f"Line {lineno}: chain break — declared prev_hash "
                            f"{declared_prev[:16]}... != computed {prev_hash[:16]}..."
                        )

                prev_hash = hashlib.sha256(raw_line + b"\n").hexdigest()
                prev_line = raw_line

        valid = len(violations) == 0
        if valid:
            logger.info("Chain verification passed (%d entries).", lineno)
        else:
            logger.warning("Chain verification found %d violation(s).", len(violations))
        return valid, violations


###############################################################################
# SECTION 6 — MODULE C: ANTI-SPOOF DETECTION
###############################################################################

class AntiSpoofDetector:
    """
    Detects four classes of presentation attack without requiring a dedicated
    neural network — all methods operate on spatial frequency and statistical
    properties of the input image.

    Moire pattern detection
    -----------------------
    Periodic interference fringes from scanning a printed halftone appear as
    strong off-axis peaks in the magnitude spectrum of the DFT.  The ratio of
    peak energy outside the DC neighbourhood to total energy is the Moiré score.

    Photocopy detection
    -------------------
    Photocopied documents exhibit characteristic histogram flattening, reduced
    dynamic range in shadow/highlight regions, and elevated Laplacian noise.
    A combination of inter-quartile histogram spread and Laplacian variance
    normalised by mean intensity forms the photocopy score.

    Screen replay detection
    -----------------------
    Displaying a document on a screen and recapturing it introduces:
      (a) a visible scan-line / subpixel grid detectable via horizontal
          frequency peaks in the DFT row spectrum, and
      (b) reduced local contrast at high spatial frequencies (blur from
          re-capture through the optical system).
    Both signals are combined into the screen replay score.

    Print-scan artefact detection
    -----------------------------
    Print-scan workflows introduce dot-gain, halftone rosettes, and
    characteristic DCT-coefficient distributions.  The presence of block
    artefacts (8x8 DCT grid) and halftone frequencies are measured.
    """

    # Thresholds for flag triggering (tunable per deployment)
    MOIRE_THRESH        = 0.18
    PHOTOCOPY_THRESH    = 0.55
    SCREEN_REPLAY_THRESH = 0.40
    PRINT_SCAN_THRESH   = 0.35

    # ------------------------------------------------------------------
    def analyse(self, image_path: str) -> AntiSpoofResult:
        bgr  = cv2.imread(image_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None else None
        if gray is None:
            return AntiSpoofResult(0.0, 0.0, 0.0, 0.0, 0.0, ["IMAGE_LOAD_FAILED"])

        moire   = self._moire_score(gray)
        photo   = self._photocopy_score(gray)
        screen  = self._screen_replay_score(gray)
        pscan   = self._print_scan_score(gray)

        # Weighted combination — screen replay and Moiré are highest-impact signals
        overall = float(np.clip(
            0.30 * moire + 0.25 * photo + 0.30 * screen + 0.15 * pscan, 0, 1
        ))

        flags = []
        if moire  >= self.MOIRE_THRESH:        flags.append("MOIRE_DETECTED")
        if photo  >= self.PHOTOCOPY_THRESH:    flags.append("PHOTOCOPY_SUSPECTED")
        if screen >= self.SCREEN_REPLAY_THRESH: flags.append("SCREEN_REPLAY_SUSPECTED")
        if pscan  >= self.PRINT_SCAN_THRESH:   flags.append("PRINT_SCAN_ARTEFACT")

        return AntiSpoofResult(moire, photo, screen, pscan, overall, flags)

    # ------------------------------------------------------------------
    def _dft_magnitude(self, gray: np.ndarray) -> np.ndarray:
        """Return log-scaled, shifted DFT magnitude spectrum."""
        f   = np.fft.fft2(gray.astype(np.float32))
        fs  = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fs))
        return mag

    # ------------------------------------------------------------------
    def _moire_score(self, gray: np.ndarray) -> float:
        """
        Ratio of spectral energy in off-axis annular region to total energy.
        Genuine captures have smooth spectra; Moiré fringes create sharp
        concentric or diagonal peaks well outside the DC neighbourhood.
        """
        mag   = self._dft_magnitude(gray)
        H, W  = mag.shape
        cy, cx = H // 2, W // 2

        # Build radial distance map
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        dist  = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        r_max = min(cy, cx)

        # DC neighbourhood to exclude (inner 5% of radius)
        dc_mask       = dist < (r_max * 0.05)
        # Outer noise boundary (beyond 85% of radius)
        outer_mask    = dist > (r_max * 0.85)
        analysis_mask = ~dc_mask & ~outer_mask

        total_energy  = float(mag[analysis_mask].sum()) + 1e-8

        # Peak detection in 8 radial sectors — high narrow peaks indicate Moiré
        sector_peaks = []
        for s in range(8):
            angle_start = math.radians(s * 45 - 22.5)
            angle_end   = math.radians(s * 45 + 22.5)
            angle_map   = np.arctan2(y_idx - cy, x_idx - cx)
            sector_mask = (analysis_mask &
                           (angle_map >= angle_start) & (angle_map < angle_end))
            if sector_mask.sum() > 0:
                sector_vals = mag[sector_mask]
                peak_ratio  = float(sector_vals.max() / (sector_vals.mean() + 1e-8))
                sector_peaks.append(peak_ratio)

        # Normalise: genuine images typically have peak_ratio < 4; Moiré > 8
        avg_peak = float(np.mean(sector_peaks)) if sector_peaks else 0.0
        score    = float(np.clip((avg_peak - 3.0) / 10.0, 0.0, 1.0))
        return score

    # ------------------------------------------------------------------
    def _photocopy_score(self, gray: np.ndarray) -> float:
        """
        Three-signal photocopy indicator:
          1. Histogram IQR spread (photocopies flatten mid-tones)
          2. Shadow/highlight clipping (photocopies clip more aggressively)
          3. Laplacian noise relative to mean (copies add grain in flat areas)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        cdf  = np.cumsum(hist)

        q25 = int(np.searchsorted(cdf, 0.25))
        q75 = int(np.searchsorted(cdf, 0.75))
        iqr_norm = (q75 - q25) / 255.0   # wider IQR = more original contrast

        shadow_clip    = float(hist[:10].sum())
        highlight_clip = float(hist[245:].sum())
        clip_score     = float(np.clip((shadow_clip + highlight_clip) / 0.15, 0, 1))

        lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_int = float(gray.mean()) + 1e-8
        noise_score = float(np.clip(lap_var / (mean_int * 50.0), 0, 1))

        # Low IQR + high clipping + high noise => photocopy
        score = float(np.clip(
            (1.0 - iqr_norm) * 0.4 + clip_score * 0.35 + noise_score * 0.25,
            0.0, 1.0
        ))
        return score

    # ------------------------------------------------------------------
    def _screen_replay_score(self, gray: np.ndarray) -> float:
        """
        Horizontal scan-line grid produces periodic peaks in the row-averaged
        DFT spectrum.  High-frequency blur from re-capture optics reduces the
        energy ratio in the top 20% of the spatial frequency band.
        """
        # Scan-line peak analysis (row FFT)
        row_spectrum = np.abs(np.fft.rfft(gray.astype(np.float32), axis=1))
        row_mean     = row_spectrum.mean(axis=0)
        N            = len(row_mean)
        low_energy   = row_mean[:N // 5].sum() + 1e-8
        high_energy  = row_mean[N // 5:].sum()
        hf_ratio     = float(high_energy / low_energy)

        # Screen displays bleed energy into low frequencies; HF ratio drops
        # Genuine documents have flatter row spectra
        scanline_score = float(np.clip(1.0 - hf_ratio / 2.0, 0.0, 1.0))

        # Re-capture blur: compare Laplacian sharpness to expected range
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = float(np.clip(1.0 - lap_var / 800.0, 0.0, 1.0))

        return float(np.clip(scanline_score * 0.6 + blur_score * 0.4, 0, 1))

    # ------------------------------------------------------------------
    def _print_scan_score(self, gray: np.ndarray) -> float:
        """
        Print-scan artefact detection via 8x8 block DCT energy distribution.
        JPEG/halftone printing introduces discontinuities at 8-pixel boundaries
        that manifest as elevated energy in specific DCT coefficients.
        """
        H, W = gray.shape
        # Pad to multiple of 8
        pH = (H // 8) * 8
        pW = (W // 8) * 8
        if pH == 0 or pW == 0:
            return 0.0
        cropped = gray[:pH, :pW].astype(np.float32) - 128.0

        block_scores = []
        for y in range(0, pH, 8):
            for x in range(0, pW, 8):
                block   = cropped[y:y + 8, x:x + 8]
                dct_blk = cv2.dct(block)
                # DC coefficient energy vs total block energy
                dc_energy    = float(dct_blk[0, 0] ** 2)
                total_energy = float((dct_blk ** 2).sum()) + 1e-8
                dc_ratio     = dc_energy / total_energy
                # Print-scan shifts energy toward DC (blocking artefacts)
                block_scores.append(dc_ratio)

        mean_dc_ratio = float(np.mean(block_scores)) if block_scores else 0.5
        # Genuine digital captures: DC ratio ~0.3–0.5; print-scan: ~0.65+
        score = float(np.clip((mean_dc_ratio - 0.45) / 0.35, 0.0, 1.0))
        return score


###############################################################################
# SECTION 7 — MODULE D: CONFIDENCE CALIBRATION
###############################################################################

class FraudScoreCalibrator:
    """
    Calibrates raw fraud probability scores using a validation dataset,
    ROC curve analysis, and isotonic regression.

    Workflow
    --------
    1.  Collect ground-truth labels (0 = genuine, 1 = fraud) and raw
        fraud_probability scores from the engine on a held-out validation set.
    2.  Call calibrator.fit(scores, labels).
    3.  Calibrator computes:
          - ROC AUC
          - Optimal operating threshold (Youden J statistic)
          - 90th-percentile threshold for high-precision deployment
          - Isotonic regression mapping for monotone score calibration
    4.  All subsequent calls to calibrator.calibrate(score) return a
        calibrated probability with statistically justified risk thresholds.

    Persistence
    -----------
    calibrator.save(path) / FraudScoreCalibrator.load(path)
    Serialises thresholds and isotonic lookup table to JSON.
    """

    DEFAULT_LOW_THRESH    = 0.25  # below this -> LOW risk
    DEFAULT_MEDIUM_THRESH = 0.60  # below this -> MEDIUM risk, else HIGH

    def __init__(self):
        self.is_fitted      = False
        self.roc_auc        = None
        self.optimal_thresh = self.DEFAULT_LOW_THRESH
        self.high_thresh    = self.DEFAULT_MEDIUM_THRESH
        self._iso_x: Optional[np.ndarray] = None
        self._iso_y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, raw_scores: List[float], labels: List[int]) -> Dict:
        """
        Fit calibration on validation data.

        Parameters
        ----------
        raw_scores : list of float  — raw fraud_probability from the engine
        labels     : list of int    — 0 = genuine, 1 = fraud

        Returns
        -------
        dict with roc_auc, optimal_threshold, high_precision_threshold.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for calibration.")

        scores = np.array(raw_scores, dtype=float)
        labs   = np.array(labels,     dtype=int)

        if len(np.unique(labs)) < 2:
            raise ValueError("Calibration requires both positive and negative examples.")

        # ROC curve
        fpr, tpr, thresholds = roc_curve(labs, scores)
        self.roc_auc = float(roc_auc_score(labs, scores))

        # Youden J — maximise (TPR - FPR)
        j_scores         = tpr - fpr
        best_idx         = int(np.argmax(j_scores))
        self.optimal_thresh = float(thresholds[best_idx])

        # High-precision threshold: FPR <= 0.05
        high_prec_mask = fpr <= 0.05
        if high_prec_mask.any():
            self.high_thresh = float(thresholds[high_prec_mask][-1])
        else:
            self.high_thresh = float(thresholds[int(np.argmin(fpr))])

        # Isotonic regression for probability calibration
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, labs)
        # Store lookup table — avoids pickling the sklearn object
        probe_x        = np.linspace(0, 1, 1000)
        self._iso_x    = probe_x
        self._iso_y    = iso.predict(probe_x)
        self.is_fitted = True

        result = {
            "roc_auc":               round(self.roc_auc, 4),
            "optimal_threshold":     round(self.optimal_thresh, 4),
            "high_precision_threshold": round(self.high_thresh, 4),
            "n_samples":             len(labs),
            "prevalence":            round(float(labs.mean()), 4),
        }
        logger.info("Calibration complete: AUC=%.4f  opt_thresh=%.4f  hi_thresh=%.4f",
                    self.roc_auc, self.optimal_thresh, self.high_thresh)
        return result

    # ------------------------------------------------------------------
    def calibrate(self, raw_score: float) -> Tuple[float, RiskLevel]:
        """
        Map a raw fraud_probability to a calibrated value and risk level.
        Falls back to default thresholds if calibrator has not been fitted.
        """
        if self.is_fitted and self._iso_x is not None:
            cal = float(np.interp(raw_score, self._iso_x, self._iso_y))
        else:
            cal = float(raw_score)

        low_t = self.optimal_thresh if self.is_fitted else self.DEFAULT_LOW_THRESH
        hi_t  = self.high_thresh    if self.is_fitted else self.DEFAULT_MEDIUM_THRESH

        if cal < low_t:
            risk = RiskLevel.LOW
        elif cal < hi_t:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.HIGH

        return cal, risk

    # ------------------------------------------------------------------
    def save(self, path: str):
        state = {
            "is_fitted":      self.is_fitted,
            "roc_auc":        self.roc_auc,
            "optimal_thresh": self.optimal_thresh,
            "high_thresh":    self.high_thresh,
            "iso_x": self._iso_x.tolist() if self._iso_x is not None else None,
            "iso_y": self._iso_y.tolist() if self._iso_y is not None else None,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Calibrator saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "FraudScoreCalibrator":
        with open(path) as f:
            state = json.load(f)
        c = cls()
        c.is_fitted      = state["is_fitted"]
        c.roc_auc        = state["roc_auc"]
        c.optimal_thresh = state["optimal_thresh"]
        c.high_thresh    = state["high_thresh"]
        c._iso_x = np.array(state["iso_x"]) if state["iso_x"] else None
        c._iso_y = np.array(state["iso_y"]) if state["iso_y"] else None
        return c


###############################################################################
# SECTION 8 — GEOMETRIC CONSISTENCY
###############################################################################

def compute_geometric_consistency(q_pts, db_pts, matches) -> GeometricResult:
    if len(matches) < 4:
        return GeometricResult(0, 0.0, float("inf"), 0.0)

    src = np.float32([[q_pts[0, i],  q_pts[1, i]]  for i, j, _ in matches])
    dst = np.float32([[db_pts[0, j], db_pts[1, j]] for i, j, _ in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return GeometricResult(0, 0.0, float("inf"), 0.0)

    inlier_mask  = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(len(matches), 1)

    if inlier_count > 0:
        si = src[inlier_mask]; di = dst[inlier_mask]
        sh = np.hstack([si, np.ones((len(si), 1))])
        pr = (H @ sh.T).T; pr /= pr[:, 2:3] + 1e-8
        reproj_err = float(np.linalg.norm(pr[:, :2] - di, axis=1).mean())
    else:
        reproj_err = float("inf")

    try:
        Hn  = H / (np.linalg.norm(H) + 1e-8)
        sv  = np.linalg.svd(Hn, compute_uv=False)
        stb = float(np.clip(sv[-1] / (sv[0] + 1e-8) * 10, 0, 1))
    except Exception:
        stb = 0.0

    return GeometricResult(inlier_count, inlier_ratio, reproj_err, stb, H)


###############################################################################
# SECTION 9 — TAMPER LOCALISATION
###############################################################################

def compute_tamper_localization(q_pts, db_pts, matches,
                                inlier_mask_bool, image_shape) -> TamperResult:
    H_i, W_i = image_shape[:2]
    heatmap   = np.zeros((H_i, W_i), dtype=np.float32)

    if inlier_mask_bool is None or len(inlier_mask_bool) != len(matches):
        inlier_mask_bool = np.ones(len(matches), dtype=bool)

    outlier_idx     = np.where(~inlier_mask_bool)[0]
    unmatched_ratio = (q_pts.shape[1] - len(matches)) / max(q_pts.shape[1], 1)
    scores          = []

    for idx in outlier_idx:
        i, j, s = matches[idx]
        x = int(np.clip(q_pts[0, i], 0, W_i - 1))
        y = int(np.clip(q_pts[1, i], 0, H_i - 1))
        dev = float(1.0 - s)
        heatmap[y, x] += dev
        scores.append(dev)

    struct_dev = float(np.mean(scores)) if scores else 0.0

    if heatmap.max() > 0:
        heatmap     = cv2.GaussianBlur(heatmap, (15, 15), 0)
        hm8         = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(hm8, cv2.COLORMAP_HOT)
    else:
        heatmap_vis = np.zeros((H_i, W_i, 3), dtype=np.uint8)

    mid_y, mid_x = H_i // 2, W_i // 2
    quads = {
        "top-left":     (slice(None, mid_y), slice(None, mid_x)),
        "top-right":    (slice(None, mid_y), slice(mid_x, None)),
        "bottom-left":  (slice(mid_y, None), slice(None, mid_x)),
        "bottom-right": (slice(mid_y, None), slice(mid_x, None)),
    }
    suspicious = [n for n, (sy, sx) in quads.items()
                  if float(heatmap[sy, sx].mean()) > 30.0]

    return TamperResult(unmatched_ratio, struct_dev, suspicious, heatmap_vis)


###############################################################################
# SECTION 10 — MULTI-SCALE VERIFICATION
###############################################################################

def compute_multiscale_verification(scanner, q_path, db_path,
                                    base_size=(640, 480)) -> MultiScaleResult:
    def score_at(sz):
        p1, d1 = scanner.scan_image(q_path,  target_size=sz)
        p2, d2 = scanner.scan_image(db_path, target_size=sz)
        if d1.shape[1] == 0 or d2.shape[1] == 0:
            return 0.0
        q = d1.T / (np.linalg.norm(d1.T, axis=1, keepdims=True) + 1e-8)
        d = d2.T / (np.linalg.norm(d2.T, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(q, d.T)
        fwd = np.argmax(sim, axis=1); bwd = np.argmax(sim, axis=0)
        r   = [(i, fwd[i], sim[i, fwd[i]]) for i in range(len(fwd)) if bwd[fwd[i]] == i]
        return float(np.mean([s for _, _, s in r])) if r else 0.0

    W, H = base_size
    o, dn, up = score_at(base_size), score_at((W // 2, H // 2)), score_at((W * 2, H * 2))
    return MultiScaleResult(o, dn, up, float(np.std([o, dn, up])))


###############################################################################
# SECTION 11 — REGION VERIFICATION
###############################################################################

REGION_DEFINITIONS = {
    "photo":         (0.00, 0.00, 0.35, 1.00),
    "hologram":      (0.55, 0.00, 0.75, 0.50),
    "serial_number": (0.35, 0.75, 1.00, 1.00),
    "logo":          (0.35, 0.00, 0.55, 0.35),
}


def _desc_from_array(scanner, arr):
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
    Ha, Wa = gray.shape
    inp = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(scanner.device)
    with torch.no_grad():
        semi, coarse = scanner.net(inp)
    semi   = semi.cpu().numpy().squeeze()
    coarse = coarse.cpu().numpy().squeeze()
    dense  = np.exp(semi - np.max(semi)); dense /= np.sum(dense, axis=0) + 1e-8
    nodust = dense[:-1]; Hc, Wc = semi.shape[1], semi.shape[2]
    hm = (nodust.transpose(1, 2, 0).reshape(Hc, Wc, 8, 8)
                .transpose(0, 2, 1, 3).reshape(Hc * 8, Wc * 8))
    ys, xs = np.where(hm >= scanner.conf_thresh)
    if not len(xs):
        return np.zeros((3, 0)), np.zeros((256, 0))
    pts  = np.array([xs, ys, hm[ys, xs]], dtype=float)
    samp = pts[:2].copy().T
    samp[:, 0] = (samp[:, 0] / max(Wa - 1, 1)) * 2 - 1
    samp[:, 1] = (samp[:, 1] / max(Ha - 1, 1)) * 2 - 1
    st   = torch.from_numpy(samp).float().unsqueeze(0).unsqueeze(0).to(scanner.device)
    ct   = torch.from_numpy(coarse).float().unsqueeze(0).to(scanner.device)
    desc = torch.nn.functional.grid_sample(
        ct, st, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze().cpu().numpy()
    if desc.ndim == 1: desc = desc.reshape(-1, 1)
    desc /= np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8
    return pts, desc


def verify_regions(scanner, q_bgr, db_bgr,
                   region_defs=REGION_DEFINITIONS) -> List[RegionResult]:
    results = []
    H,  W  = q_bgr.shape[:2]; dH, dW = db_bgr.shape[:2]
    for name, (nx0, ny0, nx1, ny1) in region_defs.items():
        qc = q_bgr [int(ny0*H):int(ny1*H),   int(nx0*W):int(nx1*W)]
        dc = db_bgr[int(ny0*dH):int(ny1*dH), int(nx0*dW):int(nx1*dW)]
        if qc.size == 0 or dc.size == 0:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0)); continue
        qp, qd = _desc_from_array(scanner, qc)
        dp, dd = _desc_from_array(scanner, dc)
        if qd.shape[1] == 0 or dd.shape[1] == 0:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0)); continue
        qn  = qd.T / (np.linalg.norm(qd.T, axis=1, keepdims=True) + 1e-8)
        dn  = dd.T / (np.linalg.norm(dd.T, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(qn, dn.T)
        fwd = np.argmax(sim, axis=1); bwd = np.argmax(sim, axis=0)
        recip = [(i, fwd[i], sim[i, fwd[i]]) for i in range(len(fwd)) if bwd[fwd[i]] == i]
        if not recip:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0)); continue
        avg_s  = float(np.mean([s for _, _, s in recip]))
        geo    = compute_geometric_consistency(qp, dp, recip)
        conf   = avg_s * 0.6 + geo.inlier_ratio * 0.4
        status = "VERIFIED" if conf >= 0.75 else ("SUSPICIOUS" if conf >= 0.45 else "FAILED")
        results.append(RegionResult(name, status, conf, geo.inlier_ratio, len(recip)))
    return results


###############################################################################
# SECTION 12 — INTEGRITY FINGERPRINT
###############################################################################

def compute_fingerprint(image_path, desc, session_id) -> IntegrityFingerprint:
    with open(image_path, "rb") as f:
        img_hash = hashlib.sha256(f.read()).hexdigest()
    return IntegrityFingerprint(
        img_hash,
        hashlib.sha256(desc.tobytes()).hexdigest(),
        time.time(), session_id
    )


###############################################################################
# SECTION 13 — FRAUD SCORING (calibration-aware)
###############################################################################

def compute_fraud_score(desc_sim, geo, tamper, multiscale,
                        calibrator: Optional["FraudScoreCalibrator"] = None):
    s_desc   = float(np.clip(desc_sim, 0, 1))
    s_inlier = float(np.clip(geo.inlier_ratio, 0, 1)) if geo else 0.0
    s_reproj = float(np.clip(1.0 - (geo.reprojection_error / 20.0), 0, 1)) \
               if geo and geo.reprojection_error < float("inf") else 0.0
    s_tamper = float(np.clip(1.0 - tamper.unmatched_ratio
                                  - tamper.structural_deviation, 0, 1)) \
               if tamper else 0.5
    s_scale  = float(np.clip(1.0 - (multiscale.consistency_score * 5), 0, 1)) \
               if multiscale else 0.5

    auth     = (0.30 * s_desc + 0.25 * s_inlier + 0.15 * s_reproj
                + 0.15 * s_tamper + 0.15 * s_scale)
    raw_fp   = float(np.clip(1.0 - auth, 0, 1))

    if calibrator is not None:
        cal_fp, risk = calibrator.calibrate(raw_fp)
    else:
        cal_fp = raw_fp
        risk   = (RiskLevel.LOW   if raw_fp < 0.25 else
                  RiskLevel.MEDIUM if raw_fp < 0.60 else
                  RiskLevel.HIGH)

    return auth, cal_fp, risk


###############################################################################
# SECTION 14 — AUDIT LOG WRITER (with signing + chain storage)
###############################################################################

def build_log_payload(report: ForensicReport) -> dict:
    def safe(v):
        if isinstance(v, np.ndarray): return "<ndarray>"
        if isinstance(v, Enum):       return v.value
        return v

    return {
        "session_id":            report.session_id,
        "timestamp_utc":         report.fingerprint.timestamp,
        "query_path":            report.query_path,
        "best_match_path":       report.best_match_path,
        "risk_level":            report.risk_level.value,
        "fraud_probability":     round(report.fraud_probability, 4),
        "authenticity_score":    round(report.authenticity_score, 4),
        "descriptor_similarity": round(report.descriptor_similarity, 4),
        "geometric": {k: safe(v) for k, v in asdict(report.geometric).items()}
                     if report.geometric else {},
        "tamper":    {k: safe(v) for k, v in asdict(report.tamper).items()}
                     if report.tamper else {},
        "multiscale":{k: safe(v) for k, v in asdict(report.multiscale).items()}
                     if report.multiscale else {},
        "anti_spoof":{k: safe(v) for k, v in asdict(report.anti_spoof).items()}
                     if report.anti_spoof else {},
        "regions":   [{k: safe(v) for k, v in asdict(r).items()}
                      for r in report.region_results],
        "fingerprint": {
            "image_sha256":    report.fingerprint.image_sha256,
            "descriptor_hash": report.fingerprint.descriptor_hash,
        },
    }


def write_audit_log(report: ForensicReport,
                    output_dir: str = "./audit_logs",
                    signer: Optional[LogSigner] = None,
                    chain_store: Optional[AuditChainStore] = None) -> Tuple[str, Optional[dict]]:
    """
    Write a signed, optionally chain-stored audit log.

    Returns (log_file_path, signature_block_or_None).
    """
    os.makedirs(output_dir, exist_ok=True)
    payload      = build_log_payload(report)
    payload_bytes = json.dumps(payload, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")

    sig_block = None
    if signer is not None:
        try:
            sig_block = signer.sign_payload(payload_bytes)
            logger.info("[%s] Log signed with RSA-PSS and ECDSA.", report.session_id)
        except Exception as e:
            logger.warning("Log signing failed: %s", e)

    # Write flat JSON log file
    log_path = os.path.join(output_dir, f"audit_{report.session_id}.json")
    full_record = {"payload": payload, "signatures": sig_block or {}}
    with open(log_path, "w") as f:
        json.dump(full_record, f, indent=2)

    # Append to hash chain
    if chain_store is not None:
        chain_store.append(payload, sig_block)

    logger.info("Audit log written: %s", log_path)
    return log_path, sig_block


###############################################################################
# SECTION 15 — TEMPLATE ALIGNMENT
###############################################################################

def align_to_template(image_bgr, template_bgr):
    orb = cv2.ORB_create(5000)
    kp1, d1 = orb.detectAndCompute(cv2.cvtColor(image_bgr,    cv2.COLOR_BGR2GRAY), None)
    kp2, d2 = orb.detectAndCompute(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(kp1) < 10:
        return image_bgr
    ms = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2),
                key=lambda m: m.distance)[:200]
    if len(ms) < 10:
        return image_bgr
    src = np.float32([kp1[m.queryIdx].pt for m in ms])
    dst = np.float32([kp2[m.trainIdx].pt for m in ms])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return image_bgr
    h, w = template_bgr.shape[:2]
    return cv2.warpPerspective(image_bgr, H, (w, h))


###############################################################################
# SECTION 16 — FORENSIC VERIFICATION ENGINE
###############################################################################

class ForensicVerificationEngine:
    """
    Orchestrates all verification layers:

      Layer 1   Descriptor matching       SuperPoint reciprocal cosine
      Layer 2   Geometric consistency     RANSAC homography + stability score
      Layer 3   Tamper localisation       Quadrant deviation heatmap
      Layer 4   Multi-scale robustness    Three-resolution consistency check
      Layer 5   Region verification       Per-zone ID document analysis
      Layer 6   Integrity fingerprint     SHA-256 image + descriptor hash
      Layer 7   Statistical fraud score   Weighted composite with calibration
      Layer 8   Anti-spoof analysis       Moire / photocopy / screen / print-scan
      Layer 9   Signed audit log          RSA-PSS + ECDSA signatures
      Layer 10  Hash-chained storage      Append-only tamper-evident audit chain
    """

    def __init__(self,
                 weights_path: str,
                 conf_thresh: float = 0.003,
                 nms_dist: int = 3,
                 border_remove: int = 4,
                 cuda: bool = False,
                 audit_log_dir: str = "./audit_logs",
                 chain_store_dir: Optional[str] = "./audit_store",
                 region_defs: dict = REGION_DEFINITIONS,
                 run_multiscale: bool = True,
                 run_region_verification: bool = True,
                 run_anti_spoof: bool = True,
                 base_size: Tuple[int, int] = (640, 480),
                 signer: Optional[LogSigner] = None,
                 calibrator: Optional[FraudScoreCalibrator] = None):

        self.scanner           = ImageScanner(weights_path, conf_thresh,
                                              nms_dist, border_remove, cuda)
        self.audit_log_dir     = audit_log_dir
        self.region_defs       = region_defs
        self.run_multiscale    = run_multiscale
        self.run_region_verify = run_region_verification
        self.run_anti_spoof    = run_anti_spoof
        self.base_size         = base_size
        self.signer            = signer
        self.calibrator        = calibrator
        self.anti_spoof_det    = AntiSpoofDetector() if run_anti_spoof else None
        self.chain_store       = (AuditChainStore(chain_store_dir)
                                  if chain_store_dir else None)

    # ------------------------------------------------------------------
    def verify(self,
               query_path: str,
               database_dir: str,
               match_threshold: float = 0.70,
               max_keypoints: int = 1000,
               visualize: bool = False,
               align_template: bool = False) -> ForensicReport:

        sid = str(uuid.uuid4())[:12]
        logger.info("[%s] Verification started: %s", sid, query_path)

        q_pts, q_desc = self.scanner.scan_image(query_path,
                                                  max_keypoints=max_keypoints)
        fp = compute_fingerprint(query_path, q_desc, sid)

        if q_desc.shape[1] == 0:
            report = self._empty(sid, query_path, fp)
            self._finalise(report)
            return report

        qd = q_desc.T / (np.linalg.norm(q_desc.T, axis=1, keepdims=True) + 1e-8)
        best_score, best_path, best_data = -1.0, None, None

        for fname in sorted(os.listdir(database_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            cp = os.path.join(database_dir, fname)
            try:
                db_pts, db_desc = self.scanner.scan_image(cp,
                                                           max_keypoints=max_keypoints)
            except Exception as e:
                logger.warning("Skipping %s: %s", fname, e); continue
            if db_desc.shape[1] == 0:
                continue

            dd  = db_desc.T / (np.linalg.norm(db_desc.T, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(qd, dd.T)
            fwd = np.argmax(sim, axis=1); bwd = np.argmax(sim, axis=0)
            matches = [(i, fwd[i], sim[i, fwd[i]])
                       for i in range(len(fwd)) if bwd[fwd[i]] == i]
            if matches:
                avg = float(np.mean([s for _, _, s in matches]))
                if avg > best_score:
                    best_score, best_path, best_data = avg, cp, (q_pts, db_pts, matches)

        if best_score < match_threshold or best_path is None:
            logger.info("[%s] No candidate meets threshold (best=%.3f).", sid, best_score)
            report = self._empty(sid, query_path, fp)
            self._finalise(report)
            self._print(report)
            return report

        qp, dbp, mb = best_data

        # Layer 2 — Geometric Consistency
        geo = compute_geometric_consistency(qp, dbp, mb)

        inlier_bool = None
        if geo.homography is not None and len(mb) >= 4:
            src_a = np.float32([[qp[0, i],  qp[1, i]]  for i, j, _ in mb])
            dst_a = np.float32([[dbp[0, j], dbp[1, j]] for i, j, _ in mb])
            _, mr = cv2.findHomography(src_a, dst_a, cv2.RANSAC, 4.0)
            if mr is not None:
                inlier_bool = mr.ravel().astype(bool)

        # Layer 3 — Tamper Localisation
        q_bgr  = cv2.imread(query_path)
        db_bgr = cv2.imread(best_path)
        shape  = q_bgr.shape if q_bgr is not None else (480, 640, 3)
        tamper = compute_tamper_localization(qp, dbp, mb, inlier_bool, shape)

        if align_template and q_bgr is not None and db_bgr is not None:
            q_bgr = align_to_template(q_bgr, db_bgr)

        # Layer 4 — Multi-Scale Verification
        ms = None
        if self.run_multiscale:
            try:
                ms = compute_multiscale_verification(
                    self.scanner, query_path, best_path, self.base_size
                )
            except Exception as e:
                logger.warning("[%s] Multi-scale failed: %s", sid, e)

        # Layer 5 — Region Verification
        regions = []
        if self.run_region_verify and q_bgr is not None and db_bgr is not None:
            regions = verify_regions(self.scanner, q_bgr, db_bgr, self.region_defs)

        # Layer 7 — Fraud Score (calibration-aware)
        auth, cal_fp, risk = compute_fraud_score(
            best_score, geo, tamper, ms, self.calibrator
        )

        # Layer 8 — Anti-Spoof
        anti_spoof = None
        if self.anti_spoof_det is not None:
            anti_spoof = self.anti_spoof_det.analyse(query_path)
            # Penalise fraud probability if strong spoof signals detected
            if anti_spoof.overall_spoof_probability > 0.35:
                cal_fp = float(np.clip(
                    cal_fp + anti_spoof.overall_spoof_probability * 0.25, 0, 1
                ))
                if cal_fp >= 0.60:
                    risk = RiskLevel.HIGH
                elif cal_fp >= 0.25:
                    risk = RiskLevel.MEDIUM

        report = ForensicReport(
            session_id=sid,
            query_path=query_path,
            best_match_path=best_path,
            risk_level=risk,
            fraud_probability=cal_fp,
            authenticity_score=auth,
            descriptor_similarity=best_score,
            geometric=geo,
            tamper=tamper,
            multiscale=ms,
            anti_spoof=anti_spoof,
            region_results=regions,
            fingerprint=fp,
        )

        self._finalise(report)

        if visualize:
            self._viz(query_path, best_path, qp, dbp, mb, tamper, report)

        self._print(report)
        return report

    # ------------------------------------------------------------------
    def _finalise(self, report: ForensicReport):
        """Write signed log and append to audit chain."""
        log_path, sig_block = write_audit_log(
            report,
            output_dir=self.audit_log_dir,
            signer=self.signer,
            chain_store=self.chain_store,
        )
        report.audit_log_path  = log_path
        report.audit_signature = sig_block

    # ------------------------------------------------------------------
    def _print(self, r: ForensicReport):
        sep = "-" * 68
        print(f"\n{sep}")
        print(f"  FORENSIC VERIFICATION REPORT    Session: {r.session_id}")
        print(sep)
        print(f"  Query              : {r.query_path}")
        print(f"  Best Match         : {r.best_match_path or 'None'}")
        print(f"  Risk Level         : {r.risk_level.value}")
        print(f"  Fraud Probability  : {r.fraud_probability * 100:.1f}%")
        print(f"  Authenticity Score : {r.authenticity_score * 100:.1f}%")
        print(f"  Descriptor Sim.    : {r.descriptor_similarity:.4f}")

        if r.geometric:
            g = r.geometric
            print(f"\n  Geometric Consistency")
            print(f"    Inliers            : {g.inlier_count}  ({g.inlier_ratio * 100:.1f}%)")
            print(f"    Reprojection Error : {g.reprojection_error:.2f} px")
            print(f"    H Stability Score  : {g.homography_stability:.4f}")

        if r.tamper:
            t = r.tamper
            print(f"\n  Tamper Analysis")
            print(f"    Unmatched Ratio    : {t.unmatched_ratio * 100:.1f}%")
            print(f"    Struct. Deviation  : {t.structural_deviation:.4f}")
            print(f"    Suspicious Zones   : {', '.join(t.suspicious_quadrants) or 'None'}")

        if r.multiscale:
            ms = r.multiscale
            print(f"\n  Multi-Scale Consistency")
            print(f"    Original           : {ms.original_score:.4f}")
            print(f"    Downscale          : {ms.downscale_score:.4f}")
            print(f"    Upscale            : {ms.upscale_score:.4f}")
            print(f"    Consistency Score  : {ms.consistency_score:.4f}  "
                  f"({'stable' if ms.consistency_score < 0.05 else 'unstable'})")

        if r.anti_spoof:
            a = r.anti_spoof
            print(f"\n  Anti-Spoof Analysis")
            print(f"    Moire Score        : {a.moire_score:.4f}")
            print(f"    Photocopy Score    : {a.photocopy_score:.4f}")
            print(f"    Screen Replay      : {a.screen_replay_score:.4f}")
            print(f"    Print-Scan Score   : {a.print_scan_score:.4f}")
            print(f"    Spoof Probability  : {a.overall_spoof_probability * 100:.1f}%")
            print(f"    Flags              : {', '.join(a.flags) or 'None'}")

        if r.region_results:
            print(f"\n  Region Verification")
            print(f"    {'Region':<18} {'Status':<12} {'Confidence':>10}  {'Inlier Ratio':>12}")
            print(f"    {'-'*56}")
            for rr in r.region_results:
                print(f"    {rr.name:<18} {rr.status:<12} "
                      f"{rr.confidence * 100:>9.1f}%  {rr.inlier_ratio * 100:>11.1f}%")

        if r.audit_signature:
            sig = r.audit_signature
            print(f"\n  Log Signatures")
            print(f"    RSA-PSS (4096)     : {sig.get('rsa_signature','')[:32]}...")
            print(f"    ECDSA (P-384)      : {sig.get('ec_signature','')[:32]}...")

        print(f"\n  Audit Log          : {r.audit_log_path}")
        print(f"  Image SHA-256      : {r.fingerprint.image_sha256[:32]}...")
        print(sep + "\n")

    # ------------------------------------------------------------------
    def _viz(self, qp, dbp, q_pts, db_pts, matches, tamper, report):
        qg  = cv2.imread(qp,  cv2.IMREAD_GRAYSCALE)
        dbg = cv2.imread(dbp, cv2.IMREAD_GRAYSCALE)
        if qg is None or dbg is None:
            return
        kp1 = [cv2.KeyPoint(float(q_pts[0, i]),  float(q_pts[1, i]),  1)
               for i in range(q_pts.shape[1])]
        kp2 = [cv2.KeyPoint(float(db_pts[0, j]), float(db_pts[1, j]), 1)
               for j in range(db_pts.shape[1])]
        dms = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1 - s)
               for i, j, s in matches]
        vis = cv2.drawMatches(qg, kp1, dbg, kp2, dms, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.putText(vis,
            f"Risk: {report.risk_level.value}  "
            f"Auth: {report.authenticity_score * 100:.1f}%  "
            f"Fraud: {report.fraud_probability * 100:.1f}%",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 0), 2)
        cv2.imshow("Forensic Match Visualisation", vis)
        if tamper.deviation_heatmap is not None:
            cv2.imshow("Tamper Deviation Heatmap", tamper.deviation_heatmap)
        cv2.waitKey(0); cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    def _empty(self, sid, qp, fp) -> ForensicReport:
        return ForensicReport(
            session_id=sid, query_path=qp, best_match_path=None,
            risk_level=RiskLevel.HIGH, fraud_probability=1.0,
            authenticity_score=0.0, descriptor_similarity=0.0,
            geometric=None, tamper=None, multiscale=None,
            anti_spoof=None, region_results=[], fingerprint=fp,
        )


###############################################################################
# ENTRY POINT
###############################################################################

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------ keys
    # Generate key pairs on first run; comment out after initial deployment.
    # key_paths = LogSigner.generate_keys("./keys")

    # Load signer (comment out if keys not yet generated)
    signer = None
    if CRYPTO_AVAILABLE and os.path.exists("./keys/rsa_private.pem"):
        with open("./keys/rsa_private.pem", "rb") as f: rsa_pem = f.read()
        with open("./keys/ec_private.pem",  "rb") as f: ec_pem  = f.read()
        signer = LogSigner(rsa_pem, ec_pem)

    # ------------------------------------------------------------------ calibrator
    # Load calibrator if available; otherwise scores use default thresholds.
    calibrator = None
    if os.path.exists("./calibrator.json"):
        calibrator = FraudScoreCalibrator.load("./calibrator.json")

    # ------------------------------------------------------------------ engine
    engine = ForensicVerificationEngine(
        weights_path="superpoint_v1.pth",
        conf_thresh=0.003,
        nms_dist=3,
        cuda=False,
        audit_log_dir="./audit_logs",
        chain_store_dir="./audit_store",
        run_multiscale=True,
        run_region_verification=True,
        run_anti_spoof=True,
        base_size=(640, 480),
        signer=signer,
        calibrator=calibrator,
    )

    report = engine.verify(
        query_path=r"C:\your_path\query_030.png",
        database_dir=r"C:\your_path\dataset\database",
        match_threshold=0.70,
        max_keypoints=1000,
        visualize=True,
        align_template=False,
    )

    # ------------------------------------------------------------------ chain verification
    if engine.chain_store is not None:
        valid, violations = engine.chain_store.verify_chain()
        print(f"Audit chain integrity: {'VALID' if valid else 'COMPROMISED'}")
        for v in violations:
            print(f"  VIOLATION: {v}")

    # ------------------------------------------------------------------ calibration example
    # Fit a calibrator from labelled validation data and save it:
    #
    #   cal = FraudScoreCalibrator()
    #   result = cal.fit(raw_scores=[...], labels=[0,1,0,1,...])
    #   cal.save("./calibrator.json")
    #   print(result)
