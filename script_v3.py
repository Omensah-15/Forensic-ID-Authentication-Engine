"""
SuperPoint Forensic Verification Engine  —  v3.0
=================================================
Production-grade document authenticity verification pipeline.

Verification Layers
-------------------
  Layer  1   Neural descriptor extraction     SuperPoint reciprocal cosine matching
  Layer  2   Geometric consistency            RANSAC homography + stability score
  Layer  3   Tamper localisation              Quadrant deviation heatmap
  Layer  4   Multi-scale robustness           Three-resolution consistency check
  Layer  5   Region verification              Per-zone document analysis
  Layer  6   Integrity fingerprint            SHA-256 image + descriptor hash
  Layer  7   Statistical fraud scoring        Weighted composite with calibration
  Layer  8   Anti-spoof analysis              Moire / photocopy / screen / print-scan
  Layer  9   Adaptive threshold management    Deployment-aware, self-updating thresholds
  Layer 10   Signed audit log                 RSA-PSS-4096 + ECDSA-P384 dual signatures
  Layer 11   Hash-chained audit storage       Append-only tamper-evident chain
  Layer 12   Template alignment               ORB-RANSAC homographic pre-alignment
  Layer 13   Database caching                 Descriptor cache with invalidation
  Layer 14   Batch verification               Concurrent multi-document processing
  Layer 15   Confidence calibration           Isotonic regression + ROC-optimal thresholds

Dependencies
------------
  Required : torch, opencv-python, numpy, scipy
  Optional : cryptography  (log signing)
             scikit-learn  (score calibration)

Usage
-----
  engine = ForensicVerificationEngine(weights_path="superpoint_v1.pth", ...)
  report = engine.verify(query_path="query.png", database_dir="./db")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn(
        "cryptography package not installed — audit log signing disabled. "
        "Install with: pip install cryptography",
        ImportWarning,
        stacklevel=2,
    )

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not installed — score calibration disabled. "
        "Install with: pip install scikit-learn",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("forensic_engine")


###############################################################################
# SECTION 1 — CONSTANTS
###############################################################################

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Default region definitions — normalised (x0, y0, x1, y1) fractions
# Override via ForensicVerificationEngine(region_defs=...) for custom document layouts.
DEFAULT_REGION_DEFINITIONS: Dict[str, Tuple[float, float, float, float]] = {
    "photo":         (0.00, 0.00, 0.35, 1.00),
    "hologram":      (0.55, 0.00, 0.75, 0.50),
    "serial_number": (0.35, 0.75, 1.00, 1.00),
    "logo":          (0.35, 0.00, 0.55, 0.35),
}


###############################################################################
# SECTION 2 — DATA STRUCTURES
###############################################################################

class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


@dataclass
class RegionResult:
    name:               str
    status:             str       # VERIFIED | SUSPICIOUS | FAILED
    confidence:         float
    inlier_ratio:       float
    matched_keypoints:  int
    avg_descriptor_sim: float = 0.0


@dataclass
class GeometricResult:
    inlier_count:         int
    inlier_ratio:         float
    reprojection_error:   float
    homography_stability: float
    condition_number:     float = 0.0          # singular value condition number
    homography:           Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TamperResult:
    unmatched_ratio:      float
    structural_deviation: float
    suspicious_quadrants: List[str]
    quadrant_scores:      Dict[str, float] = field(default_factory=dict)
    deviation_heatmap:    Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class MultiScaleResult:
    original_score:    float
    downscale_score:   float
    upscale_score:     float
    consistency_score: float
    scale_variance:    float = 0.0


@dataclass
class IntegrityFingerprint:
    image_sha256:    str
    descriptor_hash: str
    file_size_bytes: int
    timestamp:       float
    session_id:      str


@dataclass
class AntiSpoofResult:
    """Aggregated anti-spoof signals.  All scores in [0, 1]; higher = more suspicious."""
    moire_score:             float   # 0 = clean spectrum,       1 = strong Moire fringes
    photocopy_score:         float   # 0 = original contrast,    1 = flattened / copied
    screen_replay_score:     float   # 0 = physical capture,     1 = screen re-photograph
    print_scan_score:        float   # 0 = digital origin,       1 = print-scan artefact
    compression_score:       float   # 0 = lossless / minimal,   1 = heavy JPEG blocking
    overall_spoof_probability: float
    flags:                   List[str]


@dataclass
class DatabaseStats:
    total_documents:   int
    cached_documents:  int
    cache_hit_rate:    float
    index_build_time:  float


@dataclass
class ForensicReport:
    session_id:            str
    query_path:            str
    best_match_path:       Optional[str]
    risk_level:            RiskLevel
    fraud_probability:     float
    authenticity_score:    float
    descriptor_similarity: float
    geometric:             Optional[GeometricResult]
    tamper:                Optional[TamperResult]
    multiscale:            Optional[MultiScaleResult]
    anti_spoof:            Optional[AntiSpoofResult]
    region_results:        List[RegionResult]
    fingerprint:           IntegrityFingerprint
    db_stats:              Optional[DatabaseStats] = None
    processing_time_sec:   float = 0.0
    audit_log_path:        Optional[str] = None
    audit_signature:       Optional[Dict] = None


###############################################################################
# SECTION 3 — NEURAL NETWORK
###############################################################################

class SuperPointNet(torch.nn.Module):
    """
    SuperPoint self-supervised homographic adaptation network.
    Simultaneously produces keypoint heatmaps and dense 256-D descriptors.

    Architecture
    ------------
    Shared encoder  : 4 VGG-style convolutional blocks with max-pooling
    Detector head   : 2-layer convolution -> 65-channel softmax (64 cells + dustbin)
    Descriptor head : 2-layer convolution -> 256-D L2-normalised descriptors
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256

        # Shared encoder
        self.conv1a = torch.nn.Conv2d(1,  c1, 3, 1, 1)
        self.conv1b = torch.nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = torch.nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = torch.nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = torch.nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = torch.nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = torch.nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = torch.nn.Conv2d(c4, c4, 3, 1, 1)

        # Detector head
        self.convPa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = torch.nn.Conv2d(c5, 65,  1, 1, 0)

        # Descriptor head
        self.convDa = torch.nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = torch.nn.Conv2d(c5, d1,  1, 1, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector
        semi = self.convPb(self.relu(self.convPa(x)))

        # Descriptor — L2-normalise along channel axis
        desc = self.convDb(self.relu(self.convDa(x)))
        norm = torch.norm(desc, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        desc = desc / norm

        return semi, desc


###############################################################################
# SECTION 4 — DESCRIPTOR CACHE
###############################################################################

@dataclass
class CacheEntry:
    pts:       np.ndarray
    desc:      np.ndarray
    file_mtime: float
    file_size:  int


class DescriptorCache:
    """
    Thread-safe in-memory LRU descriptor cache with file-modification invalidation.
    Eliminates redundant neural forward passes for stable database images.

    Parameters
    ----------
    max_entries : int
        Maximum number of images to hold in memory simultaneously.
        Each entry occupies approximately max_keypoints * (3 + 256) * 8 bytes.
    """

    def __init__(self, max_entries: int = 512) -> None:
        self._store:   Dict[str, CacheEntry] = {}
        self._order:   List[str]             = []
        self._lock     = threading.Lock()
        self._max      = max_entries
        self._hits     = 0
        self._misses   = 0

    # ------------------------------------------------------------------
    def get(self, path: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(path)
            if entry is None:
                self._misses += 1
                return None
            try:
                stat = os.stat(path)
                if stat.st_mtime != entry.file_mtime or stat.st_size != entry.file_size:
                    self._evict(path)
                    self._misses += 1
                    return None
            except OSError:
                self._evict(path)
                self._misses += 1
                return None
            # Move to most-recently-used position
            self._order.remove(path)
            self._order.append(path)
            self._hits += 1
            return entry

    # ------------------------------------------------------------------
    def put(self, path: str, pts: np.ndarray, desc: np.ndarray) -> None:
        try:
            stat = os.stat(path)
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            return
        with self._lock:
            if path in self._store:
                self._order.remove(path)
            elif len(self._store) >= self._max:
                lru = self._order.pop(0)
                del self._store[lru]
            self._store[path] = CacheEntry(pts, desc, mtime, size)
            self._order.append(path)

    # ------------------------------------------------------------------
    def _evict(self, path: str) -> None:
        """Must be called with self._lock held."""
        self._store.pop(path, None)
        if path in self._order:
            self._order.remove(path)

    # ------------------------------------------------------------------
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)


###############################################################################
# SECTION 5 — IMAGE SCANNER
###############################################################################

class ImageScanner:
    """
    SuperPoint feature extractor.

    Improvements over original
    --------------------------
    - Vectorised non-maximum suppression via scipy maximum_filter (no Python loop)
    - Optional GPU half-precision (fp16) inference
    - Integrated descriptor cache with file-mtime invalidation
    - Configurable border removal and keypoint limit
    - Returns normalised descriptors ready for cosine similarity
    """

    def __init__(
        self,
        weights_path:  str,
        conf_thresh:   float = 0.005,
        nms_dist:      int   = 4,
        border_remove: int   = 4,
        cuda:          bool  = False,
        cache:         Optional[DescriptorCache] = None,
    ) -> None:
        self.conf_thresh   = conf_thresh
        self.nms_dist      = nms_dist
        self.cell          = 8
        self.border_remove = border_remove
        self.cuda          = cuda and torch.cuda.is_available()
        self.device        = torch.device("cuda" if self.cuda else "cpu")
        self._cache        = cache

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"SuperPoint weights not found: {weights_path}. "
                "Download from https://github.com/magicleap/SuperPointPretrainedNetwork"
            )
        self.net = SuperPointNet()
        state = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(state)
        self.net = self.net.to(self.device).eval()
        if self.cuda:
            self.net = self.net.half()
        logger.info("SuperPoint loaded on %s (fp%s)",
                    self.device, "16" if self.cuda else "32")

    # ------------------------------------------------------------------
    def load_image(self, path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load as float32 grayscale in [0, 1].  Raises IOError on failure."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {path}")
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    def scan_image(
        self,
        path:          str,
        target_size:   Optional[Tuple[int, int]] = None,
        max_keypoints: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoints and descriptors from a single image.

        Returns
        -------
        pts  : ndarray, shape (3, N) — [x, y, score] per keypoint
        desc : ndarray, shape (256, N) — L2-normalised descriptors
        """
        # Cache lookup (only for database images at default size)
        if self._cache is not None and target_size is None:
            cached = self._cache.get(path)
            if cached is not None:
                pts, desc = cached.pts, cached.desc
                if pts.shape[1] > max_keypoints:
                    idx  = np.argsort(-pts[2])[:max_keypoints]
                    pts, desc = pts[:, idx], desc[:, idx]
                return pts, desc

        img  = self.load_image(path, target_size)
        H, W = img.shape

        inp = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.cuda:
            inp = inp.half()

        with torch.no_grad():
            semi, coarse = self.net(inp)

        semi   = semi.cpu().float().numpy().squeeze()
        coarse = coarse.cpu().float().numpy().squeeze()

        # Softmax over 65 channels (64 cells + dustbin)
        dense  = np.exp(semi - semi.max())
        dense /= dense.sum(axis=0, keepdims=True) + 1e-8
        nodust = dense[:-1]

        Hc, Wc = semi.shape[1], semi.shape[2]
        heatmap = (
            nodust.transpose(1, 2, 0)
                  .reshape(Hc, Wc, 8, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(Hc * 8, Wc * 8)
        )

        # Non-maximum suppression via maximum filter
        mf = maximum_filter(heatmap, size=self.nms_dist * 2 + 1)
        ky, kx = np.where((heatmap == mf) & (heatmap >= self.conf_thresh))
        if not len(kx):
            return np.zeros((3, 0), dtype=np.float32), np.zeros((256, 0), dtype=np.float32)

        pts = np.array([kx, ky, heatmap[ky, kx]], dtype=np.float32)

        # Border removal
        b   = self.border_remove
        keep = (
            (pts[0] >= b) & (pts[0] < W - b) &
            (pts[1] >= b) & (pts[1] < H - b)
        )
        pts = pts[:, keep]
        if not pts.shape[1]:
            return pts, np.zeros((256, 0), dtype=np.float32)

        # Top-N by confidence
        if pts.shape[1] > max_keypoints:
            idx = np.argsort(-pts[2])[:max_keypoints]
            pts = pts[:, idx]

        # Descriptor interpolation via bilinear grid_sample
        samp = pts[:2].T.copy().astype(np.float32)
        samp[:, 0] = (samp[:, 0] / (W - 1)) * 2 - 1
        samp[:, 1] = (samp[:, 1] / (H - 1)) * 2 - 1

        st = torch.from_numpy(samp).to(self.device).unsqueeze(0).unsqueeze(0)
        ct = torch.from_numpy(coarse).to(self.device).unsqueeze(0)
        if self.cuda:
            st, ct = st.half(), ct.half()

        desc = F.grid_sample(ct, st, mode="bilinear",
                              padding_mode="zeros", align_corners=True)
        desc = desc.squeeze().cpu().float().numpy()
        if desc.ndim == 1:
            desc = desc.reshape(-1, 1)

        # L2-normalise
        norms = np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8
        desc  = desc / norms

        # Cache store
        if self._cache is not None and target_size is None:
            self._cache.put(path, pts.copy(), desc.copy())

        return pts, desc

    # ------------------------------------------------------------------
    def scan_array(
        self,
        bgr:           np.ndarray,
        max_keypoints: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scan a BGR ndarray directly (used for region crops)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = gray.shape

        inp = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.cuda:
            inp = inp.half()

        with torch.no_grad():
            semi, coarse = self.net(inp)

        semi   = semi.cpu().float().numpy().squeeze()
        coarse = coarse.cpu().float().numpy().squeeze()

        dense  = np.exp(semi - semi.max())
        dense /= dense.sum(axis=0, keepdims=True) + 1e-8
        nodust = dense[:-1]
        Hc, Wc = semi.shape[1], semi.shape[2]
        heatmap = (
            nodust.transpose(1, 2, 0)
                  .reshape(Hc, Wc, 8, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(Hc * 8, Wc * 8)
        )

        mf = maximum_filter(heatmap, size=self.nms_dist * 2 + 1)
        ky, kx = np.where((heatmap == mf) & (heatmap >= self.conf_thresh))
        if not len(kx):
            return np.zeros((3, 0), dtype=np.float32), np.zeros((256, 0), dtype=np.float32)

        pts = np.array([kx, ky, heatmap[ky, kx]], dtype=np.float32)
        b   = self.border_remove
        keep = (pts[0] >= b) & (pts[0] < W - b) & (pts[1] >= b) & (pts[1] < H - b)
        pts  = pts[:, keep]
        if not pts.shape[1]:
            return pts, np.zeros((256, 0), dtype=np.float32)

        if pts.shape[1] > max_keypoints:
            pts = pts[:, np.argsort(-pts[2])[:max_keypoints]]

        samp = pts[:2].T.copy().astype(np.float32)
        samp[:, 0] = (samp[:, 0] / max(W - 1, 1)) * 2 - 1
        samp[:, 1] = (samp[:, 1] / max(H - 1, 1)) * 2 - 1

        st = torch.from_numpy(samp).to(self.device).unsqueeze(0).unsqueeze(0)
        ct = torch.from_numpy(coarse).to(self.device).unsqueeze(0)
        if self.cuda:
            st, ct = st.half(), ct.half()

        desc = F.grid_sample(ct, st, mode="bilinear",
                              padding_mode="zeros", align_corners=True)
        desc = desc.squeeze().cpu().float().numpy()
        if desc.ndim == 1:
            desc = desc.reshape(-1, 1)

        norms = np.linalg.norm(desc, axis=0, keepdims=True) + 1e-8
        return pts, desc / norms


###############################################################################
# SECTION 6 — LOG SIGNING
###############################################################################

class LogSigner:
    """
    Dual-algorithm audit log signing: RSA-PSS-4096 and ECDSA-P384.

    Both algorithms sign the SHA-256 digest of the canonical JSON payload.
    Dual signing allows verification by relying parties that trust either
    algorithm, and provides cryptographic redundancy.

    Key management
    --------------
    Call LogSigner.generate_keys(output_dir) once at deployment time.
    Store private keys in a hardware security module or secrets manager.
    Distribute only the public key PEM files.
    """

    RSA_KEY_SIZE = 4096
    EC_CURVE     = ec.SECP384R1 if CRYPTO_AVAILABLE else None

    def __init__(self, rsa_private_pem: bytes, ec_private_pem: bytes) -> None:
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package is required for log signing.")
        self._rsa_key = serialization.load_pem_private_key(
            rsa_private_pem, password=None, backend=default_backend()
        )
        self._ec_key = serialization.load_pem_private_key(
            ec_private_pem, password=None, backend=default_backend()
        )

    # ------------------------------------------------------------------
    @staticmethod
    def generate_keys(output_dir: str = "./keys") -> Dict[str, str]:
        """
        Generate RSA-4096 and ECDSA-P384 key pairs and write PEM files.
        Returns a dict mapping filename -> absolute path.
        Call once during initial deployment.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package is required.")
        os.makedirs(output_dir, exist_ok=True)

        rsa_priv = rsa.generate_private_key(
            public_exponent=65537,
            key_size=LogSigner.RSA_KEY_SIZE,
            backend=default_backend(),
        )
        rsa_priv_pem = rsa_priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        rsa_pub_pem = rsa_priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        ec_priv = ec.generate_private_key(ec.SECP384R1(), default_backend())
        ec_priv_pem = ec_priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        ec_pub_pem = ec_priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        paths: Dict[str, str] = {}
        for fname, data in [
            ("rsa_private.pem", rsa_priv_pem),
            ("rsa_public.pem",  rsa_pub_pem),
            ("ec_private.pem",  ec_priv_pem),
            ("ec_public.pem",   ec_pub_pem),
        ]:
            p = os.path.join(output_dir, fname)
            with open(p, "wb") as f:
                f.write(data)
            paths[fname] = p

        logger.info("Key pairs written to %s", output_dir)
        return paths

    # ------------------------------------------------------------------
    def sign_payload(self, payload_bytes: bytes) -> Dict[str, str]:
        """Sign canonical payload bytes with both algorithms."""
        digest = hashlib.sha256(payload_bytes).digest()

        rsa_sig = self._rsa_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.Prehashed(hashes.SHA256()),
        )
        ec_sig = self._ec_key.sign(
            digest,
            ec.ECDSA(hashes.Prehashed(hashes.SHA256())),
        )

        return {
            "algorithm_rsa":  "RSA-PSS-SHA256-4096",
            "algorithm_ec":   "ECDSA-P384-SHA256",
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "rsa_signature":  rsa_sig.hex(),
            "ec_signature":   ec_sig.hex(),
            "signed_at_utc":  time.time(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def verify_signatures(
        payload_bytes:    bytes,
        signature_block:  Dict,
        rsa_public_pem:   bytes,
        ec_public_pem:    bytes,
    ) -> Dict[str, bool]:
        """Verify both signatures.  Returns {rsa_valid, ec_valid}."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package is required.")

        digest  = hashlib.sha256(payload_bytes).digest()
        rsa_pub = serialization.load_pem_public_key(rsa_public_pem, backend=default_backend())
        ec_pub  = serialization.load_pem_public_key(ec_public_pem,  backend=default_backend())

        results = {"rsa_valid": False, "ec_valid": False}

        try:
            rsa_pub.verify(
                bytes.fromhex(signature_block["rsa_signature"]),
                digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.Prehashed(hashes.SHA256()),
            )
            results["rsa_valid"] = True
        except Exception:
            pass

        try:
            ec_pub.verify(
                bytes.fromhex(signature_block["ec_signature"]),
                digest,
                ec.ECDSA(hashes.Prehashed(hashes.SHA256())),
            )
            results["ec_valid"] = True
        except Exception:
            pass

        return results


###############################################################################
# SECTION 7 — AUDIT CHAIN STORE
###############################################################################

class AuditChainStore:
    """
    Append-only, hash-chained audit log store.

    Storage layout
    --------------
    Chain  : <store_dir>/audit_chain.jsonl  (one JSON record per line)
    Index  : <store_dir>/chain_index.json   (session_id -> byte offset)

    Chain integrity
    ---------------
    Every record embeds the SHA-256 of the previous record's raw line byte
    string, forming a linked chain.  Tampering with any historical record
    invalidates all subsequent links, which verify_chain() detects.

    Thread safety
    -------------
    All append and verify operations are serialised by a reentrant lock.
    """

    CHAIN_FILENAME = "audit_chain.jsonl"
    INDEX_FILENAME = "chain_index.json"
    GENESIS_HASH   = "0" * 64

    def __init__(self, store_dir: str = "./audit_store") -> None:
        self.store_dir  = Path(store_dir)
        self.chain_path = self.store_dir / self.CHAIN_FILENAME
        self.index_path = self.store_dir / self.INDEX_FILENAME
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._load_index()

    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._index: Dict[str, int] = json.load(f)
        else:
            self._index = {}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    # ------------------------------------------------------------------
    def _last_hash(self) -> str:
        if not self.chain_path.exists() or self.chain_path.stat().st_size == 0:
            return self.GENESIS_HASH
        last_line = b""
        with open(self.chain_path, "rb") as f:
            for line in f:
                stripped = line.rstrip(b"\n")
                if stripped:
                    last_line = stripped
        return hashlib.sha256(last_line + b"\n").hexdigest() if last_line else self.GENESIS_HASH

    # ------------------------------------------------------------------
    def append(self, payload: dict, signature_block: Optional[dict] = None) -> str:
        """Append a signed entry.  Returns the SHA-256 hash of the written line."""
        with self._lock:
            entry_id    = payload.get("session_id", str(uuid.uuid4())[:12])
            prev_hash   = self._last_hash()
            entry_index = len(self._index)

            record = {
                "entry_index": entry_index,
                "prev_hash":   prev_hash,
                "payload":     payload,
                "signatures":  signature_block or {},
                "written_utc": time.time(),
            }
            line       = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            line_bytes = line.encode("utf-8")
            entry_hash = hashlib.sha256(line_bytes).hexdigest()

            byte_offset = self.chain_path.stat().st_size if self.chain_path.exists() else 0
            with open(self.chain_path, "a", encoding="utf-8") as f:
                f.write(line)

            self._index[entry_id] = byte_offset
            self._save_index()
            logger.info("Chain entry %d appended (hash prefix: %s)", entry_index, entry_hash[:16])
            return entry_hash

    # ------------------------------------------------------------------
    def retrieve(self, session_id: str) -> Optional[dict]:
        """O(1) retrieval via index.  Returns None if not found."""
        with self._lock:
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
        Returns (chain_valid, list_of_violation_messages).
        """
        violations: List[str] = []
        prev_hash = self.GENESIS_HASH
        lineno    = 0

        with self._lock:
            if not self.chain_path.exists():
                return True, []

            with open(self.chain_path, "rb") as f:
                for lineno, raw_line in enumerate(f, start=1):
                    raw_stripped = raw_line.rstrip(b"\n")
                    if not raw_stripped:
                        continue
                    try:
                        record = json.loads(raw_stripped)
                    except json.JSONDecodeError as exc:
                        violations.append(f"Line {lineno}: JSON parse error — {exc}")
                        continue

                    declared = record.get("prev_hash", "")
                    if lineno > 1 and declared != prev_hash:
                        violations.append(
                            f"Line {lineno}: chain break — "
                            f"declared {declared[:16]}... != computed {prev_hash[:16]}..."
                        )

                    prev_hash = hashlib.sha256(raw_stripped + b"\n").hexdigest()

        valid = not violations
        if valid:
            logger.info("Chain verification passed (%d entries).", lineno)
        else:
            logger.warning("Chain verification: %d violation(s) found.", len(violations))
        return valid, violations


###############################################################################
# SECTION 8 — ADAPTIVE THRESHOLD MANAGER
###############################################################################

class AdaptiveThresholdManager:
    """
    Maintains deployment-specific, self-updating detection thresholds.

    Rationale
    ---------
    Fixed thresholds are calibrated on lab data and drift as real-world
    document quality, camera equipment, and fraud patterns evolve.
    This class tracks rolling statistics (mean and variance) of observed
    scores for each signal and adjusts thresholds using an exponential
    moving average, keeping false-positive and false-negative rates stable.

    Parameters
    ----------
    ema_alpha : float
        Smoothing factor for exponential moving average (0 < alpha < 1).
        Lower values make thresholds more stable; higher values more reactive.
    persistence_path : str or None
        JSON file path for saving/loading threshold state across restarts.
    """

    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "moire":        0.18,
        "photocopy":    0.55,
        "screen_replay":0.40,
        "print_scan":   0.35,
        "compression":  0.45,
        "match":        0.70,
        "region_verify":0.75,
        "region_suspicious": 0.45,
        "inlier_ratio": 0.30,
        "reprojection": 8.0,
    }

    def __init__(
        self,
        ema_alpha:        float = 0.05,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._alpha       = ema_alpha
        self._path        = persistence_path
        self._lock        = threading.Lock()
        self._thresholds  = dict(self.DEFAULT_THRESHOLDS)
        self._ema_scores: Dict[str, float] = {}
        self._ema_var:    Dict[str, float] = {}
        self._counts:     Dict[str, int]   = {}

        if persistence_path and os.path.isfile(persistence_path):
            self._load()

    # ------------------------------------------------------------------
    def get(self, key: str) -> float:
        with self._lock:
            return self._thresholds.get(key, self.DEFAULT_THRESHOLDS.get(key, 0.5))

    # ------------------------------------------------------------------
    def update(self, key: str, observed_score: float) -> None:
        """
        Update the EMA for a given signal and nudge the threshold if
        the score distribution has drifted by more than one standard deviation.
        """
        with self._lock:
            alpha = self._alpha
            if key not in self._ema_scores:
                self._ema_scores[key] = observed_score
                self._ema_var[key]    = 0.0
                self._counts[key]     = 1
                return

            prev_mean = self._ema_scores[key]
            new_mean  = alpha * observed_score + (1 - alpha) * prev_mean
            new_var   = alpha * (observed_score - new_mean) ** 2 + (1 - alpha) * self._ema_var[key]

            self._ema_scores[key] = new_mean
            self._ema_var[key]    = new_var
            self._counts[key]     = self._counts.get(key, 0) + 1

            # Only adjust after sufficient observations
            if self._counts[key] < 50:
                return

            std = math.sqrt(new_var) if new_var > 0 else 0.0
            default = self.DEFAULT_THRESHOLDS.get(key, 0.5)
            drift   = abs(new_mean - default)

            if drift > std and std > 0:
                # Nudge threshold toward observed mean at a conservative rate
                current = self._thresholds.get(key, default)
                self._thresholds[key] = current + alpha * (new_mean - current)
                logger.debug(
                    "Threshold '%s' updated: %.4f -> %.4f (EMA mean=%.4f std=%.4f)",
                    key, current, self._thresholds[key], new_mean, std,
                )

            if self._path and self._counts[key] % 100 == 0:
                self._save()

    # ------------------------------------------------------------------
    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._thresholds  = dict(self.DEFAULT_THRESHOLDS)
                self._ema_scores  = {}
                self._ema_var     = {}
                self._counts      = {}
            else:
                self._thresholds[key] = self.DEFAULT_THRESHOLDS.get(key, 0.5)
                self._ema_scores.pop(key, None)
                self._ema_var.pop(key, None)
                self._counts.pop(key, None)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        state = {
            "thresholds": self._thresholds,
            "ema_scores": self._ema_scores,
            "ema_var":    self._ema_var,
            "counts":     self._counts,
        }
        with open(self._path, "w") as f:
            json.dump(state, f, indent=2)

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                state = json.load(f)
            self._thresholds  = {**self.DEFAULT_THRESHOLDS, **state.get("thresholds", {})}
            self._ema_scores  = state.get("ema_scores", {})
            self._ema_var     = state.get("ema_var", {})
            self._counts      = state.get("counts", {})
            logger.info("Adaptive thresholds loaded from %s", self._path)
        except Exception as exc:
            logger.warning("Could not load threshold state: %s", exc)


###############################################################################
# SECTION 9 — ANTI-SPOOF DETECTOR
###############################################################################

class AntiSpoofDetector:
    """
    Signal-processing anti-spoof detector — no separate neural network required.

    Detects five attack classes via spatial frequency and statistical analysis:

    Moire detection
        Periodic interference fringes from scanning a printed halftone
        appear as sharp off-axis peaks in the DFT magnitude spectrum.
        Measured as the ratio of peak-to-mean energy across 8 radial sectors.

    Photocopy detection
        Photocopies exhibit histogram flattening, reduced dynamic range in
        shadows/highlights, and elevated Laplacian noise in flat regions.
        Combined from IQR spread, clipping fraction, and normalised noise.

    Screen replay detection
        Recapturing a document displayed on a screen introduces horizontal
        scan-line frequency peaks and high-frequency blur from re-capture
        optics.  Both signals contribute to the screen replay score.

    Print-scan artefact detection
        Print-scan workflows introduce dot-gain and JPEG/halftone rosettes.
        8x8 DCT block analysis detects characteristic coefficient distributions.

    Compression artefact detection (new in v3)
        Heavy JPEG re-compression introduces 8x8 blocking at edges.
        Measured via average gradient discontinuity at 8-pixel boundaries
        relative to within-block gradient magnitude.
    """

    def __init__(self, threshold_manager: Optional[AdaptiveThresholdManager] = None) -> None:
        self._thresh = threshold_manager or AdaptiveThresholdManager()

    # ------------------------------------------------------------------
    def analyse(self, image_path: str) -> AntiSpoofResult:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return AntiSpoofResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ["IMAGE_LOAD_FAILED"])

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        moire   = self._moire_score(gray)
        photo   = self._photocopy_score(gray)
        screen  = self._screen_replay_score(gray)
        pscan   = self._print_scan_score(gray)
        compr   = self._compression_score(gray)

        # Update adaptive thresholds with observed scores
        for key, val in [("moire", moire), ("photocopy", photo),
                          ("screen_replay", screen), ("print_scan", pscan),
                          ("compression", compr)]:
            self._thresh.update(key, val)

        # Weighted combination — screen replay and Moire carry highest weight
        overall = float(np.clip(
            0.28 * moire + 0.22 * photo + 0.28 * screen + 0.12 * pscan + 0.10 * compr,
            0.0, 1.0,
        ))

        flags: List[str] = []
        if moire  >= self._thresh.get("moire"):          flags.append("MOIRE_DETECTED")
        if photo  >= self._thresh.get("photocopy"):      flags.append("PHOTOCOPY_SUSPECTED")
        if screen >= self._thresh.get("screen_replay"):  flags.append("SCREEN_REPLAY_SUSPECTED")
        if pscan  >= self._thresh.get("print_scan"):     flags.append("PRINT_SCAN_ARTEFACT")
        if compr  >= self._thresh.get("compression"):    flags.append("COMPRESSION_ARTEFACT")

        return AntiSpoofResult(moire, photo, screen, pscan, compr, overall, flags)

    # ------------------------------------------------------------------
    @staticmethod
    def _dft_magnitude(gray: np.ndarray) -> np.ndarray:
        f   = np.fft.fft2(gray.astype(np.float32))
        fs  = np.fft.fftshift(f)
        return np.log1p(np.abs(fs))

    # ------------------------------------------------------------------
    def _moire_score(self, gray: np.ndarray) -> float:
        mag   = self._dft_magnitude(gray)
        H, W  = mag.shape
        cy, cx = H // 2, W // 2

        y_idx, x_idx = np.mgrid[0:H, 0:W]
        dist  = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        r_max = min(cy, cx)

        dc_mask       = dist < (r_max * 0.05)
        outer_mask    = dist > (r_max * 0.85)
        analysis_mask = ~dc_mask & ~outer_mask

        sector_peaks: List[float] = []
        for s in range(8):
            a0 = math.radians(s * 45 - 22.5)
            a1 = math.radians(s * 45 + 22.5)
            angle_map   = np.arctan2(y_idx - cy, x_idx - cx)
            sector_mask = analysis_mask & (angle_map >= a0) & (angle_map < a1)
            if sector_mask.sum() > 0:
                sv = mag[sector_mask]
                sector_peaks.append(float(sv.max() / (sv.mean() + 1e-8)))

        avg_peak = float(np.mean(sector_peaks)) if sector_peaks else 0.0
        return float(np.clip((avg_peak - 3.0) / 10.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _photocopy_score(self, gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        cdf  = np.cumsum(hist)

        q25 = int(np.searchsorted(cdf, 0.25))
        q75 = int(np.searchsorted(cdf, 0.75))
        iqr_norm = (q75 - q25) / 255.0

        shadow_clip    = float(hist[:10].sum())
        highlight_clip = float(hist[245:].sum())
        clip_score     = float(np.clip((shadow_clip + highlight_clip) / 0.15, 0.0, 1.0))

        lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_int   = float(gray.mean()) + 1e-8
        noise_score = float(np.clip(lap_var / (mean_int * 50.0), 0.0, 1.0))

        return float(np.clip(
            (1.0 - iqr_norm) * 0.40 + clip_score * 0.35 + noise_score * 0.25,
            0.0, 1.0,
        ))

    # ------------------------------------------------------------------
    def _screen_replay_score(self, gray: np.ndarray) -> float:
        row_spectrum = np.abs(np.fft.rfft(gray.astype(np.float32), axis=1))
        row_mean     = row_spectrum.mean(axis=0)
        N            = len(row_mean)
        low_energy   = row_mean[:N // 5].sum() + 1e-8
        high_energy  = row_mean[N // 5:].sum()
        hf_ratio     = float(high_energy / low_energy)

        scanline_score = float(np.clip(1.0 - hf_ratio / 2.0, 0.0, 1.0))
        lap_var        = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score     = float(np.clip(1.0 - lap_var / 800.0, 0.0, 1.0))

        return float(np.clip(scanline_score * 0.60 + blur_score * 0.40, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _print_scan_score(self, gray: np.ndarray) -> float:
        H, W = gray.shape
        pH   = (H // 8) * 8
        pW   = (W // 8) * 8
        if pH == 0 or pW == 0:
            return 0.0

        cropped     = gray[:pH, :pW].astype(np.float32) - 128.0
        dc_ratios: List[float] = []

        for y in range(0, pH, 8):
            for x in range(0, pW, 8):
                block        = cropped[y:y + 8, x:x + 8]
                dct_blk      = cv2.dct(block)
                dc_energy    = float(dct_blk[0, 0] ** 2)
                total_energy = float((dct_blk ** 2).sum()) + 1e-8
                dc_ratios.append(dc_energy / total_energy)

        mean_dc = float(np.mean(dc_ratios)) if dc_ratios else 0.5
        return float(np.clip((mean_dc - 0.45) / 0.35, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _compression_score(self, gray: np.ndarray) -> float:
        """
        Detect heavy JPEG re-compression artefacts via 8-pixel boundary
        gradient discontinuities.  Genuine captures have smooth gradients
        at all locations; repeatedly re-compressed images show elevated
        gradient jumps specifically at 8-pixel block boundaries.
        """
        H, W = gray.shape
        g    = gray.astype(np.float32)

        # Horizontal boundary gradient (column boundaries at x = 8, 16, 24, ...)
        if W > 8:
            boundary_x  = np.arange(8, W - 1, 8)
            bound_grad_x = np.abs(g[:, boundary_x] - g[:, boundary_x - 1]).mean()
            all_grad_x   = np.abs(np.diff(g, axis=1)).mean() + 1e-8
            ratio_x      = float(bound_grad_x / all_grad_x)
        else:
            ratio_x = 1.0

        # Vertical boundary gradient (row boundaries at y = 8, 16, 24, ...)
        if H > 8:
            boundary_y   = np.arange(8, H - 1, 8)
            bound_grad_y = np.abs(g[boundary_y, :] - g[boundary_y - 1, :]).mean()
            all_grad_y   = np.abs(np.diff(g, axis=0)).mean() + 1e-8
            ratio_y      = float(bound_grad_y / all_grad_y)
        else:
            ratio_y = 1.0

        # Ratios significantly above 1 indicate block-boundary discontinuities
        avg_ratio = (ratio_x + ratio_y) / 2.0
        return float(np.clip((avg_ratio - 1.05) / 0.80, 0.0, 1.0))


###############################################################################
# SECTION 10 — CONFIDENCE CALIBRATION
###############################################################################

class FraudScoreCalibrator:
    """
    Calibrates raw fraud probability scores to statistically valid probabilities.

    Workflow
    --------
    1. Collect ground-truth labels (0 = genuine, 1 = fraud) and raw
       fraud_probability scores on a held-out validation set.
    2. Call calibrator.fit(raw_scores, labels).
    3. Calibrator computes:
         - ROC AUC
         - Optimal threshold via Youden J statistic (maximises TPR - FPR)
         - High-precision threshold (FPR <= 5%)
         - Isotonic regression mapping for monotone score calibration
    4. All subsequent calls to calibrator.calibrate(score) return a
       calibrated probability and a statistically justified RiskLevel.

    Persistence
    -----------
    calibrator.save(path) / FraudScoreCalibrator.load(path)
    """

    DEFAULT_LOW_THRESH    = 0.25
    DEFAULT_MEDIUM_THRESH = 0.60

    def __init__(self) -> None:
        self.is_fitted      = False
        self.roc_auc:       Optional[float]      = None
        self.optimal_thresh = self.DEFAULT_LOW_THRESH
        self.high_thresh    = self.DEFAULT_MEDIUM_THRESH
        self._iso_x:        Optional[np.ndarray] = None
        self._iso_y:        Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, raw_scores: List[float], labels: List[int]) -> Dict:
        """
        Fit calibration from validation data.

        Parameters
        ----------
        raw_scores : list[float]  — raw fraud_probability values from the engine
        labels     : list[int]    — 0 = genuine, 1 = fraud

        Returns
        -------
        dict containing roc_auc, optimal_threshold, high_precision_threshold,
        n_samples, and prevalence.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for calibration.")

        scores = np.array(raw_scores, dtype=float)
        labs   = np.array(labels,     dtype=int)

        if len(np.unique(labs)) < 2:
            raise ValueError("Calibration requires both positive and negative examples.")
        if len(scores) < 20:
            raise ValueError("At least 20 samples are required for reliable calibration.")

        fpr, tpr, thresholds = roc_curve(labs, scores)
        self.roc_auc         = float(roc_auc_score(labs, scores))

        j_scores         = tpr - fpr
        best_idx         = int(np.argmax(j_scores))
        self.optimal_thresh = float(thresholds[best_idx])

        high_prec_mask = fpr <= 0.05
        self.high_thresh = (
            float(thresholds[high_prec_mask][-1])
            if high_prec_mask.any()
            else float(thresholds[int(np.argmin(fpr))])
        )

        iso    = IsotonicRegression(out_of_bounds="clip")
        iso.fit(scores, labs)
        probe  = np.linspace(0.0, 1.0, 1000)
        self._iso_x    = probe
        self._iso_y    = iso.predict(probe)
        self.is_fitted = True

        result = {
            "roc_auc":                    round(self.roc_auc, 4),
            "optimal_threshold":          round(self.optimal_thresh, 4),
            "high_precision_threshold":   round(self.high_thresh, 4),
            "n_samples":                  len(labs),
            "prevalence":                 round(float(labs.mean()), 4),
        }
        logger.info(
            "Calibration complete — AUC: %.4f  opt_thresh: %.4f  hi_thresh: %.4f",
            self.roc_auc, self.optimal_thresh, self.high_thresh,
        )
        return result

    # ------------------------------------------------------------------
    def calibrate(self, raw_score: float) -> Tuple[float, RiskLevel]:
        """Map a raw fraud_probability to a calibrated value and RiskLevel."""
        if self.is_fitted and self._iso_x is not None:
            cal = float(np.interp(raw_score, self._iso_x, self._iso_y))
        else:
            cal = float(raw_score)

        low_t = self.optimal_thresh if self.is_fitted else self.DEFAULT_LOW_THRESH
        hi_t  = self.high_thresh    if self.is_fitted else self.DEFAULT_MEDIUM_THRESH

        risk = (RiskLevel.LOW if cal < low_t else
                RiskLevel.HIGH if cal >= hi_t else
                RiskLevel.MEDIUM)

        return cal, risk

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        state = {
            "is_fitted":       self.is_fitted,
            "roc_auc":         self.roc_auc,
            "optimal_thresh":  self.optimal_thresh,
            "high_thresh":     self.high_thresh,
            "iso_x":           self._iso_x.tolist() if self._iso_x is not None else None,
            "iso_y":           self._iso_y.tolist() if self._iso_y is not None else None,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Calibrator saved to %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "FraudScoreCalibrator":
        with open(path) as f:
            state = json.load(f)
        c = cls()
        c.is_fitted      = state.get("is_fitted", False)
        c.roc_auc        = state.get("roc_auc")
        c.optimal_thresh = state.get("optimal_thresh", cls.DEFAULT_LOW_THRESH)
        c.high_thresh    = state.get("high_thresh",    cls.DEFAULT_MEDIUM_THRESH)
        c._iso_x = np.array(state["iso_x"]) if state.get("iso_x") else None
        c._iso_y = np.array(state["iso_y"]) if state.get("iso_y") else None
        logger.info("Calibrator loaded from %s (fitted=%s)", path, c.is_fitted)
        return c


###############################################################################
# SECTION 11 — GEOMETRIC CONSISTENCY
###############################################################################

def compute_geometric_consistency(
    q_pts:   np.ndarray,
    db_pts:  np.ndarray,
    matches: List[Tuple[int, int, float]],
) -> GeometricResult:
    """
    Estimate and evaluate a RANSAC homography between matched keypoints.

    Returns GeometricResult with inlier statistics, reprojection error,
    homography stability (via singular value decomposition), and the
    condition number of the normalised homography matrix.
    """
    if len(matches) < 4:
        return GeometricResult(0, 0.0, float("inf"), 0.0, float("inf"))

    src = np.float32([[q_pts[0, i],  q_pts[1, i]]  for i, j, _ in matches])
    dst = np.float32([[db_pts[0, j], db_pts[1, j]] for i, j, _ in matches])

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=2000, confidence=0.999)
    if H is None or mask is None:
        return GeometricResult(0, 0.0, float("inf"), 0.0, float("inf"))

    inlier_mask  = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(len(matches), 1)

    if inlier_count >= 4:
        si   = src[inlier_mask]
        di   = dst[inlier_mask]
        sh   = np.hstack([si, np.ones((len(si), 1))])
        pr   = (H @ sh.T).T
        pr  /= pr[:, 2:3] + 1e-8
        reproj_err = float(np.linalg.norm(pr[:, :2] - di, axis=1).mean())
    else:
        reproj_err = float("inf")

    try:
        Hn  = H / (np.linalg.norm(H) + 1e-8)
        sv  = np.linalg.svd(Hn, compute_uv=False)
        stb = float(np.clip(sv[-1] / (sv[0] + 1e-8) * 10.0, 0.0, 1.0))
        cond = float(sv[0] / (sv[-1] + 1e-8))
    except Exception:
        stb, cond = 0.0, float("inf")

    return GeometricResult(inlier_count, inlier_ratio, reproj_err, stb, cond, H)


###############################################################################
# SECTION 12 — TAMPER LOCALISATION
###############################################################################

def compute_tamper_localization(
    q_pts:           np.ndarray,
    db_pts:          np.ndarray,
    matches:         List[Tuple[int, int, float]],
    inlier_mask_bool: Optional[np.ndarray],
    image_shape:     Tuple[int, ...],
) -> TamperResult:
    """
    Build a deviation heatmap and identify suspicious image quadrants.

    Outlier keypoints (those that failed geometric consistency) are plotted
    onto a heatmap weighted by their descriptor dissimilarity.  Gaussian
    smoothing produces a continuous saliency map.  Each image quadrant is
    scored independently, providing approximate localisation of tampered zones.
    """
    H_i, W_i = image_shape[:2]
    heatmap   = np.zeros((H_i, W_i), dtype=np.float32)

    if inlier_mask_bool is None or len(inlier_mask_bool) != len(matches):
        inlier_mask_bool = np.ones(len(matches), dtype=bool)

    outlier_idx     = np.where(~inlier_mask_bool)[0]
    unmatched_ratio = max((q_pts.shape[1] - len(matches)) / max(q_pts.shape[1], 1), 0.0)
    deviations:     List[float] = []

    for idx in outlier_idx:
        i, j, s = matches[idx]
        x   = int(np.clip(q_pts[0, i], 0, W_i - 1))
        y   = int(np.clip(q_pts[1, i], 0, H_i - 1))
        dev = float(1.0 - s)
        heatmap[y, x] += dev
        deviations.append(dev)

    struct_dev = float(np.mean(deviations)) if deviations else 0.0

    if heatmap.max() > 0:
        heatmap     = cv2.GaussianBlur(heatmap, (21, 21), 0)
        hm8         = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_vis = cv2.applyColorMap(hm8, cv2.COLORMAP_HOT)
    else:
        heatmap_vis = np.zeros((H_i, W_i, 3), dtype=np.uint8)

    mid_y, mid_x = H_i // 2, W_i // 2
    quad_defs = {
        "top-left":     (slice(None, mid_y), slice(None, mid_x)),
        "top-right":    (slice(None, mid_y), slice(mid_x, None)),
        "bottom-left":  (slice(mid_y, None), slice(None, mid_x)),
        "bottom-right": (slice(mid_y, None), slice(mid_x, None)),
    }

    quadrant_scores: Dict[str, float] = {}
    threshold = float(heatmap.mean() + heatmap.std()) if heatmap.max() > 0 else 30.0
    suspicious: List[str] = []

    for name, (sy, sx) in quad_defs.items():
        score = float(heatmap[sy, sx].mean())
        quadrant_scores[name] = round(score, 4)
        if score > threshold:
            suspicious.append(name)

    return TamperResult(unmatched_ratio, struct_dev, suspicious, quadrant_scores, heatmap_vis)


###############################################################################
# SECTION 13 — MULTI-SCALE VERIFICATION
###############################################################################

def compute_multiscale_verification(
    scanner:   ImageScanner,
    q_path:    str,
    db_path:   str,
    base_size: Tuple[int, int] = (640, 480),
) -> MultiScaleResult:
    """
    Verify descriptor consistency across three image scales.

    A genuine document produces stable matching scores regardless of
    resolution.  AI-generated or resolution-specific fakes often degrade
    at scales other than the one used for generation.
    """
    def score_at(sz: Optional[Tuple[int, int]]) -> float:
        p1, d1 = scanner.scan_image(q_path,  target_size=sz)
        p2, d2 = scanner.scan_image(db_path, target_size=sz)
        if d1.shape[1] == 0 or d2.shape[1] == 0:
            return 0.0
        q  = d1.T / (np.linalg.norm(d1.T, axis=1, keepdims=True) + 1e-8)
        d  = d2.T / (np.linalg.norm(d2.T, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(q, d.T)
        fwd = np.argmax(sim, axis=1)
        bwd = np.argmax(sim, axis=0)
        recip = [(i, fwd[i], sim[i, fwd[i]]) for i in range(len(fwd)) if bwd[fwd[i]] == i]
        return float(np.mean([s for _, _, s in recip])) if recip else 0.0

    W, H = base_size
    orig = score_at(base_size)
    down = score_at((max(W // 2, 64), max(H // 2, 48)))
    up   = score_at((W * 2, H * 2))

    scores    = [orig, down, up]
    variance  = float(np.var(scores))
    return MultiScaleResult(orig, down, up, float(np.std(scores)), variance)


###############################################################################
# SECTION 14 — REGION VERIFICATION
###############################################################################

def verify_regions(
    scanner:     ImageScanner,
    q_bgr:       np.ndarray,
    db_bgr:      np.ndarray,
    region_defs: Dict[str, Tuple[float, float, float, float]],
    thresh_manager: Optional[AdaptiveThresholdManager] = None,
) -> List[RegionResult]:
    """
    Independently verify each named region of the document.

    Each region is cropped from both the query and database images, scanned
    for keypoints and descriptors, matched with mutual nearest-neighbour,
    and geometrically verified with a local RANSAC homography.  The region
    confidence combines descriptor similarity and geometric inlier ratio.
    """
    results: List[RegionResult] = []
    H,  W   = q_bgr.shape[:2]
    dH, dW  = db_bgr.shape[:2]
    tm       = thresh_manager or AdaptiveThresholdManager()
    verify_t = tm.get("region_verify")
    suspect_t = tm.get("region_suspicious")

    for name, (nx0, ny0, nx1, ny1) in region_defs.items():
        qc = q_bgr [int(ny0 * H):int(ny1 * H),   int(nx0 * W):int(nx1 * W)]
        dc = db_bgr[int(ny0 * dH):int(ny1 * dH), int(nx0 * dW):int(nx1 * dW)]

        if qc.size == 0 or dc.size == 0:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0, 0.0))
            continue

        qp, qd = scanner.scan_array(qc)
        dp, dd = scanner.scan_array(dc)

        if qd.shape[1] == 0 or dd.shape[1] == 0:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0, 0.0))
            continue

        qn  = qd.T / (np.linalg.norm(qd.T, axis=1, keepdims=True) + 1e-8)
        dn  = dd.T / (np.linalg.norm(dd.T, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(qn, dn.T)
        fwd = np.argmax(sim, axis=1)
        bwd = np.argmax(sim, axis=0)
        recip = [(i, fwd[i], sim[i, fwd[i]]) for i in range(len(fwd)) if bwd[fwd[i]] == i]

        if not recip:
            results.append(RegionResult(name, "FAILED", 0.0, 0.0, 0, 0.0))
            continue

        avg_sim = float(np.mean([s for _, _, s in recip]))
        geo     = compute_geometric_consistency(qp, dp, recip)
        conf    = avg_sim * 0.60 + geo.inlier_ratio * 0.40
        status  = ("VERIFIED"   if conf >= verify_t  else
                   "SUSPICIOUS" if conf >= suspect_t else
                   "FAILED")

        results.append(RegionResult(name, status, conf, geo.inlier_ratio, len(recip), avg_sim))

    return results


###############################################################################
# SECTION 15 — INTEGRITY FINGERPRINT
###############################################################################

def compute_fingerprint(image_path: str, desc: np.ndarray, session_id: str) -> IntegrityFingerprint:
    """Compute SHA-256 of the raw image bytes and a hash of the descriptor matrix."""
    with open(image_path, "rb") as f:
        raw = f.read()
    file_size  = len(raw)
    img_hash   = hashlib.sha256(raw).hexdigest()
    desc_hash  = hashlib.sha256(desc.tobytes()).hexdigest()
    return IntegrityFingerprint(img_hash, desc_hash, file_size, time.time(), session_id)


###############################################################################
# SECTION 16 — FRAUD SCORING
###############################################################################

def compute_fraud_score(
    desc_sim:   float,
    geo:        Optional[GeometricResult],
    tamper:     Optional[TamperResult],
    multiscale: Optional[MultiScaleResult],
    anti_spoof: Optional[AntiSpoofResult],
    calibrator: Optional[FraudScoreCalibrator] = None,
) -> Tuple[float, float, RiskLevel]:
    """
    Compute a weighted composite authenticity score and calibrated fraud probability.

    Signal weights
    --------------
    Descriptor similarity    30%  — core neural matching quality
    Geometric inlier ratio   22%  — spatial consistency of matches
    Reprojection error       13%  — sub-pixel geometric accuracy
    Tamper deviation         15%  — proportion of unmatched / outlier keypoints
    Multi-scale consistency  10%  — resolution-invariance test
    Anti-spoof penalty       10%  — physical capture authenticity

    Returns
    -------
    (authenticity_score, calibrated_fraud_probability, risk_level)
    """
    s_desc   = float(np.clip(desc_sim, 0.0, 1.0))
    s_inlier = float(np.clip(geo.inlier_ratio, 0.0, 1.0))               if geo else 0.0
    s_reproj = (
        float(np.clip(1.0 - (geo.reprojection_error / 20.0), 0.0, 1.0))
        if geo and geo.reprojection_error < float("inf") else 0.0
    )
    s_tamper = (
        float(np.clip(1.0 - tamper.unmatched_ratio - tamper.structural_deviation, 0.0, 1.0))
        if tamper else 0.5
    )
    s_scale  = (
        float(np.clip(1.0 - multiscale.consistency_score * 5.0, 0.0, 1.0))
        if multiscale else 0.5
    )
    s_spoof  = (
        float(np.clip(1.0 - anti_spoof.overall_spoof_probability, 0.0, 1.0))
        if anti_spoof else 1.0
    )

    auth   = (
        0.30 * s_desc +
        0.22 * s_inlier +
        0.13 * s_reproj +
        0.15 * s_tamper +
        0.10 * s_scale  +
        0.10 * s_spoof
    )
    raw_fp = float(np.clip(1.0 - auth, 0.0, 1.0))

    if calibrator is not None:
        cal_fp, risk = calibrator.calibrate(raw_fp)
    else:
        cal_fp = raw_fp
        risk   = (RiskLevel.LOW    if raw_fp < 0.25 else
                  RiskLevel.MEDIUM if raw_fp < 0.60 else
                  RiskLevel.HIGH)

    return auth, cal_fp, risk


###############################################################################
# SECTION 17 — AUDIT LOG WRITER
###############################################################################

def _serialise(v):
    """JSON-safe serialisation of common forensic result types."""
    if isinstance(v, np.ndarray):
        return "<ndarray>"
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, float) and math.isinf(v):
        return None
    return v


def build_log_payload(report: ForensicReport) -> dict:
    def safe_dict(obj) -> dict:
        return {k: _serialise(v) for k, v in asdict(obj).items()}

    return {
        "session_id":            report.session_id,
        "timestamp_utc":         report.fingerprint.timestamp,
        "processing_time_sec":   round(report.processing_time_sec, 4),
        "query_path":            report.query_path,
        "best_match_path":       report.best_match_path,
        "risk_level":            report.risk_level.value,
        "fraud_probability":     round(report.fraud_probability, 4),
        "authenticity_score":    round(report.authenticity_score, 4),
        "descriptor_similarity": round(report.descriptor_similarity, 4),
        "geometric":    safe_dict(report.geometric)    if report.geometric    else {},
        "tamper":       safe_dict(report.tamper)       if report.tamper       else {},
        "multiscale":   safe_dict(report.multiscale)   if report.multiscale   else {},
        "anti_spoof":   safe_dict(report.anti_spoof)   if report.anti_spoof   else {},
        "regions":      [safe_dict(r) for r in report.region_results],
        "fingerprint":  {
            "image_sha256":    report.fingerprint.image_sha256,
            "descriptor_hash": report.fingerprint.descriptor_hash,
            "file_size_bytes": report.fingerprint.file_size_bytes,
        },
    }


def write_audit_log(
    report:      ForensicReport,
    output_dir:  str = "./audit_logs",
    signer:      Optional[LogSigner] = None,
    chain_store: Optional[AuditChainStore] = None,
) -> Tuple[str, Optional[dict]]:
    """Write a signed, optionally chain-stored audit log.  Returns (path, sig_block)."""
    os.makedirs(output_dir, exist_ok=True)
    payload       = build_log_payload(report)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    sig_block: Optional[dict] = None
    if signer is not None:
        try:
            sig_block = signer.sign_payload(payload_bytes)
            logger.info("[%s] Audit log signed with RSA-PSS and ECDSA.", report.session_id)
        except Exception as exc:
            logger.warning("[%s] Log signing failed: %s", report.session_id, exc)

    log_path = os.path.join(output_dir, f"audit_{report.session_id}.json")
    with open(log_path, "w") as f:
        json.dump({"payload": payload, "signatures": sig_block or {}}, f, indent=2)

    if chain_store is not None:
        chain_store.append(payload, sig_block)

    logger.info("[%s] Audit log written: %s", report.session_id, log_path)
    return log_path, sig_block


###############################################################################
# SECTION 18 — TEMPLATE ALIGNMENT
###############################################################################

def align_to_template(image_bgr: np.ndarray, template_bgr: np.ndarray) -> np.ndarray:
    """
    Align image_bgr to template_bgr via ORB feature matching and RANSAC homography.
    Returns the warped image, or the original image if alignment fails.
    """
    orb = cv2.ORB_create(nfeatures=5000)
    g1  = cv2.cvtColor(image_bgr,    cv2.COLOR_BGR2GRAY)
    g2  = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    kp1, d1 = orb.detectAndCompute(g1, None)
    kp2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        return image_bgr

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms  = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(ms) < 10:
        return image_bgr

    src = np.float32([kp1[m.queryIdx].pt for m in ms])
    dst = np.float32([kp2[m.trainIdx].pt for m in ms])

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=2000, confidence=0.999)
    if H is None or (mask is not None and mask.sum() < 10):
        return image_bgr

    h, w = template_bgr.shape[:2]
    return cv2.warpPerspective(image_bgr, H, (w, h))


###############################################################################
# SECTION 19 — FORENSIC VERIFICATION ENGINE
###############################################################################

class ForensicVerificationEngine:
    """
    Orchestrates all 15 verification layers into a single, production-ready pipeline.

    Parameters
    ----------
    weights_path : str
        Path to SuperPoint pre-trained weights (.pth file).
    conf_thresh : float
        Keypoint confidence threshold (lower = more keypoints, more noise).
    nms_dist : int
        Non-maximum suppression radius in pixels.
    border_remove : int
        Pixels to exclude from image borders.
    cuda : bool
        Use GPU inference if available.
    audit_log_dir : str
        Directory for per-session JSON audit logs.
    chain_store_dir : str or None
        Directory for hash-chained audit storage.  None to disable.
    region_defs : dict
        Normalised region definitions {name: (x0, y0, x1, y1)}.
    run_multiscale : bool
        Enable three-resolution consistency check.
    run_region_verification : bool
        Enable per-zone document analysis.
    run_anti_spoof : bool
        Enable anti-spoof detection.
    base_size : tuple
        (width, height) for normalised image processing.
    signer : LogSigner or None
        Cryptographic audit log signer.
    calibrator : FraudScoreCalibrator or None
        Fitted score calibrator for probability calibration.
    cache_size : int
        Maximum number of database descriptors to cache.
    max_workers : int
        Number of threads for parallel database scanning.
    threshold_persistence_path : str or None
        Path for persisting adaptive thresholds across restarts.
    """

    def __init__(
        self,
        weights_path:               str,
        conf_thresh:                float = 0.005,
        nms_dist:                   int   = 4,
        border_remove:              int   = 4,
        cuda:                       bool  = False,
        audit_log_dir:              str   = "./audit_logs",
        chain_store_dir:            Optional[str] = "./audit_store",
        region_defs:                Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        run_multiscale:             bool  = True,
        run_region_verification:    bool  = True,
        run_anti_spoof:             bool  = True,
        base_size:                  Tuple[int, int] = (640, 480),
        signer:                     Optional[LogSigner] = None,
        calibrator:                 Optional[FraudScoreCalibrator] = None,
        cache_size:                 int   = 512,
        max_workers:                int   = 4,
        threshold_persistence_path: Optional[str] = None,
    ) -> None:

        self._thresh_manager  = AdaptiveThresholdManager(
            persistence_path=threshold_persistence_path
        )
        self._cache           = DescriptorCache(max_entries=cache_size)
        self.scanner          = ImageScanner(
            weights_path, conf_thresh, nms_dist, border_remove, cuda,
            cache=self._cache,
        )
        self.audit_log_dir    = audit_log_dir
        self.region_defs      = region_defs or DEFAULT_REGION_DEFINITIONS
        self.run_multiscale   = run_multiscale
        self.run_region_verify= run_region_verification
        self.run_anti_spoof   = run_anti_spoof
        self.base_size        = base_size
        self.signer           = signer
        self.calibrator       = calibrator
        self.max_workers      = max_workers
        self.anti_spoof_det   = (AntiSpoofDetector(self._thresh_manager)
                                  if run_anti_spoof else None)
        self.chain_store      = (AuditChainStore(chain_store_dir)
                                  if chain_store_dir else None)

    # ------------------------------------------------------------------
    def verify(
        self,
        query_path:      str,
        database_dir:    str,
        match_threshold: float = 0.70,
        max_keypoints:   int   = 1000,
        visualize:       bool  = False,
        align_template:  bool  = False,
    ) -> ForensicReport:
        """
        Run the full verification pipeline on a single query image.

        Parameters
        ----------
        query_path : str
            Path to the query document image.
        database_dir : str
            Directory containing reference document images.
        match_threshold : float
            Minimum mean descriptor similarity to consider a database match.
        max_keypoints : int
            Maximum keypoints to extract per image.
        visualize : bool
            Display match visualisation and tamper heatmap windows.
        align_template : bool
            Pre-align the query to the best matching template before analysis.

        Returns
        -------
        ForensicReport
        """
        t_start = time.perf_counter()
        sid     = str(uuid.uuid4())[:12]
        logger.info("[%s] Verification started: %s", sid, query_path)

        if not os.path.isfile(query_path):
            raise FileNotFoundError(f"Query image not found: {query_path}")
        if not os.path.isdir(database_dir):
            raise NotADirectoryError(f"Database directory not found: {database_dir}")

        # Layer 1 — Extract query descriptors
        try:
            q_pts, q_desc = self.scanner.scan_image(query_path, max_keypoints=max_keypoints)
        except IOError as exc:
            logger.error("[%s] Cannot read query image: %s", sid, exc)
            fp = IntegrityFingerprint("", "", 0, time.time(), sid)
            return self._empty(sid, query_path, fp, t_start)

        fp = compute_fingerprint(query_path, q_desc, sid)

        if q_desc.shape[1] == 0:
            logger.warning("[%s] No keypoints detected in query image.", sid)
            report = self._empty(sid, query_path, fp, t_start)
            self._finalise(report)
            return report

        # Database scan — parallel with thread pool
        t_db_start    = time.perf_counter()
        db_candidates = self._scan_database(
            database_dir, q_pts, q_desc, max_keypoints, match_threshold
        )
        db_time = time.perf_counter() - t_db_start
        logger.info(
            "[%s] Database scan complete in %.2fs (%d candidates evaluated, cache hit rate %.1f%%)",
            sid, db_time, len(db_candidates) + 1,
            self._cache.hit_rate * 100,
        )

        best = max(db_candidates, key=lambda x: x[0]) if db_candidates else None

        if best is None or best[0] < match_threshold:
            best_score = best[0] if best else 0.0
            logger.info("[%s] No match meets threshold (best=%.4f).", sid, best_score)
            db_stats = DatabaseStats(
                total_documents  = len(db_candidates) + 1,
                cached_documents = self._cache.size,
                cache_hit_rate   = self._cache.hit_rate,
                index_build_time = db_time,
            )
            report = self._empty(sid, query_path, fp, t_start, db_stats)
            self._finalise(report)
            self._print(report)
            return report

        best_score, best_path, qp, dbp, mb = best

        # Optional template alignment
        q_bgr  = cv2.imread(query_path)
        db_bgr = cv2.imread(best_path)

        if align_template and q_bgr is not None and db_bgr is not None:
            q_bgr = align_to_template(q_bgr, db_bgr)

        # Layer 2 — Geometric consistency
        geo = compute_geometric_consistency(qp, dbp, mb)

        # Recompute inlier mask for tamper localisation
        inlier_bool: Optional[np.ndarray] = None
        if geo.homography is not None and len(mb) >= 4:
            src_a = np.float32([[qp[0, i],  qp[1, i]]  for i, j, _ in mb])
            dst_a = np.float32([[dbp[0, j], dbp[1, j]] for i, j, _ in mb])
            _, mr = cv2.findHomography(src_a, dst_a, cv2.RANSAC, 4.0,
                                       maxIters=2000, confidence=0.999)
            if mr is not None:
                inlier_bool = mr.ravel().astype(bool)

        # Layer 3 — Tamper localisation
        shape  = q_bgr.shape if q_bgr is not None else (480, 640, 3)
        tamper = compute_tamper_localization(qp, dbp, mb, inlier_bool, shape)

        # Layer 8 — Anti-spoof (run before scoring so penalty is included)
        anti_spoof: Optional[AntiSpoofResult] = None
        if self.anti_spoof_det is not None:
            try:
                anti_spoof = self.anti_spoof_det.analyse(query_path)
            except Exception as exc:
                logger.warning("[%s] Anti-spoof analysis failed: %s", sid, exc)

        # Layer 4 — Multi-scale verification
        ms: Optional[MultiScaleResult] = None
        if self.run_multiscale:
            try:
                ms = compute_multiscale_verification(
                    self.scanner, query_path, best_path, self.base_size
                )
            except Exception as exc:
                logger.warning("[%s] Multi-scale verification failed: %s", sid, exc)

        # Layer 5 — Region verification
        regions: List[RegionResult] = []
        if self.run_region_verify and q_bgr is not None and db_bgr is not None:
            try:
                regions = verify_regions(
                    self.scanner, q_bgr, db_bgr, self.region_defs, self._thresh_manager
                )
            except Exception as exc:
                logger.warning("[%s] Region verification failed: %s", sid, exc)

        # Layer 7 — Fraud scoring
        auth, cal_fp, risk = compute_fraud_score(
            best_score, geo, tamper, ms, anti_spoof, self.calibrator
        )

        # Update adaptive thresholds with observed match score
        self._thresh_manager.update("match", best_score)
        self._thresh_manager.update("inlier_ratio", geo.inlier_ratio)
        if geo.reprojection_error < float("inf"):
            self._thresh_manager.update("reprojection", geo.reprojection_error)

        db_stats = DatabaseStats(
            total_documents  = len(db_candidates) + 1,
            cached_documents = self._cache.size,
            cache_hit_rate   = self._cache.hit_rate,
            index_build_time = db_time,
        )

        report = ForensicReport(
            session_id            = sid,
            query_path            = query_path,
            best_match_path       = best_path,
            risk_level            = risk,
            fraud_probability     = cal_fp,
            authenticity_score    = auth,
            descriptor_similarity = best_score,
            geometric             = geo,
            tamper                = tamper,
            multiscale            = ms,
            anti_spoof            = anti_spoof,
            region_results        = regions,
            fingerprint           = fp,
            db_stats              = db_stats,
            processing_time_sec   = time.perf_counter() - t_start,
        )

        self._finalise(report)

        if visualize:
            self._viz(query_path, best_path, qp, dbp, mb, tamper, report)

        self._print(report)
        return report

    # ------------------------------------------------------------------
    def verify_batch(
        self,
        query_paths:     List[str],
        database_dir:    str,
        match_threshold: float = 0.70,
        max_keypoints:   int   = 1000,
        max_workers:     Optional[int] = None,
    ) -> List[ForensicReport]:
        """
        Verify multiple query images concurrently.

        Returns a list of ForensicReport objects in the same order as query_paths.
        Failed verifications return a HIGH-risk empty report rather than raising.
        """
        workers  = max_workers or self.max_workers
        results  = [None] * len(query_paths)
        futures  = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, qp in enumerate(query_paths):
                future = pool.submit(
                    self.verify, qp, database_dir, match_threshold, max_keypoints
                )
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Batch verification failed for index %d: %s", idx, exc)
                    fp = IntegrityFingerprint("", "", 0, time.time(), "batch-error")
                    results[idx] = self._empty("batch-error", query_paths[idx], fp)

        return results

    # ------------------------------------------------------------------
    def _scan_database(
        self,
        database_dir:    str,
        q_pts:           np.ndarray,
        q_desc:          np.ndarray,
        max_keypoints:   int,
        match_threshold: float,
    ) -> List[Tuple[float, str, np.ndarray, np.ndarray, list]]:
        """
        Scan the reference database and return all candidates that may meet
        the match threshold.  Scanning is parallelised across max_workers threads.
        """
        qd = q_desc.T / (np.linalg.norm(q_desc.T, axis=1, keepdims=True) + 1e-8)

        files = [
            os.path.join(database_dir, f)
            for f in sorted(os.listdir(database_dir))
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]

        candidates: List[Tuple[float, str, np.ndarray, np.ndarray, list]] = []
        lock = threading.Lock()

        def process(path: str) -> None:
            try:
                db_pts, db_desc = self.scanner.scan_image(path, max_keypoints=max_keypoints)
            except Exception as exc:
                logger.warning("Skipping database image %s: %s", path, exc)
                return

            if db_desc.shape[1] == 0:
                return

            dd  = db_desc.T / (np.linalg.norm(db_desc.T, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(qd, dd.T)
            fwd = np.argmax(sim, axis=1)
            bwd = np.argmax(sim, axis=0)
            matches = [
                (i, fwd[i], float(sim[i, fwd[i]]))
                for i in range(len(fwd)) if bwd[fwd[i]] == i
            ]

            if not matches:
                return

            avg = float(np.mean([s for _, _, s in matches]))
            # Admit candidates well below threshold — final ranking decides the winner
            if avg >= match_threshold * 0.80:
                with lock:
                    candidates.append((avg, path, q_pts, db_pts, matches))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            list(pool.map(process, files))

        return candidates

    # ------------------------------------------------------------------
    def _finalise(self, report: ForensicReport) -> None:
        log_path, sig_block = write_audit_log(
            report,
            output_dir  = self.audit_log_dir,
            signer      = self.signer,
            chain_store = self.chain_store,
        )
        report.audit_log_path  = log_path
        report.audit_signature = sig_block

    # ------------------------------------------------------------------
    def _print(self, r: ForensicReport) -> None:
        sep = "-" * 72
        print(f"\n{sep}")
        print(f"  FORENSIC VERIFICATION REPORT       Session : {r.session_id}")
        print(sep)
        print(f"  Query              : {r.query_path}")
        print(f"  Best Match         : {r.best_match_path or 'None'}")
        print(f"  Risk Level         : {r.risk_level.value}")
        print(f"  Fraud Probability  : {r.fraud_probability * 100:.1f}%")
        print(f"  Authenticity Score : {r.authenticity_score * 100:.1f}%")
        print(f"  Descriptor Sim.    : {r.descriptor_similarity:.4f}")
        print(f"  Processing Time    : {r.processing_time_sec:.2f}s")

        if r.geometric:
            g = r.geometric
            print(f"\n  Geometric Consistency")
            print(f"    Inliers            : {g.inlier_count}  ({g.inlier_ratio * 100:.1f}%)")
            print(f"    Reprojection Error : {g.reprojection_error:.2f} px")
            print(f"    H Stability Score  : {g.homography_stability:.4f}")
            print(f"    Condition Number   : {'inf' if math.isinf(g.condition_number) else f'{g.condition_number:.2f}'}")

        if r.tamper:
            t = r.tamper
            print(f"\n  Tamper Analysis")
            print(f"    Unmatched Ratio    : {t.unmatched_ratio * 100:.1f}%")
            print(f"    Struct. Deviation  : {t.structural_deviation:.4f}")
            print(f"    Suspicious Zones   : {', '.join(t.suspicious_quadrants) or 'None'}")
            if t.quadrant_scores:
                for qname, qscore in t.quadrant_scores.items():
                    print(f"      {qname:<16} : {qscore:.4f}")

        if r.multiscale:
            ms = r.multiscale
            status = "stable" if ms.consistency_score < 0.05 else "unstable"
            print(f"\n  Multi-Scale Consistency")
            print(f"    Original           : {ms.original_score:.4f}")
            print(f"    Downscale          : {ms.downscale_score:.4f}")
            print(f"    Upscale            : {ms.upscale_score:.4f}")
            print(f"    Consistency Score  : {ms.consistency_score:.4f}  ({status})")
            print(f"    Scale Variance     : {ms.scale_variance:.6f}")

        if r.anti_spoof:
            a = r.anti_spoof
            print(f"\n  Anti-Spoof Analysis")
            print(f"    Moire Score        : {a.moire_score:.4f}")
            print(f"    Photocopy Score    : {a.photocopy_score:.4f}")
            print(f"    Screen Replay      : {a.screen_replay_score:.4f}")
            print(f"    Print-Scan Score   : {a.print_scan_score:.4f}")
            print(f"    Compression Score  : {a.compression_score:.4f}")
            print(f"    Spoof Probability  : {a.overall_spoof_probability * 100:.1f}%")
            print(f"    Flags              : {', '.join(a.flags) or 'None'}")

        if r.region_results:
            print(f"\n  Region Verification")
            print(f"    {'Region':<18} {'Status':<12} {'Confidence':>10}  "
                  f"{'Inlier Ratio':>12}  {'Desc Sim':>8}")
            print(f"    {'-' * 64}")
            for rr in r.region_results:
                print(f"    {rr.name:<18} {rr.status:<12} "
                      f"{rr.confidence * 100:>9.1f}%  "
                      f"{rr.inlier_ratio * 100:>11.1f}%  "
                      f"{rr.avg_descriptor_sim:>8.4f}")

        if r.db_stats:
            ds = r.db_stats
            print(f"\n  Database Statistics")
            print(f"    Documents Evaluated: {ds.total_documents}")
            print(f"    Cached Descriptors : {ds.cached_documents}")
            print(f"    Cache Hit Rate     : {ds.cache_hit_rate * 100:.1f}%")
            print(f"    DB Scan Time       : {ds.index_build_time:.2f}s")

        if r.audit_signature:
            sig = r.audit_signature
            print(f"\n  Audit Signatures")
            print(f"    RSA-PSS-4096       : {sig.get('rsa_signature', '')[:40]}...")
            print(f"    ECDSA-P384         : {sig.get('ec_signature',  '')[:40]}...")

        print(f"\n  Audit Log          : {r.audit_log_path}")
        print(f"  Image SHA-256      : {r.fingerprint.image_sha256[:40]}...")
        print(f"  File Size          : {r.fingerprint.file_size_bytes:,} bytes")
        print(sep + "\n")

    # ------------------------------------------------------------------
    def _viz(
        self,
        qp:      str,
        dbp:     str,
        q_pts:   np.ndarray,
        db_pts:  np.ndarray,
        matches: list,
        tamper:  TamperResult,
        report:  ForensicReport,
    ) -> None:
        qg  = cv2.imread(qp,  cv2.IMREAD_GRAYSCALE)
        dbg = cv2.imread(dbp, cv2.IMREAD_GRAYSCALE)
        if qg is None or dbg is None:
            return

        kp1 = [cv2.KeyPoint(float(q_pts[0, i]),  float(q_pts[1, i]),  1)
               for i in range(q_pts.shape[1])]
        kp2 = [cv2.KeyPoint(float(db_pts[0, j]), float(db_pts[1, j]), 1)
               for j in range(db_pts.shape[1])]
        dms = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=1.0 - s)
               for i, j, s in matches]

        vis = cv2.drawMatches(
            qg, kp1, dbg, kp2, dms, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        label = (
            f"Risk: {report.risk_level.value}  "
            f"Auth: {report.authenticity_score * 100:.1f}%  "
            f"Fraud: {report.fraud_probability * 100:.1f}%"
        )
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        cv2.imshow("Forensic Match Visualisation", vis)

        if tamper.deviation_heatmap is not None:
            cv2.imshow("Tamper Deviation Heatmap", tamper.deviation_heatmap)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    def _empty(
        self,
        sid:      str,
        qp:       str,
        fp:       IntegrityFingerprint,
        t_start:  float = 0.0,
        db_stats: Optional[DatabaseStats] = None,
    ) -> ForensicReport:
        return ForensicReport(
            session_id            = sid,
            query_path            = qp,
            best_match_path       = None,
            risk_level            = RiskLevel.HIGH,
            fraud_probability     = 1.0,
            authenticity_score    = 0.0,
            descriptor_similarity = 0.0,
            geometric             = None,
            tamper                = None,
            multiscale            = None,
            anti_spoof            = None,
            region_results        = [],
            fingerprint           = fp,
            db_stats              = db_stats,
            processing_time_sec   = time.perf_counter() - t_start if t_start else 0.0,
        )


###############################################################################
# ENTRY POINT
###############################################################################

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # Key generation (run once at initial deployment, then comment out)
    # key_paths = LogSigner.generate_keys("./keys")
    # print("Keys written:", key_paths)
    # sys.exit(0)

    # ------------------------------------------------------------------
    # Load signer
    signer: Optional[LogSigner] = None
    if CRYPTO_AVAILABLE:
        rsa_path = "./keys/rsa_private.pem"
        ec_path  = "./keys/ec_private.pem"
        if os.path.isfile(rsa_path) and os.path.isfile(ec_path):
            with open(rsa_path, "rb") as f: rsa_pem = f.read()
            with open(ec_path,  "rb") as f: ec_pem  = f.read()
            signer = LogSigner(rsa_pem, ec_pem)
            logger.info("Log signer initialised.")
        else:
            logger.warning("Key files not found — audit log signing disabled.")

    # ------------------------------------------------------------------
    # Load calibrator
    calibrator: Optional[FraudScoreCalibrator] = None
    cal_path = "./calibrator.json"
    if os.path.isfile(cal_path):
        calibrator = FraudScoreCalibrator.load(cal_path)

    # ------------------------------------------------------------------
    # Initialise engine
    engine = ForensicVerificationEngine(
        weights_path               = "superpoint_v1.pth",
        conf_thresh                = 0.005,
        nms_dist                   = 4,
        border_remove              = 4,
        cuda                       = False,
        audit_log_dir              = "./audit_logs",
        chain_store_dir            = "./audit_store",
        run_multiscale             = True,
        run_region_verification    = True,
        run_anti_spoof             = True,
        base_size                  = (640, 480),
        signer                     = signer,
        calibrator                 = calibrator,
        cache_size                 = 512,
        max_workers                = 4,
        threshold_persistence_path = "./adaptive_thresholds.json",
    )

    # ------------------------------------------------------------------
    # Single-image verification
    report = engine.verify(
        query_path      = r"C:\your_path\query_030.png",
        database_dir    = r"C:\your_path\dataset\database",
        match_threshold = 0.70,
        max_keypoints   = 1000,
        visualize       = True,
        align_template  = False,
    )

    # ------------------------------------------------------------------
    # Audit chain integrity check
    if engine.chain_store is not None:
        valid, violations = engine.chain_store.verify_chain()
        print(f"\nAudit chain integrity: {'VALID' if valid else 'COMPROMISED'}")
        for v in violations:
            print(f"  VIOLATION: {v}")

    # ------------------------------------------------------------------
    # Calibration example
    # Collect raw_scores and labels from a labelled validation set, then:
    #
    #   cal = FraudScoreCalibrator()
    #   result = cal.fit(raw_scores=[0.12, 0.87, ...], labels=[0, 1, ...])
    #   print(result)
    #   cal.save("./calibrator.json")

    # ------------------------------------------------------------------
    # Batch verification example
    # query_list = ["doc1.png", "doc2.png", "doc3.png"]
    # reports = engine.verify_batch(query_list, database_dir=r"C:\your_path\database")
    # for r in reports:
    #     print(r.session_id, r.risk_level.value, f"{r.fraud_probability * 100:.1f}%")
