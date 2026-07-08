# =========================================================
# CTR-GCN training for COCO17, M=1 (attacker-only dataset)
# - Compatible with dataset_builder M=1 outputs
# - Soft quality weight sample["q"] in loss + optional in sampler
# - Early stopping
# - Metrics + plots:
#     1) Top-1 Accuracy + Macro F1
#     2) Per-class Precision/Recall/F1 table (TXT + JSON)
#     3) Confusion matrix (counts + normalized)  [BLUE/WHITE]
#     4) Top-3 accuracy
#     5) Loss curves (Train vs Val)
#     6) Accuracy curves (Train vs Val)  -> Top-1 + Top-3
# - Saves BEST "bundle" checkpoint for real-time detection
#
# Added stronger CCTV-style augmentations (2D skeleton):
#   (1) Temporal jitter / random crop (small shift + crop)
#   (2) Keypoint dropout + LIMB dropout (structured occlusion)
#   (3) Mild coordinate noise + mild TIME WARP (speed change)
# =========================================================

import os
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from ctrgcn_model import MultiStreamCTRGCN, build_adjacency

try:
    from sklearn.metrics import classification_report
    SKLEARN_OK = True
except Exception as e:
    SKLEARN_OK = False
    print("   Scikit-learn is optional (classification report).")
    print("   Install: pip install scikit-learn")
    print("   Error:", repr(e))

# -------------------------
# CONFIG (EDIT)
# -------------------------
DATA_DIR   = r"D:\NTU_DATASET_OUTPUT"
OUT_DIR    = r"D:\NTU_DATASET_OUTPUT\RESULTS_MAR3"

TRAIN_PKL  = os.path.join(DATA_DIR, "train.pkl")
VAL_PKL    = os.path.join(DATA_DIR, "val.pkl")
TEST_PKL   = os.path.join(DATA_DIR, "test.pkl")
CLASS_MAP  = os.path.join(DATA_DIR, "class_map_new.json")
NORM_STATS = os.path.join(DATA_DIR, "norm_stats_coco17_2d_T36_M1.pkl")

T_WINDOW = 36
MAX_M    = 1

EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

GRAD_CLIP_ENABLE = True
GRAD_CLIP_NORM   = 1.0

USE_CLASS_AWARE_SAMPLING = True
LABEL_SMOOTHING = 0.05

USE_Q_WEIGHTING_IN_LOSS    = True
USE_Q_WEIGHTING_IN_SAMPLER = True

TRAIN_T_CHOICES = [18, 27, 36]  # variable-T masking choices for augmentation

# -------------------------
# AUGMENTATION (CCTV-oriented, 2D skeleton)
#   Designed for your classes: Falling Down, Punch, Kick, Chest Pain, Running, Walking
# -------------------------
AUG_ENABLE = True

# Spatial aug (x,y). Keep small for 2D normalized skeletons.
AUG_FLIP_PROB   = 0.5
AUG_ROT_DEG     = 8.0
AUG_SCALE_MIN   = 0.92
AUG_SCALE_MAX   = 1.08
AUG_TRANS_MAX   = 0.035

# Pose noise / occlusion realism
AUG_XY_NOISE_STD        = 0.010
AUG_JOINT_DROPOUT_PROB  = 0.045
AUG_FRAME_DROPOUT_PROB  = 0.025

# Confidence channel (z) perturbation
AUG_SCORE_DROP_PROB     = 0.18
AUG_SCORE_DROP_RANGE    = (0.35, 0.80)

# Temporal shift (existing)
AUG_TIME_SHIFT_MAX      = 2

# NEW: Temporal jitter / random crop (small shift + crop)
AUG_TEMP_CROP_PROB      = 0.35     # apply probability
AUG_TEMP_CROP_MAX_SHIFT = 2        # start offset +/- frames
AUG_TEMP_CROP_DROP_MAX  = 2        # crop can shorten valid L by up to this many frames

# NEW: Limb dropout (structured occlusion)
AUG_LIMB_DROPOUT_PROB   = 0.22     # probability per sample
AUG_LIMB_DROPOUT_FRAC   = (0.25, 0.55)  # fraction of valid frames to occlude when applied

# NEW: Mild time warp (speed change / timing drift)
AUG_TIME_WARP_PROB      = 0.35
AUG_TIME_WARP_FACTOR    = (0.90, 1.12)  # <1 slows, >1 speeds

# EXTRA CCTV AUGS: squash + quantization + bbox-drift + id-switch + block occlusion
AUG_SQUASH_PROB        = 0.30
AUG_SQUASH_ALPHA_RANGE = (0.65, 0.95)  # motion amplitude compression

AUG_QUANTIZE_PROB      = 0.30
AUG_QUANTIZE_STEP      = (0.002, 0.010)  # step in normalized coord units (tune if your coords are pixel-space)

AUG_DRIFT_PROB         = 0.30
AUG_DRIFT_STEP_STD     = 0.003   # random-walk step std per frame (normalized units)
AUG_DRIFT_MAX          = 0.025   # clamp drift magnitude

AUG_BLOCK_OCC_PROB     = 0.20
AUG_BLOCK_OCC_LEN      = (4, 12)  # frames
AUG_BLOCK_OCC_MODE     = "upper"  # 'upper'|'lower'|'wrists'|'random'

AUG_IDSWITCH_PROB      = 0.15
AUG_IDSWITCH_SAME_LABEL_ONLY = True

# =========================
# ADD: CCTV domain aug (skeleton equivalents)
# =========================
AUG_FRAME_HOLD_PROB      = 0.25   # simulate stream "freezes" / repeated frames
AUG_FRAME_HOLD_LEN       = (2, 6)

AUG_DUP_DROP_PROB        = 0.20   # simulate missing frames + jitter by dropping and re-sampling
AUG_DROP_FRAC_RANGE      = (0.05, 0.20)  # fraction of valid frames to drop when applied

AUG_TEMPORAL_BLUR_PROB   = 0.25   # motion blur / low shutter -> temporal smoothing
AUG_TEMPORAL_BLUR_ALPHA  = (0.55, 0.85)  # higher => more blur (more smoothing)

AUG_LOWLIGHT_PROB        = 0.30   # low light -> lower conf + more jitter + more dropout
AUG_LOWLIGHT_CONF_SCALE  = (0.25, 0.70)
AUG_LOWLIGHT_NOISE_MULT  = (1.5, 3.5)
AUG_LOWLIGHT_DROP_MULT   = (1.5, 3.0)

# =========================
# ADD: Crowd aug for skeleton pipelines
# =========================
AUG_SWAP_NOISE_PROB      = 0.18   # mis-association: swap a few joints L/R or random
AUG_SWAP_JOINTS_RANGE    = (1, 4) # number of joints to mess with

AUG_PARTIAL_BODY_PROB    = 0.22   # partial body visible (crop upper/lower)
AUG_PARTIAL_MODE         = "random"  # "upper"|"lower"|"random"
AUG_PARTIAL_FRAC         = (0.35, 0.70)  # fraction of valid frames affected

AUG_OCCLUDE_SET_CONF0    = True   # when we occlude, also set conf channel to 0 (stronger)

TOPK = 3

EARLY_STOP_ENABLE    = True
EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 1e-4
EARLY_STOP_MONITOR   = "val_loss"

# -------------------------
# OUTPUT
# -------------------------
SAVE_DIR = os.path.join(OUT_DIR, f"ctrgcn_ckpt_coco17_T{T_WINDOW}_M{MAX_M}_M1_QUALITY_ES")
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_BUNDLE = os.path.join(SAVE_DIR, f"ctrgcn_coco17_best_bundle_T{T_WINDOW}_M{MAX_M}.pth")
LAST_CKPT   = os.path.join(SAVE_DIR, f"ctrgcn_coco17_last_T{T_WINDOW}_M{MAX_M}.pth")

METRICS_JSON    = os.path.join(SAVE_DIR, "metrics_history.json")
LOSS_CURVE_PNG  = os.path.join(SAVE_DIR, "loss_curves.png")
ACC_CURVE_PNG   = os.path.join(SAVE_DIR, "acc_curves.png")
CM_COUNTS_PNG   = os.path.join(SAVE_DIR, "confusion_counts.png")
CM_NORM_PNG     = os.path.join(SAVE_DIR, "confusion_norm.png")
ACC_OVERALL_CURVE_PNG = os.path.join(SAVE_DIR, "acc_overall_curves.png")

TEST_SUMMARY_JSON = os.path.join(SAVE_DIR, "test_summary.json")
TEST_REPORT_TXT   = os.path.join(SAVE_DIR, "per_class_report_test.txt")
TEST_REPORT_JSON  = os.path.join(SAVE_DIR, "per_class_report_test.json")

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
scaler = GradScaler(enabled=use_amp)
torch.backends.cudnn.benchmark = True
print("Device:", device, "| AMP:", use_amp)

# -------------------------
# Leak-check helper (optional)
# -------------------------
def find_id_key(samples):
    for k in ["name", "video", "video_id", "filename", "source", "path", "orig"]:
        if len(samples) > 0 and isinstance(samples[0], dict) and (k in samples[0]):
            return k
    return None

def load_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_head(pkl_path, n=2000):
    s = load_samples(pkl_path)
    return s[:min(len(s), n)]

def _to_hashable(v):
    if isinstance(v, (str, int, float)):
        return v
    return str(v)

try:
    tr_head = load_head(TRAIN_PKL, n=2000)
    va_head = load_head(VAL_PKL, n=2000)
    te_head = load_head(TEST_PKL, n=2000)

    k = find_id_key(tr_head) or find_id_key(va_head) or find_id_key(te_head)
    print("Leak-check key:", k)

    if k is not None:
        tr = load_samples(TRAIN_PKL)
        va = load_samples(VAL_PKL)
        te = load_samples(TEST_PKL)

        tr_ids = set(_to_hashable(x.get(k, "")) for x in tr)
        va_ids = set(_to_hashable(x.get(k, "")) for x in va)
        te_ids = set(_to_hashable(x.get(k, "")) for x in te)

        print("train∩val:", len(tr_ids & va_ids))
        print("train∩test:", len(tr_ids & te_ids))
        print("val∩test:", len(va_ids & te_ids))
    else:
        print("No id key found in PKL samples. (If you split by windows, leakage is still possible.)")
except Exception as e:
    print("[Leak-check] skipped due to error:", repr(e))

# -------------------------
# class map
# -------------------------
with open(CLASS_MAP, "r", encoding="utf-8") as f:
    idx_to_name = json.load(f)
CLASS_NAMES = {int(k): v for k, v in idx_to_name.items()}
NUM_CLASS = len(CLASS_NAMES)
print("Classes:", NUM_CLASS, CLASS_NAMES)

# -------------------------
# load norm stats
# -------------------------
mean = std = None
if os.path.exists(NORM_STATS):
    with open(NORM_STATS, "rb") as f:
        ns = pickle.load(f)
    mean = ns.get("mean", None)
    std  = ns.get("std", None)
    print("Loaded norm stats:", NORM_STATS)
else:
    print("[WARN] norm stats not found -> training will be unnormalized!")

def _as_ctvm_stats(x, C=3):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 4 and x.shape[0] == C:
        return x
    if x.ndim == 5 and x.shape[0] == C:
        return x[:, :, :, :, 0]
    if x.ndim == 1 and x.shape[0] == C:
        return x.reshape(C, 1, 1, 1)
    raise ValueError(f"Unsupported mean/std shape: {x.shape}")

if mean is not None and std is not None:
    mean = {
        "joint":  _as_ctvm_stats(mean["joint"],  3),
        "bone":   _as_ctvm_stats(mean["bone"],   3),
        "motion": _as_ctvm_stats(mean["motion"], 3),
    }
    std = {
        "joint":  _as_ctvm_stats(std["joint"],  3),
        "bone":   _as_ctvm_stats(std["bone"],   3),
        "motion": _as_ctvm_stats(std["motion"], 3),
    }

    STD_FLOOR = 1e-2
    for kk in ["joint", "bone", "motion"]:
        std[kk] = np.maximum(std[kk], STD_FLOOR).astype(np.float32)

    for kk in ["joint", "bone", "motion"]:
        s = std[kk]
        print(f"[STD] {kk}: min={float(np.min(s)):.6f} max={float(np.max(s)):.6f}")
else:
    print("[STD] skipped (no norm stats loaded)")

# -------------------------
# Dataset (M=1)
# -------------------------
class SkeletonDatasetM1(Dataset):
    def __init__(self, pkl_path, mean=None, std=None, augment=False,
                 desired_T=36, variable_T_choices=None):
        with open(pkl_path, "rb") as f:
            samples = pickle.load(f)

        self.samples = samples
        self.desired_T = int(desired_T)
        self.augment = bool(augment)
        self.variable_T_choices = None if variable_T_choices is None else [int(x) for x in variable_T_choices]

        self.joint, self.bone, self.motion = [], [], []
        self.labels, self.lengths, self.q = [], [], []

        for s in samples:
            j = np.array(s["joint"], dtype=np.float32)   # (3,T,17,1)
            b = np.array(s["bone"], dtype=np.float32)
            m = np.array(s["motion"], dtype=np.float32)
            y = int(s["label"])
            L = int(s.get("length", j.shape[1]))
            qv = float(s.get("q", 1.0))

            if j.ndim == 3:
                j = j[..., None]
                b = b[..., None]
                m = m[..., None]

            if j.shape[-1] != 1:
                j = j[..., :1]
                b = b[..., :1]
                m = m[..., :1]

            C, T, V, _ = j.shape

            if T < self.desired_T:
                pad_T = self.desired_T - T
                j = np.concatenate([j, np.zeros((C, pad_T, V, 1), np.float32)], axis=1)
                b = np.concatenate([b, np.zeros((C, pad_T, V, 1), np.float32)], axis=1)
                m = np.concatenate([m, np.zeros((C, pad_T, V, 1), np.float32)], axis=1)
            elif T > self.desired_T:
                j = j[:, :self.desired_T]
                b = b[:, :self.desired_T]
                m = m[:, :self.desired_T]

            L = max(1, min(L, self.desired_T))

            self.joint.append(j)
            self.bone.append(b)
            self.motion.append(m)
            self.labels.append(y)
            self.lengths.append(L)
            self.q.append(qv)

        self.joint   = np.stack(self.joint,  0)
        self.bone    = np.stack(self.bone,   0)
        self.motion  = np.stack(self.motion, 0)
        self.labels  = np.asarray(self.labels, np.int64)
        self.lengths = np.asarray(self.lengths, np.int64)
        self.q       = np.asarray(self.q, np.float32)

        # For CCTV-style ID-switch augmentation (track swaps), keep label->indices lookup
        self.label_to_indices = {}
        for ii, yy in enumerate(self.labels.tolist()):
            self.label_to_indices.setdefault(int(yy), []).append(int(ii))

        if mean is not None and std is not None:
            self.joint  = (self.joint  - mean["joint"])  / (std["joint"]  + 1e-6)
            self.bone   = (self.bone   - mean["bone"])   / (std["bone"]   + 1e-6)
            self.motion = (self.motion - mean["motion"]) / (std["motion"] + 1e-6)

        if mean is None or std is None:
            self.PAD_J = np.zeros((3, 1, 1, 1), np.float32)
            self.PAD_B = np.zeros((3, 1, 1, 1), np.float32)
            self.PAD_M = np.zeros((3, 1, 1, 1), np.float32)
        else:
            self.PAD_J = (0.0 - mean["joint"])  / (std["joint"]  + 1e-6)
            self.PAD_B = (0.0 - mean["bone"])   / (std["bone"]   + 1e-6)
            self.PAD_M = (0.0 - mean["motion"]) / (std["motion"] + 1e-6)

    # -------------------------
    # COCO17 joint LR swap pairs (indices)
    # -------------------------
    _COCO17_LR_PAIRS = [
        (1, 2),    # left_eye <-> right_eye
        (3, 4),    # left_ear <-> right_ear
        (5, 6),    # left_shoulder <-> right_shoulder
        (7, 8),    # left_elbow <-> right_elbow
        (9, 10),   # left_wrist <-> right_wrist
        (11, 12),  # left_hip <-> right_hip
        (13, 14),  # left_knee <-> right_knee
        (15, 16),  # left_ankle <-> right_ankle
    ]

    # Limb groups (structured occlusion) — COCO17
    _LIMB_GROUPS = {
        "left_arm":  [5, 7, 9],
        "right_arm": [6, 8, 10],
        "left_leg":  [11, 13, 15],
        "right_leg": [12, 14, 16],
    }

    def _hflip_coco17(self, arr):
        out = arr.copy()
        out[0] = -out[0]  # x -> -x
        for l, r in self._COCO17_LR_PAIRS:
            tmp = out[:, :, l:l+1, :].copy()
            out[:, :, l:l+1, :] = out[:, :, r:r+1, :]
            out[:, :, r:r+1, :] = tmp
        return out

    def _apply_affine_xy(self, arr, rot_deg=0.0, scale=1.0, tx=0.0, ty=0.0, translate=True):
        if rot_deg == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
            return arr
        out = arr.copy()
        th = np.deg2rad(rot_deg)
        c, s = float(np.cos(th)), float(np.sin(th))
        x = out[0].copy()
        y = out[1].copy()
        xr = scale * (c * x - s * y)
        yr = scale * (s * x + c * y)
        if translate:
            xr = xr + tx
            yr = yr + ty
        out[0] = xr
        out[1] = yr
        return out

    def _time_shift(self, arr, L, shift, pad_value):
        if shift == 0 or L <= 1:
            return arr, int(L)

        C, T, V, M = arr.shape
        out = arr.copy()

        valid = out[:, :L].copy()
        out[:, :, :, :] = pad_value  # fully padded

        if shift > 0:
            new_start = min(T, shift)
            new_end = min(T, shift + L)
            copy_len = max(0, new_end - new_start)
            if copy_len > 0:
                out[:, new_start:new_end] = valid[:, :copy_len]
            new_L = min(T, shift + L)
        else:
            sh = -shift
            new_start = 0
            new_end = max(0, min(T, L - sh))
            copy_len = max(0, new_end - new_start)
            if copy_len > 0:
                out[:, new_start:new_end] = valid[:, sh:sh + copy_len]
            new_L = max(1, L - sh)

        return out, int(max(1, min(new_L, T)))

    # NEW: temporal random crop / jitter (keeps fixed desired_T, crops within valid L)
    def _temporal_crop_jitter(self, arr, L, start, new_L, pad_value):
        """
        Keep arr shape (C,T,V,1) but replace content with cropped valid segment.
        start: crop start within [0, L-1]
        new_L: crop length within [1, L]
        """
        C, T, V, M = arr.shape
        out = arr.copy()
        out[:, :, :, :] = pad_value

        start = int(np.clip(start, 0, max(0, L - 1)))
        new_L = int(np.clip(new_L, 1, min(L, T)))

        end = min(L, start + new_L)
        seg = arr[:, start:end].copy()
        copy_len = seg.shape[1]
        out[:, :copy_len] = seg
        return out, int(copy_len)

    # NEW: time warp (speed change) via linear interpolation along time
    def _time_warp(self, arr, L, factor, pad_value):
        """
        Warp only valid [0:L) frames, output keeps same T with padded remainder.
        factor > 1.0 => faster progression; factor < 1.0 => slower.
        """
        if L <= 2:
            return arr, int(L)

        C, T, V, M = arr.shape
        out = arr.copy()
        out[:, :, :, :] = pad_value

        valid = arr[:, :L].copy()  # (C,L,V,1)

        # new time positions (length L)
        t = np.linspace(0.0, float(L - 1), num=L, dtype=np.float32)
        t_new = t * float(factor)
        t_new = np.clip(t_new, 0.0, float(L - 1))

        t0 = np.floor(t_new).astype(np.int32)
        t1 = np.clip(t0 + 1, 0, L - 1)
        w = (t_new - t0).astype(np.float32)  # (L,)

        # interpolate: valid[:, t0] and valid[:, t1]
        # shapes: (C,L,V,1)
        v0 = valid[:, t0, :, :]
        v1 = valid[:, t1, :, :]
        w_ = w.reshape(1, L, 1, 1)
        warped = (1.0 - w_) * v0 + w_ * v1

        out[:, :L] = warped
        return out, int(L)

    # NEW: limb dropout (structured occlusion)
    def _apply_limb_dropout(self, j, b, m, L, padJ, padB, padM):
        """
        Limb dropout / occlusion simulation.
        Drops a random limb group (set of joints) for a time segment inside [0, L).

        j,b,m: (3, T, V, 1)
        padJ,padB,padM: (3, 1, 1, 1) broadcastable
        """
        if L <= 2:
            return j, b, m

        # Use global prob + frac config
        prob = float(globals().get("AUG_LIMB_DROPOUT_PROB", 0.0))
        if prob <= 0 or (np.random.rand() >= prob):
            return j, b, m

        frac_lo, frac_hi = globals().get("AUG_LIMB_DROPOUT_FRAC", (0.25, 0.55))
        frac = float(np.random.uniform(frac_lo, frac_hi))
        seg_len = int(max(1, min(L, round(frac * L))))

        C, T, V, M = j.shape

        padJv = padJ[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padBv = padB[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padMv = padM[:, 0, 0, 0].reshape(3, 1, 1, 1)

        limb_groups = [
            [5, 7, 9],       # left arm
            [6, 8, 10],      # right arm
            [11, 13, 15],    # left leg
            [12, 14, 16],    # right leg
            [0, 1, 2, 5, 6], # head+shoulders
        ]

        joints = limb_groups[int(np.random.randint(0, len(limb_groups)))]
        joints = [jj for jj in joints if 0 <= jj < V]
        if not joints:
            return j, b, m

        start = int(np.random.randint(0, max(1, L - seg_len + 1)))
        end = start + seg_len

        t_idx = np.arange(start, end, dtype=np.int64)
        v_idx = np.array(joints, dtype=np.int64)

        j[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M))] = padJv
        b[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M))] = padBv
        m[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M))] = padMv

        # optionally set conf to 0 for those joints
        if bool(globals().get("AUG_OCCLUDE_SET_CONF0", True)):
            j[2, t_idx[:, None], v_idx[None, :], :] = 0.0
            b[2, t_idx[:, None], v_idx[None, :], :] = 0.0
            m[2, t_idx[:, None], v_idx[None, :], :] = 0.0

        return j, b, m

    # -------------------------
    # EXTRA CCTV AUG HELPERS
    # -------------------------
    def _apply_random_walk_drift(self, arr, L, step_std=0.003, max_abs=0.025):
        """
        Simulate detector bbox jitter as a per-frame random-walk translation.
        Applies to x/y only, keeps z/conf unchanged.
        """
        if L <= 1 or step_std <= 0:
            return arr
        out = arr.copy()
        dx = 0.0
        dy = 0.0
        for t in range(int(L)):
            dx += float(np.random.normal(0.0, step_std))
            dy += float(np.random.normal(0.0, step_std))
            dx = float(np.clip(dx, -max_abs, max_abs))
            dy = float(np.clip(dy, -max_abs, max_abs))
            out[0, t] += dx
            out[1, t] += dy
        return out

    def _apply_quantize_xy(self, arr, step=0.005):
        """
        Quantize x/y to mimic low-res / blocky keypoints.
        """
        if step <= 0:
            return arr
        out = arr.copy()
        out[0:2] = (np.round(out[0:2] / float(step)) * float(step)).astype(np.float32)
        return out

    def _apply_motion_squash(self, arr, L, alpha=0.85):
        """
        Compress motion amplitude around a per-frame center (mid-hip if available, else mean of hips).
        This makes punches/kicks look weaker like tiny/noisy CCTV poses.
        """
        if L <= 1:
            return arr
        out = arr.copy()

        V = out.shape[2]
        # COCO17 hips: 11 (L), 12 (R). Use their mean as center if both exist.
        if V >= 13:
            cx = 0.5 * (out[0, :L, 11, :] + out[0, :L, 12, :])
            cy = 0.5 * (out[1, :L, 11, :] + out[1, :L, 12, :])
        else:
            cx = np.mean(out[0, :L, :, :], axis=1, keepdims=False)
            cy = np.mean(out[1, :L, :, :], axis=1, keepdims=False)

        # broadcast to (L,V,M)
        cx = cx.reshape(L, 1, 1)
        cy = cy.reshape(L, 1, 1)

        out[0, :L] = cx + float(alpha) * (out[0, :L] - cx)
        out[1, :L] = cy + float(alpha) * (out[1, :L] - cy)
        return out

    def _apply_block_occlusion(self, j, b, m, L, padJ, padB, padM, mode="upper", len_range=(4, 12)):
        """
        Drop a contiguous time block for a joint set (more realistic crowd occlusion).
        """
        if L <= 2:
            return j, b, m

        C, T, V, M_ = j.shape
        lo, hi = int(len_range[0]), int(len_range[1])
        seg_len = int(np.random.randint(lo, hi + 1))
        seg_len = max(1, min(seg_len, L))
        start = int(np.random.randint(0, max(1, L - seg_len + 1)))
        end = start + seg_len

        if mode == "upper":
            joints = [0, 5, 6, 7, 8, 9, 10]  # head + arms
        elif mode == "lower":
            joints = [11, 12, 13, 14, 15, 16]
        elif mode == "wrists":
            joints = [9, 10]
        elif mode == "random":
            # random 4-8 joints
            k = int(np.random.randint(4, min(9, V + 1)))
            joints = np.random.choice(np.arange(V), size=k, replace=False).tolist()
        else:
            joints = [9, 10]  # safe default

        joints = [jj for jj in joints if 0 <= jj < V]
        if not joints:
            return j, b, m

        padJv = padJ[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padBv = padB[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padMv = padM[:, 0, 0, 0].reshape(3, 1, 1, 1)

        t_idx = np.arange(start, end, dtype=np.int64)
        v_idx = np.array(joints, dtype=np.int64)

        j[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padJv
        b[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padBv
        m[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padMv
        return j, b, m

    def _apply_id_switch(self, j, b, m, y, L):
        """
        Simulate tracking ID switch by replacing the tail of the sequence with another sample's tail.
        Default: choose a sample from the same label to avoid label noise.
        """
        if L <= 4:
            return j, b, m

        # choose donor index
        donor_idx = None
        if bool(globals().get("AUG_IDSWITCH_SAME_LABEL_ONLY", True)):
            cands = self.label_to_indices.get(int(y), [])
            if len(cands) >= 2:
                donor_idx = int(np.random.choice(cands))
        if donor_idx is None:
            donor_idx = int(np.random.randint(0, len(self.labels)))

        if donor_idx is None or donor_idx == -1:
            return j, b, m

        j2 = self.joint[donor_idx].copy()
        b2 = self.bone[donor_idx].copy()
        m2 = self.motion[donor_idx].copy()
        L2 = int(self.lengths[donor_idx])

        # pick switch point in middle
        t0 = int(np.random.randint(max(1, L // 4), max(2, (3 * L) // 4)))
        copy_len = int(min(L - t0, L2))
        if copy_len <= 0:
            return j, b, m

        j[:, t0:t0 + copy_len] = j2[:, :copy_len]
        b[:, t0:t0 + copy_len] = b2[:, :copy_len]
        m[:, t0:t0 + copy_len] = m2[:, :copy_len]
        return j, b, m
    
    def _temporal_smooth(self, arr, L, alpha=0.7):
        """
        Motion blur / low shutter: exponential moving average along time on x/y (and optionally conf).
        """
        if L <= 2:
            return arr
        out = arr.copy()
        for t in range(1, int(L)):
            out[0, t] = alpha * out[0, t-1] + (1.0 - alpha) * out[0, t]
            out[1, t] = alpha * out[1, t-1] + (1.0 - alpha) * out[1, t]
            # keep conf mostly intact; uncomment if you want conf smoothing too:
            # out[2, t] = alpha * out[2, t-1] + (1.0 - alpha) * out[2, t]
        return out

    def _frame_hold(self, arr, L, hold_len=3):
        """
        Stream freeze: repeat frames for a short segment.
        """
        if L <= 2:
            return arr
        out = arr.copy()
        hold_len = int(max(1, min(hold_len, L-1)))
        start = int(np.random.randint(0, max(1, L - hold_len)))
        ref = out[:, start:start+1].copy()
        out[:, start:start+hold_len] = ref
        return out

    def _drop_and_resample_time(self, arr, L, drop_frac=0.1, pad_value=None):
        """
        Frame drops / jitter: drop some frames and resample back to length L.
        Works on valid part only; keeps overall T fixed by padding remainder.
        """
        if L <= 3:
            return arr
        drop_frac = float(np.clip(drop_frac, 0.0, 0.5))
        keep = int(max(2, round((1.0 - drop_frac) * L)))
        idx = np.sort(np.random.choice(np.arange(L), size=keep, replace=False))

        valid = arr[:, :L].copy()                 # (C,L,V,1)
        kept  = valid[:, idx].copy()              # (C,keep,V,1)

        # resample kept back to length L
        t_old = np.linspace(0.0, 1.0, num=keep, dtype=np.float32)
        t_new = np.linspace(0.0, 1.0, num=L,    dtype=np.float32)
        pos = t_new * (keep - 1)
        p0 = np.floor(pos).astype(np.int32)
        p1 = np.clip(p0 + 1, 0, keep - 1)
        w  = (pos - p0).astype(np.float32).reshape(1, L, 1, 1)

        v0 = kept[:, p0, :, :]
        v1 = kept[:, p1, :, :]
        warped = (1.0 - w) * v0 + w * v1

        out = arr.copy()
        if pad_value is not None:
            out[:, :, :, :] = pad_value
        out[:, :L] = warped
        return out

    def _swap_noise(self, arr, L, num_swaps=2):
        """
        Mis-association: randomly swap a few joints' positions (x,y,conf) between joints at random times.
        """
        if L <= 2:
            return arr
        out = arr.copy()
        V = out.shape[2]
        num_swaps = int(np.clip(num_swaps, 1, max(1, V//4)))

        for _ in range(num_swaps):
            t = int(np.random.randint(0, L))
            a = int(np.random.randint(0, V))
            b = int(np.random.randint(0, V))
            if a == b:
                continue
            tmp = out[:, t, a:a+1, :].copy()
            out[:, t, a:a+1, :] = out[:, t, b:b+1, :]
            out[:, t, b:b+1, :] = tmp
        return out

    def _partial_body_mask(self, j, b, m, L, padJ, padB, padM, mode="random", frac_range=(0.35, 0.70), conf0=True):
        """
        Partial body visible: mask upper or lower joints for a portion of the clip.
        """
        if L <= 2:
            return j, b, m
        V = j.shape[2]
        frac = float(np.random.uniform(frac_range[0], frac_range[1]))
        seg_len = int(max(1, min(L, round(frac * L))))
        start = int(np.random.randint(0, max(1, L - seg_len + 1)))
        end = start + seg_len

        if mode == "random":
            mode = "upper" if np.random.rand() < 0.5 else "lower"

        if mode == "upper":
            joints = [0,1,2,3,4,5,6,7,8,9,10]     # head+arms+shoulders
        elif mode == "lower":
            joints = [11,12,13,14,15,16]          # hips+legs
        else:
            joints = [9,10]                        # safe fallback

        joints = [jj for jj in joints if 0 <= jj < V]
        if not joints:
            return j, b, m

        padJv = padJ[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padBv = padB[:, 0, 0, 0].reshape(3, 1, 1, 1)
        padMv = padM[:, 0, 0, 0].reshape(3, 1, 1, 1)

        t_idx = np.arange(start, end, dtype=np.int64)
        v_idx = np.array(joints, dtype=np.int64)
        C = 3
        M_ = j.shape[3]

        j[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padJv
        b[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padBv
        m[np.ix_(np.arange(C), t_idx, v_idx, np.arange(M_))] = padMv

        if conf0:
            j[2, t_idx[:, None], v_idx[None, :], :] = 0.0
            b[2, t_idx[:, None], v_idx[None, :], :] = 0.0
            m[2, t_idx[:, None], v_idx[None, :], :] = 0.0

        return j, b, m

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        j = self.joint[idx].copy()
        b = self.bone[idx].copy()
        m = self.motion[idx].copy()
        y = int(self.labels[idx])
        L = int(self.lengths[idx])
        qv = float(self.q[idx])

        if L < self.desired_T:
            j[:, L:] = self.PAD_J
            b[:, L:] = self.PAD_B
            m[:, L:] = self.PAD_M

        if self.augment and AUG_ENABLE:
            # 0) Temporal jitter / random crop (small shift + crop)
            if (AUG_TEMP_CROP_PROB > 0) and (np.random.rand() < AUG_TEMP_CROP_PROB) and (L > 3):
                drop = int(np.random.randint(0, AUG_TEMP_CROP_DROP_MAX + 1))
                new_L = int(max(2, min(L, L - drop)))
                start_shift = int(np.random.randint(-AUG_TEMP_CROP_MAX_SHIFT, AUG_TEMP_CROP_MAX_SHIFT + 1))
                base = int(np.random.randint(0, max(1, L - new_L + 1)))
                start = int(np.clip(base + start_shift, 0, max(0, L - new_L)))
                j, L2 = self._temporal_crop_jitter(j, L, start, new_L, self.PAD_J)
                b, _  = self._temporal_crop_jitter(b, L, start, new_L, self.PAD_B)
                m, _  = self._temporal_crop_jitter(m, L, start, new_L, self.PAD_M)
                L = int(L2)

            # 1) Temporal shift (existing)
            if AUG_TIME_SHIFT_MAX > 0 and L > 1:
                shift = int(np.random.randint(-AUG_TIME_SHIFT_MAX, AUG_TIME_SHIFT_MAX + 1))
                if shift != 0:
                    j, L = self._time_shift(j, L, shift, self.PAD_J)
                    b, _ = self._time_shift(b, L, shift, self.PAD_B)
                    m, _ = self._time_shift(m, L, shift, self.PAD_M)

            # 2) Mild time warp (speed/timing drift)
            if (AUG_TIME_WARP_PROB > 0) and (np.random.rand() < AUG_TIME_WARP_PROB) and (L > 3):
                f_lo, f_hi = AUG_TIME_WARP_FACTOR
                factor = float(np.random.uniform(f_lo, f_hi))
                j, _ = self._time_warp(j, L, factor, self.PAD_J)
                b, _ = self._time_warp(b, L, factor, self.PAD_B)
                m, _ = self._time_warp(m, L, factor, self.PAD_M)

            # 3) Horizontal flip + joint LR swap
            if np.random.rand() < AUG_FLIP_PROB:
                j = self._hflip_coco17(j)
                b = self._hflip_coco17(b)
                m = self._hflip_coco17(m)

            # 4) Mild spatial affine: rotate + scale (+ translate joints only)
            rot = float(np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)) if AUG_ROT_DEG > 0 else 0.0
            sc  = float(np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)) if (AUG_SCALE_MAX > 0) else 1.0
            tx  = float(np.random.uniform(-AUG_TRANS_MAX, AUG_TRANS_MAX)) if AUG_TRANS_MAX > 0 else 0.0
            ty  = float(np.random.uniform(-AUG_TRANS_MAX, AUG_TRANS_MAX)) if AUG_TRANS_MAX > 0 else 0.0

            j = self._apply_affine_xy(j, rot_deg=rot, scale=sc, tx=tx, ty=ty, translate=True)
            b = self._apply_affine_xy(b, rot_deg=rot, scale=sc, tx=0.0, ty=0.0, translate=False)
            m = self._apply_affine_xy(m, rot_deg=rot, scale=sc, tx=0.0, ty=0.0, translate=False)

            # 5) Gaussian jitter (pose noise) on x,y
            if AUG_XY_NOISE_STD > 0:
                j[0:2] += np.random.normal(0, AUG_XY_NOISE_STD, j[0:2].shape).astype(np.float32)
                b[0:2] += np.random.normal(0, AUG_XY_NOISE_STD, b[0:2].shape).astype(np.float32)
                m[0:2] += np.random.normal(0, AUG_XY_NOISE_STD, m[0:2].shape).astype(np.float32)


            # 5.5) Per-frame bbox drift (random-walk) — simulates detector/tracker wobble
            if (AUG_DRIFT_PROB > 0) and (np.random.rand() < AUG_DRIFT_PROB) and (L > 2):
                j = self._apply_random_walk_drift(j, L, step_std=AUG_DRIFT_STEP_STD, max_abs=AUG_DRIFT_MAX)
                b = self._apply_random_walk_drift(b, L, step_std=AUG_DRIFT_STEP_STD, max_abs=AUG_DRIFT_MAX)
                m = self._apply_random_walk_drift(m, L, step_std=AUG_DRIFT_STEP_STD, max_abs=AUG_DRIFT_MAX)

            # 5.6) Motion squash (amplitude compression) — CCTV tiny / smeared motion
            if (AUG_SQUASH_PROB > 0) and (np.random.rand() < AUG_SQUASH_PROB) and (L > 2):
                a_lo, a_hi = AUG_SQUASH_ALPHA_RANGE
                alpha = float(np.random.uniform(a_lo, a_hi))
                j = self._apply_motion_squash(j, L, alpha=alpha)
                b = self._apply_motion_squash(b, L, alpha=alpha)
                m = self._apply_motion_squash(m, L, alpha=alpha)

            # 5.7) Quantization — blocky keypoints like low-res CCTV
            if (AUG_QUANTIZE_PROB > 0) and (np.random.rand() < AUG_QUANTIZE_PROB):
                q_lo, q_hi = AUG_QUANTIZE_STEP
                step = float(np.random.uniform(q_lo, q_hi))
                j = self._apply_quantize_xy(j, step=step)
                b = self._apply_quantize_xy(b, step=step)
                m = self._apply_quantize_xy(m, step=step)

            # 5.8) ID-switch (track swap) — crowded CCTV realism (keep label-consistent by default)
            if (AUG_IDSWITCH_PROB > 0) and (np.random.rand() < AUG_IDSWITCH_PROB) and (L > 4):
                j, b, m = self._apply_id_switch(j, b, m, y=y, L=L)

            # 5.9) Block occlusion (temporal) — continuous occlusions in crowds
            if (AUG_BLOCK_OCC_PROB > 0) and (np.random.rand() < AUG_BLOCK_OCC_PROB) and (L > 3):
                j, b, m = self._apply_block_occlusion(
                    j, b, m, L=L,
                    padJ=self.PAD_J, padB=self.PAD_B, padM=self.PAD_M,
                    mode=str(AUG_BLOCK_OCC_MODE),
                    len_range=tuple(AUG_BLOCK_OCC_LEN),
                )
            # Limb dropout (structured occlusion)
            if (AUG_LIMB_DROPOUT_PROB > 0) and (L > 2):
                j, b, m = self._apply_limb_dropout(j, b, m, L, self.PAD_J, self.PAD_B, self.PAD_M)
                
            # ---- A) CCTV domain aug (skeleton equivalents) ----
            # A1) motion blur / low shutter -> temporal smoothing
            if (AUG_TEMPORAL_BLUR_PROB > 0) and (np.random.rand() < AUG_TEMPORAL_BLUR_PROB) and (L > 3):
                a_lo, a_hi = AUG_TEMPORAL_BLUR_ALPHA
                alpha = float(np.random.uniform(a_lo, a_hi))
                j = self._temporal_smooth(j, L, alpha=alpha)
                b = self._temporal_smooth(b, L, alpha=alpha)
                m = self._temporal_smooth(m, L, alpha=alpha)

            # A2) occasional stream freeze -> frame hold
            if (AUG_FRAME_HOLD_PROB > 0) and (np.random.rand() < AUG_FRAME_HOLD_PROB) and (L > 3):
                lo, hi = AUG_FRAME_HOLD_LEN
                hold_len = int(np.random.randint(lo, hi + 1))
                j = self._frame_hold(j, L, hold_len=hold_len)
                b = self._frame_hold(b, L, hold_len=hold_len)
                m = self._frame_hold(m, L, hold_len=hold_len)

            # A3) frame drops / jitter -> drop then resample
            if (AUG_DUP_DROP_PROB > 0) and (np.random.rand() < AUG_DUP_DROP_PROB) and (L > 4):
                d_lo, d_hi = AUG_DROP_FRAC_RANGE
                drop_frac = float(np.random.uniform(d_lo, d_hi))
                j = self._drop_and_resample_time(j, L, drop_frac=drop_frac, pad_value=self.PAD_J)
                b = self._drop_and_resample_time(b, L, drop_frac=drop_frac, pad_value=self.PAD_B)
                m = self._drop_and_resample_time(m, L, drop_frac=drop_frac, pad_value=self.PAD_M)

            # A4) low-light -> reduce conf + amplify noise/dropout (strong domain shift)
            if (AUG_LOWLIGHT_PROB > 0) and (np.random.rand() < AUG_LOWLIGHT_PROB):
                # confidence scale
                c_lo, c_hi = AUG_LOWLIGHT_CONF_SCALE
                cscale = float(np.random.uniform(c_lo, c_hi))
                j[2] *= cscale
                b[2] *= cscale
                m[2] *= cscale

                # amplify coordinate noise a bit
                n_lo, n_hi = AUG_LOWLIGHT_NOISE_MULT
                mult = float(np.random.uniform(n_lo, n_hi))
                if AUG_XY_NOISE_STD > 0:
                    std2 = float(AUG_XY_NOISE_STD) * mult
                    j[0:2] += np.random.normal(0, std2, j[0:2].shape).astype(np.float32)
                    b[0:2] += np.random.normal(0, std2, b[0:2].shape).astype(np.float32)
                    m[0:2] += np.random.normal(0, std2, m[0:2].shape).astype(np.float32)

                # amplify dropout probabilities by multiplying a mask (without changing globals)
                d_lo, d_hi = AUG_LOWLIGHT_DROP_MULT
                dmult = float(np.random.uniform(d_lo, d_hi))
                # quick & safe: do an extra light joint-drop pass
                V = j.shape[2]
                extra_drop_p = float(np.clip(AUG_JOINT_DROPOUT_PROB * (dmult - 1.0), 0.0, 0.25))
                if extra_drop_p > 0 and L > 1:
                    drop = (np.random.rand(L, V) < extra_drop_p)
                    if drop.any():
                        padJ = self.PAD_J[:, 0, 0, 0].reshape(3, 1, 1)
                        padB = self.PAD_B[:, 0, 0, 0].reshape(3, 1, 1)
                        padM = self.PAD_M[:, 0, 0, 0].reshape(3, 1, 1)
                        for t in range(L):
                            if drop[t].any():
                                idxs = np.where(drop[t])[0]
                                j[:, t, idxs, :] = padJ
                                b[:, t, idxs, :] = padB
                                m[:, t, idxs, :] = padM
                                if AUG_OCCLUDE_SET_CONF0:
                                    j[2, t, idxs, :] = 0.0
                                    b[2, t, idxs, :] = 0.0
                                    m[2, t, idxs, :] = 0.0

            # ---- B) Crowd aug (skeleton robustness) ----
            # B1) mis-association: swap a few joints at random times
            if (AUG_SWAP_NOISE_PROB > 0) and (np.random.rand() < AUG_SWAP_NOISE_PROB) and (L > 3):
                s_lo, s_hi = AUG_SWAP_JOINTS_RANGE
                num_swaps = int(np.random.randint(s_lo, s_hi + 1))
                j = self._swap_noise(j, L, num_swaps=num_swaps)
                b = self._swap_noise(b, L, num_swaps=num_swaps)
                m = self._swap_noise(m, L, num_swaps=num_swaps)

            # B2) partial body visibility (upper/lower missing)
            if (AUG_PARTIAL_BODY_PROB > 0) and (np.random.rand() < AUG_PARTIAL_BODY_PROB) and (L > 3):
                j, b, m = self._partial_body_mask(
                    j, b, m, L=L,
                    padJ=self.PAD_J, padB=self.PAD_B, padM=self.PAD_M,
                    mode=str(AUG_PARTIAL_MODE),
                    frac_range=tuple(AUG_PARTIAL_FRAC),
                    conf0=bool(AUG_OCCLUDE_SET_CONF0),
                )

            # 7) Joint dropout (simulate partial occlusion / missed keypoints)
            if AUG_JOINT_DROPOUT_PROB > 0 and L > 1:
                V = j.shape[2]
                drop = (np.random.rand(L, V) < AUG_JOINT_DROPOUT_PROB)
                if drop.any():
                    padJ = self.PAD_J[:, 0, 0, 0].reshape(3, 1, 1)
                    padB = self.PAD_B[:, 0, 0, 0].reshape(3, 1, 1)
                    padM = self.PAD_M[:, 0, 0, 0].reshape(3, 1, 1)
                    for t in range(L):
                        if drop[t].any():
                            idxs = np.where(drop[t])[0]
                            j[:, t, idxs, :] = padJ
                            b[:, t, idxs, :] = padB
                            m[:, t, idxs, :] = padM

            # 8) Frame dropout (simulate full pose failure for a frame)
            if AUG_FRAME_DROPOUT_PROB > 0 and L > 1:
                frame_drop = (np.random.rand(L) < AUG_FRAME_DROPOUT_PROB)
                if frame_drop.any():
                    for t in np.where(frame_drop)[0]:
                        j[:, t:t+1] = self.PAD_J
                        b[:, t:t+1] = self.PAD_B
                        m[:, t:t+1] = self.PAD_M

            # 9) Confidence attenuation (z channel) occasionally
            if AUG_SCORE_DROP_PROB > 0 and np.random.rand() < AUG_SCORE_DROP_PROB:
                lo, hi = AUG_SCORE_DROP_RANGE
                factor = float(np.random.uniform(lo, hi))
                j[2] *= factor
                b[2] *= factor
                m[2] *= factor

            j[2] = np.clip(j[2], -5.0, 5.0)

            # 10) Variable-T masking (existing)
            if self.variable_T_choices is not None:
                T_choice = int(np.random.choice(self.variable_T_choices))
                L = int(max(1, min(L, T_choice, self.desired_T)))
                if L < self.desired_T:
                    j[:, L:] = self.PAD_J
                    b[:, L:] = self.PAD_B
                    m[:, L:] = self.PAD_M

        return {
            "joint": torch.from_numpy(j),
            "bone": torch.from_numpy(b),
            "motion": torch.from_numpy(m),
            "length": torch.tensor(L, dtype=torch.long),
            "label": torch.tensor(y, dtype=torch.long),
            "q": torch.tensor(qv, dtype=torch.float32),
        }

# -------------------------
# loss (label smoothing)
# -------------------------
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        if self.smoothing <= 0:
            return nn.functional.cross_entropy(logits, target, reduction="none")
        n_class = logits.size(1)
        logp = nn.functional.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.smoothing / (n_class - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(true_dist * logp).sum(dim=1)

# -------------------------
# metrics helpers
# -------------------------
def topk_accuracy(logits, y, k=3):
    k = int(min(k, logits.size(1)))
    _, pred = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)
    correct = pred.eq(y.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()

def accuracy_top1(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

def confusion_matrix_counts(y_true, y_pred, num_class):
    cm = np.zeros((num_class, num_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_class and 0 <= p < num_class:
            cm[t, p] += 1
    return cm

def macro_f1_from_cm(cm):
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        f1s.append(float(f1))
    return float(np.mean(f1s))

def _auto_figsize(n_classes, base=7.5, per_class=0.35, max_size=18):
    s = min(max_size, max(base, base + per_class * n_classes))
    return (s, s)

def _auto_fontsizes(n_classes):
    tick_fs = int(np.clip(14 - 0.25 * n_classes, 6, 12))
    cell_fs = int(np.clip(14 - 0.35 * n_classes, 5, 12))
    title_fs = int(np.clip(16 - 0.20 * n_classes, 10, 16))
    return tick_fs, cell_fs, title_fs

def _shorten_label(s, max_len=18):
    s = str(s)
    return s if len(s) <= max_len else (s[:max_len-1] + "…")

def plot_confusion(cm, class_names, out_png, normalize=False, title="Confusion",
                   show_values=True, value_fmt=None):
    cm = np.asarray(cm)
    C = cm.shape[0]
    fig_w, fig_h = _auto_figsize(C)
    tick_fs, cell_fs, title_fs = _auto_fontsizes(C)

    if normalize:
        denom = cm.sum(axis=1, keepdims=True) + 1e-12
        cm_disp = (cm / denom) * 100.0
        default_fmt = "{:.1f}%"
        vmin, vmax = 0.0, 100.0
    else:
        cm_disp = cm.astype(np.float32)
        default_fmt = "{:d}"
        vmin, vmax = 0.0, float(cm_disp.max()) if cm_disp.size else 1.0

    if value_fmt is None:
        value_fmt = default_fmt

    labels = [_shorten_label(x) for x in class_names]

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(cm_disp, interpolation="nearest", vmin=vmin, vmax=vmax, cmap="Blues")
    plt.title(title, fontsize=title_fs, pad=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(C)
    plt.xticks(tick_marks, labels, rotation=45, ha="right", fontsize=tick_fs)
    plt.yticks(tick_marks, labels, fontsize=tick_fs)
    plt.ylabel("True", fontsize=tick_fs + 1)
    plt.xlabel("Predicted", fontsize=tick_fs + 1)

    if show_values and C <= 60:
        thresh = (cm_disp.max() * 0.60) if cm_disp.size else 0.0
        for i in range(C):
            for j in range(C):
                val = cm_disp[i, j]
                if normalize:
                    txt = value_fmt.format(val)
                else:
                    txt = value_fmt.format(int(cm[i, j]))
                plt.text(j, i, txt, ha="center", va="center",
                         fontsize=cell_fs,
                         color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_loss_curves(hist, out_png):
    train_loss = np.asarray(hist.get("train_loss", []), dtype=np.float32)
    val_loss   = np.asarray(hist.get("val_loss", []), dtype=np.float32)
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_loss, label="train_loss")
    if len(val_loss) == len(train_loss) and len(val_loss) > 0:
        plt.plot(epochs, val_loss, label="val_loss")

    best_epoch = hist.get("best_epoch", None)
    if best_epoch is not None and isinstance(best_epoch, (int, float)) and best_epoch > 0:
        plt.axvline(int(best_epoch), linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Train vs Val)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_acc_curves(hist, out_png, topk=3):
    train_top1 = np.asarray(hist.get("train_acc", []), dtype=np.float32) * 100.0
    val_top1   = np.asarray(hist.get("val_acc", []), dtype=np.float32) * 100.0
    train_topk = np.asarray(hist.get("train_topk", []), dtype=np.float32) * 100.0
    val_topk   = np.asarray(hist.get("val_topk", []), dtype=np.float32) * 100.0
    epochs = np.arange(1, len(train_top1) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_top1, label="train_top1 (%)")
    if len(val_top1) == len(train_top1) and len(val_top1) > 0:
        plt.plot(epochs, val_top1, label="val_top1 (%)")
    if len(train_topk) == len(train_top1) and len(train_topk) > 0:
        plt.plot(epochs, train_topk, label=f"train_top{topk} (%)")
    if len(val_topk) == len(train_top1) and len(val_topk) > 0:
        plt.plot(epochs, val_topk, label=f"val_top{topk} (%)")

    best_epoch = hist.get("best_epoch", None)
    if best_epoch is not None and isinstance(best_epoch, (int, float)) and best_epoch > 0:
        plt.axvline(int(best_epoch), linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")

    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curves (Train vs Val) — Top-1 & Top-{topk}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_overall_acc_curves(hist, out_png):
    train_acc = np.asarray(hist.get("train_acc", []), dtype=np.float32) * 100.0
    val_acc   = np.asarray(hist.get("val_acc", []), dtype=np.float32) * 100.0
    epochs = np.arange(1, len(train_acc) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_acc, label="train_overall_acc (Top-1) (%)")
    if len(val_acc) == len(train_acc) and len(val_acc) > 0:
        plt.plot(epochs, val_acc, label="val_overall_acc (Top-1) (%)")

    best_epoch = hist.get("best_epoch", None)
    if best_epoch is not None and isinstance(best_epoch, (int, float)) and best_epoch > 0:
        plt.axvline(int(best_epoch), linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")

    plt.ylim(0, 100)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Overall Accuracy Curves (Train vs Val) — Top-1")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def save_per_class_report(y_true, y_pred, class_names, out_txt, out_json=None):
    if SKLEARN_OK:
        from sklearn.metrics import classification_report
        rep_dict = classification_report(
            y_true, y_pred,
            target_names=[class_names[i] for i in range(len(class_names))],
            digits=4,
            output_dict=True,
            zero_division=0
        )
        rep_txt = classification_report(
            y_true, y_pred,
            target_names=[class_names[i] for i in range(len(class_names))],
            digits=4,
            zero_division=0
        )
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(rep_txt)

        if out_json is not None:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(rep_dict, f, indent=2)

        return rep_dict

    cm = confusion_matrix_counts(y_true, y_pred, len(class_names))
    rows = []
    for c in range(len(class_names)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = (2 * prec * rec) / (prec + rec + 1e-12)
        rows.append((class_names[c], prec, rec, f1, int(cm[c, :].sum())))

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Per-class Precision/Recall/F1 (fallback, sklearn not installed)\n")
        f.write(f"{'class':<28} {'prec':>8} {'rec':>8} {'f1':>8} {'support':>8}\n")
        for name, p, r, f1, sup in rows:
            f.write(f"{name:<28} {p:8.4f} {r:8.4f} {f1:8.4f} {sup:8d}\n")

    rep_dict = {
        "note": "fallback (no sklearn)",
        "per_class": [
            {"class": name, "precision": float(p), "recall": float(r), "f1": float(f1), "support": int(sup)}
            for name, p, r, f1, sup in rows
        ]
    }
    if out_json is not None:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)

    return rep_dict

# -------------------------
# loaders
# -------------------------
train_ds = SkeletonDatasetM1(
    TRAIN_PKL, mean=mean, std=std, augment=True,
    desired_T=T_WINDOW, variable_T_choices=TRAIN_T_CHOICES
)
val_ds = SkeletonDatasetM1(
    VAL_PKL, mean=mean, std=std, augment=False, desired_T=T_WINDOW
)
test_ds = SkeletonDatasetM1(
    TEST_PKL, mean=mean, std=std, augment=False, desired_T=T_WINDOW
)

def scan_dataset(ds, name="train", max_check=3000):
    bad = 0
    for i in range(min(len(ds), max_check)):
        x = ds[i]
        j = x["joint"].numpy()
        b = x["bone"].numpy()
        m = x["motion"].numpy()
        q = float(x["q"].item())
        if (not np.isfinite(j).all()) or (not np.isfinite(b).all()) or (not np.isfinite(m).all()) or (not np.isfinite(q)):
            print(f"[BAD {name}] idx={i} finite(j,b,m,q)=({np.isfinite(j).all()},{np.isfinite(b).all()},{np.isfinite(m).all()},{np.isfinite(q)}) q={q}")
            bad += 1
            if bad >= 10:
                break
    print(f"[SCAN] {name}: checked={min(len(ds), max_check)} bad={bad}")

scan_dataset(train_ds, "train")
scan_dataset(val_ds, "val")
scan_dataset(test_ds, "test")

print("Samples:", "train", len(train_ds), "| val", len(val_ds), "| test", len(test_ds))

sampler = None
shuffle = True
if USE_CLASS_AWARE_SAMPLING and len(train_ds) > 0:
    counts = Counter(train_ds.labels.tolist())
    inv = {c: 1.0 / max(1, counts[c]) for c in counts}
    w = []
    for i in range(len(train_ds)):
        y = int(train_ds.labels[i])
        base = inv.get(y, 1.0)
        qv = float(train_ds.q[i])
        if USE_Q_WEIGHTING_IN_SAMPLER:
            base *= (0.30 + 0.70 * float(np.clip(qv, 0.0, 1.0)))
        w.append(base)
    w = torch.tensor(w, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
    shuffle = False

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=shuffle,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

# -------------------------
# model
# -------------------------
A = build_adjacency(skeleton_layout="coco17")
model = MultiStreamCTRGCN(
    num_class=NUM_CLASS,
    num_joints=17,
    adjacency_matrix=A,
    dropout=0.3,
    in_channels=3,
    max_person=MAX_M,
).to(device).float()

criterion = LabelSmoothingCE(smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))

# -------------------------
# early stopping state
# -------------------------
def _is_better(new, best, mode="max", min_delta=0.0):
    if best is None:
        return True
    if mode == "max":
        return (new - best) > min_delta
    return (best - new) > min_delta

best_metric = None
best_epoch = -1
patience_left = int(EARLY_STOP_PATIENCE)

if EARLY_STOP_MONITOR == "val_loss":
    monitor_mode = "min"
else:
    monitor_mode = "max"

if len(val_ds) == 0:
    print("[WARN] val.pkl has 0 samples. Early stopping + best selection fallback to TRAIN acc.")
    EARLY_STOP_MONITOR = "train_acc"
    monitor_mode = "max"

# -------------------------
# eval
# -------------------------
def run_eval(loader):
    model.eval()
    loss_sum, n_sum = 0.0, 0
    acc1_sum, acck_sum = 0.0, 0.0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in loader:
            j = batch["joint"].to(device, non_blocking=True).float()
            b = batch["bone"].to(device, non_blocking=True).float()
            m = batch["motion"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True).long()
            L = batch["length"].to(device, non_blocking=True).long()
            qv = batch["q"].to(device, non_blocking=True).float()

            logits = model(j, b, m, L)
            per_sample = criterion(logits.float(), y)

            if USE_Q_WEIGHTING_IN_LOSS:
                wq = 0.25 + 0.75 * torch.clamp(qv, 0.0, 1.0)
                per_sample = per_sample * wq

            loss = per_sample.mean()
            acc1 = accuracy_top1(logits.float(), y)
            acck = topk_accuracy(logits.float(), y, k=TOPK)

            bs = y.size(0)
            loss_sum += float(loss.item()) * bs
            acc1_sum += float(acc1) * bs
            acck_sum += float(acck) * bs
            n_sum += bs

            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
            yt   = y.detach().cpu().numpy().astype(np.int64)
            y_true_all.append(yt)
            y_pred_all.append(pred)

    if n_sum == 0:
        return {
            "loss": 0.0, "acc1": 0.0, "acck": 0.0,
            "macro_f1": 0.0, "cm": np.zeros((NUM_CLASS, NUM_CLASS), dtype=np.int64), "n": 0
        }

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    cm = confusion_matrix_counts(y_true, y_pred, NUM_CLASS)
    mf1 = macro_f1_from_cm(cm)

    return {
        "loss": loss_sum / n_sum,
        "acc1": acc1_sum / n_sum,
        "acck": acck_sum / n_sum,
        "macro_f1": mf1,
        "cm": cm,
        "n": n_sum,
        "y_true": y_true,
        "y_pred": y_pred,
    }

# -------------------------
# train loop
# -------------------------
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_topk": [],
    "val_topk": [],
    "val_macro_f1": [],
    "lr": [],
    "best_epoch": None,
    "best_metric": None,
}

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [train]")

    loss_sum, acc1_sum, acck_sum, n_sum = 0.0, 0.0, 0.0, 0

    for batch in pbar:
        j = batch["joint"].to(device, non_blocking=True).float()
        b = batch["bone"].to(device, non_blocking=True).float()
        m = batch["motion"].to(device, non_blocking=True).float()
        y = batch["label"].to(device, non_blocking=True).long()
        L = batch["length"].to(device, non_blocking=True).long()
        qv = batch["q"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(j, b, m, L)
            per_sample = criterion(logits, y)

            if USE_Q_WEIGHTING_IN_LOSS:
                wq = 0.25 + 0.75 * torch.clamp(qv, 0.0, 1.0)
                per_sample = per_sample * wq

            loss = per_sample.mean()

        scaler.scale(loss).backward()
        if GRAD_CLIP_ENABLE:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            acc1 = accuracy_top1(logits.detach().float(), y)
            acck = topk_accuracy(logits.detach().float(), y, k=TOPK)

        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        acc1_sum += float(acc1) * bs
        acck_sum += float(acck) * bs
        n_sum += bs

        pbar.set_postfix(
            loss=loss_sum / max(1, n_sum),
            acc=acc1_sum / max(1, n_sum),
            topk=acck_sum / max(1, n_sum),
        )

    scheduler.step()
    lr_now = float(optimizer.param_groups[0]["lr"])

    train_loss = loss_sum / max(1, n_sum)
    train_acc  = acc1_sum / max(1, n_sum)
    train_topk = acck_sum / max(1, n_sum)

    val_res = run_eval(val_loader) if len(val_ds) > 0 else {
        "loss": train_loss, "acc1": train_acc, "acck": train_topk,
        "macro_f1": 0.0, "cm": np.zeros((NUM_CLASS, NUM_CLASS), dtype=np.int64), "n": 0
    }

    val_loss = float(val_res["loss"])
    val_acc  = float(val_res["acc1"])
    val_topk = float(val_res["acck"])
    val_f1   = float(val_res["macro_f1"])

    print(f"[E{epoch:03d}] "
          f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% top{TOPK}={train_topk*100:.2f}% | "
          f"val loss={val_loss:.4f} acc={val_acc*100:.2f}% top{TOPK}={val_topk*100:.2f}% f1(macro)={val_f1*100:.2f}% | "
          f"lr={lr_now:.6f}")

    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["train_acc"].append(float(train_acc))
    history["val_acc"].append(float(val_acc))
    history["train_topk"].append(float(train_topk))
    history["val_topk"].append(float(val_topk))
    history["val_macro_f1"].append(float(val_f1))
    history["lr"].append(lr_now)

    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, LAST_CKPT)

    if EARLY_STOP_MONITOR == "val_loss":
        current_metric = val_loss
        metric_name = "val_loss"
        mode = "min"
    elif EARLY_STOP_MONITOR == "val_acc":
        current_metric = val_acc
        metric_name = "val_acc"
        mode = "max"
    elif EARLY_STOP_MONITOR == "val_macro_f1":
        current_metric = val_f1
        metric_name = "val_macro_f1"
        mode = "max"
    elif EARLY_STOP_MONITOR == "train_acc":
        current_metric = train_acc
        metric_name = "train_acc"
        mode = "max"
    else:
        current_metric = val_acc
        metric_name = "val_acc"
        mode = "max"

    if _is_better(current_metric, best_metric, mode=mode, min_delta=EARLY_STOP_MIN_DELTA):
        best_metric = float(current_metric)
        best_epoch = int(epoch)
        patience_left = int(EARLY_STOP_PATIENCE)

        bundle = {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "T_WINDOW": int(T_WINDOW),
            "MAX_M": int(MAX_M),
            "num_class": int(NUM_CLASS),
            "layout": "coco17",
            "class_names": CLASS_NAMES,
            "best_epoch": int(best_epoch),
            "best_metric_name": metric_name,
            "best_metric_value": float(best_metric),
        }
        torch.save(bundle, BEST_BUNDLE)
        print(f"Saved BEST bundle: {BEST_BUNDLE} ({metric_name}={best_metric:.6f} @ epoch {best_epoch})")

        try:
            history["best_epoch"] = best_epoch
            history["best_metric"] = best_metric
            with open(METRICS_JSON, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            plot_loss_curves(history, LOSS_CURVE_PNG)
            plot_acc_curves(history, ACC_CURVE_PNG, topk=TOPK)
            plot_overall_acc_curves(history, ACC_OVERALL_CURVE_PNG)
        except Exception as e:
            print("[WARN] could not save plots/metrics json:", e)

    else:
        if EARLY_STOP_ENABLE:
            patience_left -= 1
            print(f"[EarlyStop] no improvement on {metric_name}. patience_left={patience_left}/{EARLY_STOP_PATIENCE}")
            if patience_left <= 0:
                print(f"🛑 Early stopping triggered at epoch {epoch}. Best at epoch {best_epoch} ({metric_name}={best_metric:.6f}).")
                break

history["best_epoch"] = best_epoch
history["best_metric"] = best_metric
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

try:
    plot_loss_curves(history, LOSS_CURVE_PNG)
    plot_acc_curves(history, ACC_CURVE_PNG, topk=TOPK)
    plot_overall_acc_curves(history, ACC_OVERALL_CURVE_PNG)
    print("Saved plots:")
    print(" -", LOSS_CURVE_PNG)
    print(" -", ACC_CURVE_PNG)
except Exception as e:
    print("[WARN] final plotting failed:", e)

elapsed = time.time() - start_time
print(f"Training done. Best epoch={best_epoch}, best_metric={best_metric}. Elapsed={elapsed/60.0:.1f} min")
print("Best bundle:", BEST_BUNDLE)

# -------------------------
# test + confusion matrix plots (from BEST bundle)
# -------------------------
if os.path.exists(BEST_BUNDLE):
    ck = torch.load(BEST_BUNDLE, map_location="cpu")
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()

test_res = run_eval(test_loader) if len(test_ds) > 0 else None
if test_res is not None:
    print(f"[TEST] loss={test_res['loss']:.4f} acc={test_res['acc1']*100:.2f}% top{TOPK}={test_res['acck']*100:.2f}% f1(macro)={test_res['macro_f1']*100:.2f}%")

    cm = test_res["cm"]
    labels_order = [CLASS_NAMES[i] for i in range(NUM_CLASS)]

    try:
        plot_confusion(
            cm, labels_order, CM_COUNTS_PNG, normalize=False,
            title=f"Confusion (counts) | Test Top1={test_res['acc1']*100:.2f}% | MacroF1={test_res['macro_f1']*100:.2f}%"
        )
        plot_confusion(
            cm, labels_order, CM_NORM_PNG, normalize=True,
            title=f"Confusion (row-normalized %) | Test Top1={test_res['acc1']*100:.2f}% | MacroF1={test_res['macro_f1']*100:.2f}%"
        )
        print("Saved confusion matrix plots:")
        print(" -", CM_COUNTS_PNG)
        print(" -", CM_NORM_PNG)
    except Exception as e:
        print("[WARN] confusion plotting failed:", e)

    try:
        rep = save_per_class_report(
            test_res["y_true"], test_res["y_pred"],
            class_names=CLASS_NAMES,
            out_txt=TEST_REPORT_TXT,
            out_json=TEST_REPORT_JSON
        )
        print("Saved per-class report:")
        print(" -", TEST_REPORT_TXT)
        print(" -", TEST_REPORT_JSON)
    except Exception as e:
        print("[WARN] saving per-class report failed:", e)

    try:
        summary = {
            "Top1_Accuracy": float(test_res["acc1"]),
            "Top3_Accuracy": float(test_res["acck"]),
            "Macro_F1": float(test_res["macro_f1"]),
            "Test_Loss": float(test_res["loss"]),
            "TOPK": int(TOPK),
            "num_class": int(NUM_CLASS),
            "class_names": CLASS_NAMES,
        }
        with open(TEST_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("Saved test summary:")
        print(" -", TEST_SUMMARY_JSON)
    except Exception as e:
        print("[WARN] saving test summary failed:", e)

    if SKLEARN_OK:
        try:
            print("\n[Classification Report - TEST]")
            print(classification_report(
                test_res["y_true"], test_res["y_pred"],
                target_names=[CLASS_NAMES[i] for i in range(NUM_CLASS)],
                digits=4,
                zero_division=0
            ))
        except Exception as e:
            print("[WARN] classification_report failed:", e)
else:
    print("[TEST] skipped (no test samples).")
