# detection_core.py  (Python 3.9 compatible)  [RTSP-ONLY]
# 1 Sec Notif Max
# =========================================================
# MultiCamDetectionRunner (3 cams) - RTSP focus
# - Uses LatestFrameGrabber to keep only the most recent frame (reduces buffering latency)
# - Opens RTSP via FFMPEG with low-latency options
# - Runs YOLO+RTMPose+CTR-GCN on smaller inference frame for speed
#
# NOTE:
# This version removes DroidCam HTTP special handling and focuses on RTSP streams. So basically it is just for cctv now.
# =========================================================

import os
import time
import json
import base64
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Optional, Union, List, Dict, Tuple
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO
from mmengine.registry import init_default_scope
from mmpose.apis import init_model as init_mmpose_model, inference_topdown
from ctrgcn_model import MultiStreamCTRGCN, build_adjacency, COCO17_PARENT
torch.set_default_dtype(torch.float32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS  = os.path.join(BASE_DIR, "assets")
YOLO_WEIGHTS = os.path.join(ASSETS, "yolov8n.pt")

CLASS_MAP_PATH   = os.path.join(ASSETS, "class_map.json")
BUNDLE_CKPT_PATH = os.path.join(ASSETS, "ctrgcn_bundle.pth")

RTM_BODY_CFG  = os.path.join(ASSETS, "rtmpose_cfg.py")
RTM_BODY_CKPT = os.path.join(ASSETS, "rtmpose.pth")

# -------------------------
# Runtime config
# -------------------------
FRAME_W, FRAME_H = 1280 , 720
CONF_TH = 0.40

MAX_TRACKS = 6
TRACK_TTL_FRAMES = 30
TRACK_IOU_TH = 0.30

KP_CONF_TH = 0.10
BBOX_EXPAND = 1.65

STRIDE_PRED = 2
PRED_SMOOTH = 3

IDLE_ENABLE = True
IDLE_MIN_FRAMES = 18
NO_ACTION_MOTION_THRESH = 1

# ---- NOTIFICATIONS (Web UI) ----
# Only raise notifications for these actions at/above threshold.
NOTIF_LABELS = {"Punching", "Kicking"}
NOTIF_MIN_CONF = 0.85
NOTIF_COOLDOWN_SEC = 1.0   # 1 second cooldown per (cam,label)

def violent_streak_needed(label: str, fps: float) -> int:
    label = str(label or "").strip().lower()

    # lower consistency targets so punch/kick triggers earlier
    if label == "punching":
        sec = 0.25
    elif label == "kicking":
        sec = 0.35
    else:
        sec = 0.30

    if fps <= 0:
        return 3 if label == "punching" else 4

    return max(1, int(round(float(fps) * sec)))

# Put your RTSP sources here (or set them via set_source)
DEFAULT_SOURCES = [
    None,
    None,
    None
]

# ---- display stream knobs ----
STREAM_W, STREAM_H = 960, 540
STREAM_JPEG_QUALITY = 60

# ---- SPEED: inference resolution ----
INFER_W, INFER_H = 960, 540

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = (device.type == "cuda")
print("Device:", device, "AMP:", use_amp)

# Optional: RTSP over TCP & low latency (FFMPEG) for OpenCV FFMPEG backend
# (This helps but some cameras still buffer ~0.5-2s internally.)
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

# -------------------------
# COCO17 indices
# -------------------------
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16
TORSO_JOINTS = [L_HIP, R_HIP, L_SHO, R_SHO]

COCO_EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (5,11),(6,12), (11,12),
    (0,5),(0,6), (0,1),(0,2),(1,3),(2,4)
]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def expand_bbox(bb, W, H, scale=1.25):
    x1,y1,x2,y2 = bb.astype(np.float32)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = max(0, cx-w/2), max(0, cy-h/2)
    nx2, ny2 = min(W-1, cx+w/2), min(H-1, cy+h/2)
    return np.array([nx1, ny1, nx2, ny2], np.float32)

def draw_skeleton(img, kpt17_3, color, conf_th=0.2):
    k = kpt17_3
    for a,b in COCO_EDGES:
        if k[a,2] >= conf_th and k[b,2] >= conf_th:
            cv2.line(img, (int(k[a,0]), int(k[a,1])), (int(k[b,0]), int(k[b,1])), color, 2)
    for j in range(17):
        if k[j,2] >= conf_th:
            cv2.circle(img, (int(k[j,0]), int(k[j,1])), 3, color, -1)

def count_valid_frames(seq_T17_3, kp_conf_th=0.20):
    sc = seq_T17_3[:, :, 2]
    torso = sc[:, TORSO_JOINTS]
    return int((torso.mean(axis=1) >= kp_conf_th).sum())

def motion_level_from_seq(seq_T17_3):
    if seq_T17_3.shape[0] < 2:
        return 0.0
    xy = seq_T17_3[:, TORSO_JOINTS, 0:2].mean(axis=1)
    v = xy[1:] - xy[:-1]
    return float(np.linalg.norm(v, axis=1).mean())

def compute_bone_motion_from_joint_CTV(joint_CTV):
    bone = np.zeros_like(joint_CTV, dtype=np.float32)
    motion = np.zeros_like(joint_CTV, dtype=np.float32)
    for v, p in enumerate(COCO17_PARENT):
        if int(p) >= 0:
            bone[0:2, :, v] = joint_CTV[0:2, :, v] - joint_CTV[0:2, :, int(p)]
    bone[2, :, :] = 0.0
    motion[0:2, 1:, :] = joint_CTV[0:2, 1:, :] - joint_CTV[0:2, :-1, :]
    motion[2, :, :] = 0.0
    return bone, motion

def center_scale_single(seq_T17_3, kp_conf_th=0.20):
    out = seq_T17_3.copy().astype(np.float32)
    T = out.shape[0]
    for t in range(T):
        sc_torso = out[t, TORSO_JOINTS, 2].mean()
        if sc_torso < kp_conf_th:
            center = np.array([0.0, 0.0], np.float32)
        else:
            center = 0.5 * (out[t, L_HIP, 0:2] + out[t, R_HIP, 0:2])
        xy = out[t, :, 0:2] - center[None, :]
        r = np.linalg.norm(xy, axis=1)
        s = float(np.max(r))
        if s < 1e-6:
            s = 1.0
        out[t, :, 0:2] = xy / s
        out[t, :, 2] = np.clip(out[t, :, 2], 0.0, 1.0)
    return out

def infer_pose_batch(pose_model, frame_bgr, bboxes_xyxy):
    if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
        return []
    bboxes_xyxy = np.asarray(bboxes_xyxy, dtype=np.float32)
    try:
        with torch.no_grad():
            return inference_topdown(pose_model, frame_bgr, bboxes_xyxy)
    except Exception:
        persons = [{"bbox": bb.astype(np.float32), "bbox_score": 1.0} for bb in bboxes_xyxy]
        with torch.no_grad():
            return inference_topdown(pose_model, frame_bgr, persons)

def jpeg_b64(frame_bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _coerce_meanstd_ctvm(x, C=3, M=1):
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 4 and x.shape[0] == C:
        if x.shape[2] == 17:
            x = x.mean(axis=2, keepdims=True)
        if x.shape[3] == 1 and M > 1:
            x = np.repeat(x, M, axis=3)
        return x.astype(np.float32)
    raise ValueError("Unsupported mean/std shape: {}".format(x.shape))

class LatestFrameGrabber:
    """
    Latest-frame grabber for streaming sources.
    - reads continuously and keeps only the most recent frame (drops old frames)
    - reduces perceived latency from buffering
    """
    def __init__(self, cap: 'cv2.VideoCapture', name: str = 'cam', reopen_src=None, reopen_backend=None):
        self.cap = cap
        self.name = name
        self.reopen_src = reopen_src
        self.reopen_backend = reopen_backend
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._ts = 0.0
        self._ok = False

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            ts = time.time()
            if not ok or frame is None:
                # mark as not ok
                with self._lock:
                    self._ok = False

                # try reconnect (release and reopen)
                try:
                    self.cap.release()
                except Exception:
                    pass

                time.sleep(1.0)

                if self.reopen_src is None:
                    continue

                try:
                    if self.reopen_backend is None:
                        self.cap = cv2.VideoCapture(self.reopen_src)
                    else:
                        self.cap = cv2.VideoCapture(self.reopen_src, self.reopen_backend)
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                except Exception:
                    pass

                continue
            with self._lock:
                self._ok = True
                self._frame = frame
                self._ts = ts

    def get(self):
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None, 0.0
            return True, self._frame, self._ts

@dataclass
class Track:
    tid: int
    bbox: np.ndarray
    last_seen: int
    score: float = 0.0
    kf_mean: Optional[np.ndarray] = None
    kf_cov: Optional[np.ndarray] = None
    pose3: deque = field(default_factory=lambda: deque(maxlen=256))
    pred_hist: deque = field(default_factory=lambda: deque(maxlen=PRED_SMOOTH))
    last_pred: Optional[Tuple[str, float]] = None
    last_pred_frame: int = -999999
    violent_streak: int = 0
    last_violent_label: str = ""

# -------------------------
# BYTETrack (lightweight, pure numpy)
# -------------------------
def xyxy_to_xyah(bb):
    x1, y1, x2, y2 = bb.astype(np.float32)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_xyxy(xyah):
    cx, cy, a, h = xyah.astype(np.float32)
    h = max(1.0, float(h))
    w = max(1.0, float(a) * h)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)

class KalmanFilterXYAH:
    """
    SORT/ByteTrack-style Kalman filter:
      state: (cx, cy, a, h, vx, vy, va, vh)
      meas : (cx, cy, a, h)
    """
    def __init__(self):
        self._ndim = 4
        self._dim_x = 8

        self._motion_mat = np.eye(self._dim_x, dtype=np.float32)
        for i in range(self._ndim):
            self._motion_mat[i, self._ndim + i] = 1.0

        self._update_mat = np.eye(self._ndim, self._dim_x, dtype=np.float32)

        # Noise weights (typical SORT defaults)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement_xyah):
        mean = np.zeros((self._dim_x,), dtype=np.float32)
        mean[:4] = measurement_xyah.astype(np.float32)

        std = np.zeros((self._dim_x,), dtype=np.float32)
        h = float(max(1.0, measurement_xyah[3]))
        std[:4] = 2.0 * self._std_weight_position * h
        std[4:] = 10.0 * self._std_weight_velocity * h
        cov = np.diag(std * std).astype(np.float32)
        return mean, cov

    def predict(self, mean, cov):
        mean = self._motion_mat @ mean
        h = float(max(1.0, mean[3]))

        std_pos = self._std_weight_position * h
        std_vel = self._std_weight_velocity * h

        q = np.zeros((self._dim_x,), dtype=np.float32)
        q[:4] = std_pos
        q[4:] = std_vel
        Q = np.diag(q * q).astype(np.float32)

        cov = (self._motion_mat @ cov @ self._motion_mat.T) + Q
        return mean, cov

    def project(self, mean, cov):
        h = float(max(1.0, mean[3]))
        std = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ], dtype=np.float32)
        R = np.diag(std * std).astype(np.float32)

        mean_z = self._update_mat @ mean
        cov_z = self._update_mat @ cov @ self._update_mat.T + R
        return mean_z, cov_z

    def update(self, mean, cov, measurement_xyah):
        mean_z, cov_z = self.project(mean, cov)
        S_inv = np.linalg.inv(cov_z)
        K = cov @ self._update_mat.T @ S_inv

        innovation = measurement_xyah.astype(np.float32) - mean_z
        mean = mean + (K @ innovation)
        cov = cov - (K @ cov_z @ K.T)
        return mean, cov

# Tracking thresholds (ByteTrack uses two-stage matching: high & low score)
TRACK_HIGH_TH = 0.60   # high-confidence association
TRACK_LOW_TH  = 0.10   # low-confidence "recovery" association
NEW_TRACK_TH  = 0.60   # start new track if score >= this

class ByteTracker:
    """
    Minimal ByteTrack-like tracker (no external deps).
    - per-frame KF predict
    - 2-stage association (high then low confidence detections)
    - greedy IoU matching (works well for MAX_TRACKS=1; still OK for small counts)
    """
    def __init__(self, iou_th=0.30, ttl=30,
                 high_th=TRACK_HIGH_TH, low_th=TRACK_LOW_TH, new_th=NEW_TRACK_TH):
        self.iou_th = float(iou_th)
        self.ttl = int(ttl)
        self.high_th = float(high_th)
        self.low_th = float(low_th)
        self.new_th = float(new_th)

        self.next_id = 0
        self.tracks = {}  # type: Dict[int, Track]
        self.kf = KalmanFilterXYAH()

    def _predict_all(self):
        for tr in self.tracks.values():
            if tr.kf_mean is not None and tr.kf_cov is not None:
                tr.kf_mean, tr.kf_cov = self.kf.predict(tr.kf_mean, tr.kf_cov)
                tr.bbox = xyah_to_xyxy(tr.kf_mean[:4])

    def _match_greedy(self, track_ids, dets):
        """
        dets: list of (bbox_xyxy, score, det_index)
        returns matches: list of (tid, det_index)
        """
        used_t = set()
        used_d = set()
        matches = []

        pairs = []
        for bb, sc, di in dets:
            for tid in track_ids:
                tr = self.tracks.get(tid)
                if tr is None:
                    continue
                i = iou_xyxy(bb, tr.bbox)
                pairs.append((i, tid, di, bb, sc))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_v, tid, di, bb, sc in pairs:
            if tid in used_t or di in used_d:
                continue
            if iou_v < self.iou_th:
                continue
            used_t.add(tid)
            used_d.add(di)
            matches.append((tid, di, bb, sc))
        return matches, used_t, used_d

    def update(self, dets_xyxy_score, frame_idx):
        """
        dets_xyxy_score: list of (bbox_xyxy, score) in preferred order (already top-k).
        returns: list of (tid, bbox_xyxy) aligned with input detection order for this frame.
        """
        # Remove dead
        dead = [tid for tid, tr in self.tracks.items() if (frame_idx - tr.last_seen) > self.ttl]
        for tid in dead:
            del self.tracks[tid]

        # Predict all tracks
        self._predict_all()

        dets_xyxy_score = dets_xyxy_score or []
        dets_all = [(np.asarray(bb, np.float32), float(sc), di) for di, (bb, sc) in enumerate(dets_xyxy_score)]

        high = [(bb, sc, di) for (bb, sc, di) in dets_all if sc >= self.high_th]
        low  = [(bb, sc, di) for (bb, sc, di) in dets_all if (self.low_th <= sc < self.high_th)]

        track_ids = list(self.tracks.keys())
        assigned = {}  # det_index -> tid
        used_t = set()

        # 1) Match high score detections
        matches1, used_t1, used_d1 = self._match_greedy(track_ids, high)
        for tid, di, bb, sc in matches1:
            tr = self.tracks[tid]
            tr.last_seen = frame_idx
            tr.score = float(sc)
            meas = xyxy_to_xyah(bb)
            if tr.kf_mean is None or tr.kf_cov is None:
                tr.kf_mean, tr.kf_cov = self.kf.initiate(meas)
            else:
                tr.kf_mean, tr.kf_cov = self.kf.update(tr.kf_mean, tr.kf_cov, meas)
            tr.bbox = xyah_to_xyxy(tr.kf_mean[:4])
            assigned[di] = tid
        used_t |= used_t1

        # 2) Match low score detections to recover unmatched tracks
        unmatched_t = [tid for tid in track_ids if tid not in used_t]
        if unmatched_t and low:
            low_unassigned = [(bb, sc, di) for (bb, sc, di) in low if di not in assigned]
            matches2, used_t2, used_d2 = self._match_greedy(unmatched_t, low_unassigned)
            for tid, di, bb, sc in matches2:
                tr = self.tracks[tid]
                tr.last_seen = frame_idx
                tr.score = float(sc)
                meas = xyxy_to_xyah(bb)
                if tr.kf_mean is None or tr.kf_cov is None:
                    tr.kf_mean, tr.kf_cov = self.kf.initiate(meas)
                else:
                    tr.kf_mean, tr.kf_cov = self.kf.update(tr.kf_mean, tr.kf_cov, meas)
                tr.bbox = xyah_to_xyxy(tr.kf_mean[:4])
                assigned[di] = tid
            used_t |= used_t2

        # 3) Create new tracks from unmatched high detections
        for bb, sc, di in high:
            if di in assigned:
                continue
            if sc < self.new_th:
                continue
            tid = self.next_id
            self.next_id += 1
            meas = xyxy_to_xyah(bb)
            mean, cov = self.kf.initiate(meas)
            self.tracks[tid] = Track(
                tid=tid,
                bbox=xyah_to_xyxy(mean[:4]),
                last_seen=frame_idx,
                score=float(sc),
                kf_mean=mean,
                kf_cov=cov
            )
            assigned[di] = tid

        # Return assignments aligned with the detection list order
        out = []
        for di, (bb, sc) in enumerate(dets_xyxy_score):
            if di in assigned:
                out.append((int(assigned[di]), np.asarray(bb, np.float32)))
        return out

@dataclass
class CamState:
    source: Any = None
    source_kind: str = 'none'
    pending_source: Any = None
    pending_dirty: bool = False
    cap: Optional[cv2.VideoCapture] = None
    grabber: Optional[LatestFrameGrabber] = None
    connected: bool = False
    fps: float = 0.0
    frame_b64: Optional[str] = None
    tracks_out: list = field(default_factory=list)
    last_ts_ms: int = 0  # timestamp (ms) of latest processed frame for this cam
    ended: bool = False

    tracker: ByteTracker = field(default_factory=lambda: ByteTracker(TRACK_IOU_TH, TRACK_TTL_FRAMES))
    current_pose: dict = field(default_factory=dict)

    frame_idx: int = 0
    fps_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    prev_t: float = field(default_factory=time.time)

class MultiCamDetectionRunner:
    def __init__(self, num_cams=3, recorder=None):
        self.num_cams = int(num_cams)
        self.recorder = recorder

        self.lock = threading.Lock()
        self._thread = None
        self._stop_evt = threading.Event()
        self._packet_id = 0

        self.cams = [CamState(source=(DEFAULT_SOURCES[i] if i < len(DEFAULT_SOURCES) else None)) for i in range(self.num_cams)]

        # (cam_idx, label_lower) -> last_emit_time_seconds
        self._notif_last: Dict[Tuple[int, str], float] = {}

        print("Loading bundle:", BUNDLE_CKPT_PATH)
        bundle = torch.load(BUNDLE_CKPT_PATH, map_location="cpu")
        self.STATE_DICT = bundle.get("state_dict", bundle)
        self.MEAN = bundle.get("mean", None)
        self.STD  = bundle.get("std", None)

        self.T_WINDOW = int(bundle.get("T_WINDOW", 36))
        self.MAX_M    = int(bundle.get("MAX_M", 1))
        self.NUM_CLASS = int(bundle.get("num_class", 0) or 0)

        self.CLASS_NAMES = None
        class_map_source = "none"

        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
                self.CLASS_NAMES = {int(k): str(v) for k, v in json.load(f).items()}
            class_map_source = f"json:{CLASS_MAP_PATH}"

        if self.CLASS_NAMES is None and isinstance(bundle, dict):
            for key in ["class_map", "class_names"]:
                if key in bundle and isinstance(bundle[key], dict):
                    self.CLASS_NAMES = {int(k): str(v) for k, v in bundle[key].items()}
                    class_map_source = f"bundle:{key}"
                    break

        if self.CLASS_NAMES is None:
            self.CLASS_NAMES = {i: f"C{i}" for i in range(self.NUM_CLASS if self.NUM_CLASS else 1)}
            class_map_source = "fallback:C*"

        if not self.NUM_CLASS:
            self.NUM_CLASS = len(self.CLASS_NAMES)

        print("[CLASS_MAP] Using", class_map_source)
        print("[CLASS_MAP] =", self.CLASS_NAMES)

        if (self.MEAN is None) or (self.STD is None):
            self.MEAN_J = np.zeros((3,1,1,1), np.float32); self.STD_J = np.ones((3,1,1,1), np.float32)
            self.MEAN_B = np.zeros((3,1,1,1), np.float32); self.STD_B = np.ones((3,1,1,1), np.float32)
            self.MEAN_M = np.zeros((3,1,1,1), np.float32); self.STD_M = np.ones((3,1,1,1), np.float32)
        else:
            self.MEAN_J = _coerce_meanstd_ctvm(self.MEAN.get("joint"), 3, 1)
            self.MEAN_B = _coerce_meanstd_ctvm(self.MEAN.get("bone"), 3, 1)
            self.MEAN_M = _coerce_meanstd_ctvm(self.MEAN.get("motion"), 3, 1)
            self.STD_J  = np.maximum(_coerce_meanstd_ctvm(self.STD.get("joint"), 3, 1), 1e-2)
            self.STD_B  = np.maximum(_coerce_meanstd_ctvm(self.STD.get("bone"), 3, 1), 1e-2)
            self.STD_M  = np.maximum(_coerce_meanstd_ctvm(self.STD.get("motion"), 3, 1), 1e-2)

        self.PAD_J = (0.0 - self.MEAN_J) / (self.STD_J + 1e-6)
        self.PAD_B = (0.0 - self.MEAN_B) / (self.STD_B + 1e-6)
        self.PAD_M = (0.0 - self.MEAN_M) / (self.STD_M + 1e-6)

        print("Loading YOLO...")
        self.yolo = YOLO(YOLO_WEIGHTS)

        print("Loading RTMPose...")
        init_default_scope("mmpose")
        self.pose_model = init_mmpose_model(
            RTM_BODY_CFG,
            RTM_BODY_CKPT,
            device=("cuda:0" if device.type == "cuda" else "cpu")
        ).eval()

        print("Loading CTR-GCN...")
        A = build_adjacency(skeleton_layout="coco17")
        self.model = MultiStreamCTRGCN(
            num_class=self.NUM_CLASS,
            num_joints=17,
            adjacency_matrix=A,
            dropout=0.3,
            in_channels=3,
            max_person=1,
        ).to(device).float().eval()
        self.model.load_state_dict(self.STATE_DICT, strict=False)

        print("Bundle resolved: T_WINDOW={} MAX_M={} num_class={}".format(self.T_WINDOW, self.MAX_M, self.NUM_CLASS))
        print("MultiCamDetectionRunner ready. [RTSP-ONLY]")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        for c in self.cams:
            try:
                if c.grabber is not None:
                    c.grabber.stop()
                    c.grabber = None
            except Exception:
                pass
            try:
                if c.cap is not None:
                    c.cap.release()
            except Exception:
                pass
            c.cap = None
            c.connected = False

    def set_source(self, cam_idx, source):
        cam_idx = int(cam_idx)
        if cam_idx < 0 or cam_idx >= self.num_cams:
            return {"ok": False, "error": "bad_cam_index"}

        if isinstance(source, str):
            s = source.strip()
            if s == "":
                source = None
            elif s.isdigit():
                source = int(s)  # you can remove this if you want ONLY RTSP URLs
            else:
                source = s

        with self.lock:
            self.cams[cam_idx].pending_source = source
            self.cams[cam_idx].pending_dirty = True
        return {"ok": True, "cam": cam_idx, "source": source}

    def get_status(self):
        with self.lock:
            out = {"ok": True}
            for i, c in enumerate(self.cams):
                out["cam{}".format(i)] = {
                    "connected": bool(c.connected),
                    "source": c.source,
                    "fps": float(c.fps),
                }
            return out

    def get_latest(self):
        with self.lock:
            pkt = {"id": int(self._packet_id)}
            fps_vals = [c.fps for c in self.cams if c.connected]
            pkt["fps"] = float(sum(fps_vals) / len(fps_vals)) if fps_vals else 0.0
            for i, c in enumerate(self.cams):
                pkt["frame{}".format(i)] = c.frame_b64
                pkt["cam{}".format(i)] = {
                    "tracks": list(c.tracks_out),
                    "connected": bool(c.connected),
                    "source": c.source,
                    "fps": float(c.fps),
                }

            # -------------------------
            # Notifications (Punching/Kicking >= threshold)
            # -------------------------
            # -------------------------
            # Notifications (same trigger as recording)
            # -------------------------
            notifs = []
            now_s = time.time()

            for cam_i, c in enumerate(self.cams):
                if not c.tracks_out:
                    continue

                # pick the most confident violent track that is also ready for recording/notification
                try:
                    violent_tracks = [
                        t for t in c.tracks_out
                        if str(t.get("label", "")).strip() in NOTIF_LABELS
                        and float(t.get("conf", 0.0) or 0.0) >= NOTIF_MIN_CONF
                        and int(t.get("violent_streak", 0)) >= int(t.get("violent_needed", 999999))
                    ]

                    if not violent_tracks:
                        continue

                    top = max(violent_tracks, key=lambda t: float(t.get("conf", 0.0)))

                except Exception:
                    continue

                label = str(top.get("label", "")).strip()
                confv = float(top.get("conf", 0.0) or 0.0)

                key = (cam_i, label.lower())
                last_emit = float(self._notif_last.get(key, 0.0))
                if (now_s - last_emit) < NOTIF_COOLDOWN_SEC:
                    continue

                self._notif_last[key] = now_s

                pct = int(round(confv * 100.0))
                ts_ms = int(c.last_ts_ms or int(now_s * 1000))
                tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000.0))

                notifs.append({
                    "type": "violence",
                    "cam": f"cam{cam_i + 1}",
                    "camIndex": int(cam_i),
                    "label": label,
                    "conf": float(confv),
                    "title": "Violence Detected",
                    "desc": f"{label} detected with {pct}% confidence at cam {cam_i + 1} on time: {tstr}",
                    "ts": ts_ms
                })

            pkt["notifications"] = notifs
            return pkt

    def _rtsp_url_with_opts(self, url: str) -> str:
        # Inject low-latency ffmpeg opts in URL when possible
        # (Some OpenCV builds respect these; if not, environment var still helps.)
        if "?" in url:
            return url
        return url + "?rtsp_transport=tcp&fflags=nobuffer&flags=low_delay&max_delay=0"

    def _source_kind(self, src):
        if src is None:
            return "none"
        if isinstance(src, int):
            return "device"
        s = str(src).strip()
        if s.lower().startswith("rtsp://"):
            return "rtsp"
        if os.path.isfile(s):
            return "file"
        return "other"

    def _open_cap(self, src):
        if src is None:
            return None

        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
            return cap

        src_str = str(src).strip()
        kind = self._source_kind(src_str)

        if kind == "file":
            cap = cv2.VideoCapture(src_str)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap

        if kind != "rtsp":
            print("[WARN] Unsupported source rejected:", src_str)
            return None

        url2 = self._rtsp_url_with_opts(src_str)
        cap = cv2.VideoCapture(url2, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        return cap

    def _apply_pending_source(self, cam):
        auto_reopen = (
            cam.cap is None
            and cam.source is not None
            and getattr(cam, "source_kind", "") != "file"
            and not getattr(cam, "ended", False)
        )

        do_apply = cam.pending_dirty or auto_reopen
        if not do_apply:
            return

        pending = cam.pending_source if cam.pending_dirty else cam.source
        cam.pending_source = None
        cam.pending_dirty = False

        cam.source = pending
        cam.source_kind = self._source_kind(cam.source)

        try:
            if cam.grabber is not None:
                cam.grabber.stop()
                cam.grabber = None
        except Exception:
            pass

        try:
            if cam.cap is not None:
                cam.cap.release()
        except Exception:
            pass

        cam.cap = None
        cam.connected = False
        cam.fps = 0.0
        cam.frame_b64 = None
        cam.tracks_out = []
        cam.last_ts_ms = 0
        cam.ended = False

        cam.tracker = ByteTracker(TRACK_IOU_TH, TRACK_TTL_FRAMES)
        cam.current_pose = {}
        cam.frame_idx = 0
        cam.fps_hist.clear()
        cam.prev_t = time.time()

        cam.cap = self._open_cap(cam.source)
        cam.connected = bool(cam.cap is not None and cam.cap.isOpened())

        if cam.connected and cam.source_kind == 'rtsp':
            try:
                cam.grabber = LatestFrameGrabber(cam.cap, name='cam', reopen_src=cam.source, reopen_backend=cv2.CAP_FFMPEG).start()
            except Exception:
                cam.grabber = None

    def _capture_latest_frame(self, cam):
        if cam.cap is None or not cam.cap.isOpened():
            return False, None, 0.0

        if cam.source_kind == 'rtsp':
            if cam.grabber is None:
                return False, None, 0.0
            return cam.grabber.get()

        ok, frame = cam.cap.read()
        ts = time.time()
        if not ok or frame is None:
            cam.connected = False
            cam.ended = True if cam.source_kind == 'file' else False
            try:
                cam.cap.release()
            except Exception:
                pass
            cam.cap = None
            return False, None, 0.0

        cam.connected = True
        cam.ended = False
        return True, frame, ts

    def _get_people_dets(self, frame_bgr):
        yolo_device = "0" if device.type == "cuda" else "cpu"
        r = self.yolo.predict(
            frame_bgr,
            verbose=False,
            device=yolo_device,
            conf=CONF_TH,
            classes=[0],
            max_det=max(6, MAX_TRACKS)
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.detach().cpu().numpy().astype(np.float32)

        dets = [(bb, float(c)) for bb, c in zip(xyxy, conf)]
        dets.sort(key=lambda x: x[1], reverse=True)
        return dets[:MAX_TRACKS]

    def _pack_and_predict(self, window_att_T17_3, length_L):
        T = window_att_T17_3.shape[0]
        j = window_att_T17_3.transpose(2, 0, 1).astype(np.float32)

        joint_ctvm = j[..., None]
        bone, motion = compute_bone_motion_from_joint_CTV(j)
        bone_ctvm = bone[..., None]
        motion_ctvm = motion[..., None]

        joint_n  = (joint_ctvm  - self.MEAN_J) / (self.STD_J + 1e-6)
        bone_n   = (bone_ctvm   - self.MEAN_B) / (self.STD_B + 1e-6)
        motion_n = (motion_ctvm - self.MEAN_M) / (self.STD_M + 1e-6)

        if length_L < T:
            joint_n[:, length_L:]  = self.PAD_J
            bone_n[:, length_L:]   = self.PAD_B
            motion_n[:, length_L:] = self.PAD_M

        jt = torch.from_numpy(joint_n).unsqueeze(0).float().to(device)
        bt = torch.from_numpy(bone_n).unsqueeze(0).float().to(device)
        mt = torch.from_numpy(motion_n).unsqueeze(0).float().to(device)
        Lt = torch.tensor([int(max(1, min(length_L, T)))], dtype=torch.long, device=device)

        with torch.no_grad():
            with autocast(enabled=use_amp):
                logits = self.model(jt, bt, mt, Lt)[0]
            probs = torch.softmax(logits.float(), dim=0).detach().cpu().numpy()

        k = int(np.argmax(probs))
        name = self.CLASS_NAMES.get(k, str(k))
        conf = float(probs[k])
        return name, conf

    def _process_cam_frame(self, cam_idx, cam, raw_frame, ts):
        cam.connected = True
        cam.frame_idx += 1

        # Keep a timestamp for UI notifications
        cam.last_ts_ms = int(ts * 1000) if ts else int(time.time() * 1000)

        if self.recorder is not None:
            try:
                self.recorder.push_frame(cam_idx, raw_frame, ts)
            except Exception:
                pass

        frame = cv2.resize(raw_frame, (INFER_W, INFER_H), interpolation=cv2.INTER_AREA)

        dets = self._get_people_dets(frame)
        assigned = cam.tracker.update(dets, cam.frame_idx)

        cam.current_pose = {}
        pose_samples = []
        if len(assigned) > 0:
            H, W = frame.shape[:2]
            bb_list = [expand_bbox(bb, W, H, scale=BBOX_EXPAND) for (_, bb) in assigned]
            bbox_array = np.stack(bb_list, axis=0).astype(np.float32)
            try:
                pose_samples = infer_pose_batch(self.pose_model, frame, bbox_array)
            except Exception:
                pose_samples = []

        for i, (tid, bb) in enumerate(assigned):
            tr = cam.tracker.tracks.get(tid)
            if tr is None:
                continue
            tr.bbox = bb
            tr.last_seen = cam.frame_idx

            pose3 = None

            # 1) try real pose
            if i < len(pose_samples):
                ds = pose_samples[i]
                try:
                    pi = ds.pred_instances
                    kpts = np.asarray(pi.keypoints, dtype=np.float32)[0]
                    scs  = np.asarray(pi.keypoint_scores, dtype=np.float32)[0]
                    pose3 = np.concatenate([kpts, scs[:, None]], axis=1).astype(np.float32)
                    cam.current_pose[tid] = pose3
                except Exception:
                    pose3 = None

            # 2) fallback pose if missing
            if pose3 is None:
                if len(tr.pose3) > 0:
                    ff = tr.pose3[-1].copy()
                    ff[:, 2] = 0.0
                    pose3 = ff
                else:
                    # create a zero-confidence "dummy" pose at bbox center
                    x1, y1, x2, y2 = bb.astype(np.float32)
                    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                    pose3 = np.zeros((17, 3), dtype=np.float32)
                    pose3[:, 0] = cx
                    pose3[:, 1] = cy
                    pose3[:, 2] = 0.0

            tr.pose3.append(pose3)

        # ------------------------------------------------------------
        # Stable IDs + NO GHOST BOXES:
        # - Keep tracks alive in tracker (TTL) for stable IDs
        # - Draw ONLY tracks that were updated (seen) on THIS frame
        # ------------------------------------------------------------
        tracks_out = []  # always define

        # Use a consistent ordering (recently seen first) and limit count
        active_all = sorted(
            cam.tracker.tracks.items(),
            key=lambda kv: kv[1].last_seen,
            reverse=True
        )[:MAX_TRACKS]

        # Prediction can run for active_all (including recently-lost) to keep IDs stable
        for tid, tr in active_all:
            if (cam.frame_idx - tr.last_pred_frame) < STRIDE_PRED:
                continue
            if len(tr.pose3) < self.T_WINDOW:
                continue

            try:
                seq = np.stack(list(tr.pose3)[-self.T_WINDOW:], axis=0).astype(np.float32)  # (T,17,3)
            except Exception:
                continue

            length_L = count_valid_frames(seq, kp_conf_th=KP_CONF_TH)
            length_L = int(max(1, min(length_L, self.T_WINDOW)))

            if IDLE_ENABLE:
                mot = motion_level_from_seq(seq)
                # If motion is extremely low, don't update prediction (keeps last label stable)
                if (length_L >= IDLE_MIN_FRAMES) and (mot <= float(NO_ACTION_MOTION_THRESH)):
                    # Emit an explicit idle state instead of skipping
                    tr.last_pred = ("Idle", 1.0)
                    tr.last_pred_frame = cam.frame_idx
                    tr.pred_hist.append(("Idle", 1.0))
                    continue

            seq_norm = center_scale_single(seq, kp_conf_th=KP_CONF_TH)

            try:
                name, conf = self._pack_and_predict(seq_norm, length_L)
            except Exception:
                continue

            # Smooth predictions (majority label + avg confidence for that label)
            tr.pred_hist.append((str(name), float(conf)))
            labels = [lb for (lb, _) in tr.pred_hist]
            if labels:
                top_label = max(set(labels), key=labels.count)
                confs = [cf for (lb, cf) in tr.pred_hist if lb == top_label]
                conf_s = float(sum(confs) / max(1, len(confs)))
            else:
                top_label, conf_s = str(name), float(conf)

            tr.last_pred = (top_label, conf_s)
            tr.last_pred_frame = cam.frame_idx

        # Draw ONLY tracks seen this frame => no "left behind" ghosts
        active_draw = [(tid, tr) for tid, tr in active_all if tr.last_seen == cam.frame_idx]

        for tid, tr in active_draw:
            x1, y1, x2, y2 = map(int, tr.bbox.tolist())
            x1 = int(max(0, min(x1, INFER_W - 1)))
            y1 = int(max(0, min(y1, INFER_H - 1)))
            x2 = int(max(0, min(x2, INFER_W - 1)))
            y2 = int(max(0, min(y2, INFER_H - 1)))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            label_draw = "ID {}".format(tid)
            confv = 0.0
            out_label = "UNKNOWN"

            if tr.last_pred is not None:
                out_label = str(tr.last_pred[0])
                confv = float(tr.last_pred[1])
                needed = violent_streak_needed(out_label, cam.fps)

                # update violent streak only for strong violent predictions
                if out_label in NOTIF_LABELS and confv >= NOTIF_MIN_CONF:
                    if tr.last_violent_label == out_label:
                        tr.violent_streak += 1
                    else:
                        tr.violent_streak = 1
                        tr.last_violent_label = out_label
                else:
                    tr.violent_streak = 0
                    tr.last_violent_label = ""

                label_draw = "ID {}: {} {:.1f}%".format(tid, out_label, confv * 100.0)

                if out_label in NOTIF_LABELS:
                    label_draw += " [{} / {}]".format(tr.violent_streak, needed)
            else:
                tr.violent_streak = 0
                tr.last_violent_label = ""

            (tw, th), _ = cv2.getTextSize(label_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), (0,255,0), -1)
            cv2.putText(frame, label_draw, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            if tid in cam.current_pose:
                draw_skeleton(frame, cam.current_pose[tid], (0,255,255), conf_th=0.25)

            tracks_out.append({
                "tid": int(tid),
                "label": str(out_label),
                "conf": float(confv),
                "violent_streak": int(tr.violent_streak),
                "violent_needed": int(violent_streak_needed(out_label, cam.fps)) if out_label in NOTIF_LABELS else 0,
            })

        if self.recorder is not None and tracks_out:
            try:
                violent_ready = [
                    t for t in tracks_out
                    if t.get("label") in NOTIF_LABELS
                    and float(t.get("conf", 0.0)) >= NOTIF_MIN_CONF
                    and int(t.get("violent_streak", 0)) >= int(t.get("violent_needed", 999999))
                ]

                if violent_ready:
                    top = max(violent_ready, key=lambda t: float(t.get("conf", 0.0)))
                    self.recorder.on_action(
                        cam_idx,
                        str(top.get("label", "")),
                        float(top.get("conf", 0.0)),
                        ts
                    )
            except Exception:
                pass

        now = time.time()
        dt = now - cam.prev_t
        cam.prev_t = now
        if dt > 0:
            cam.fps_hist.append(1.0 / dt)
        cam.fps = float(sum(cam.fps_hist)/len(cam.fps_hist)) if cam.fps_hist else 0.0

        cv2.putText(frame, "FPS: {:.1f}".format(cam.fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        stream = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)
        cam.frame_b64 = jpeg_b64(stream, quality=STREAM_JPEG_QUALITY)
        cam.tracks_out = tracks_out

    def _run_forever(self):
        for c in self.cams:
            c.source_kind = self._source_kind(c.source)
            c.cap = self._open_cap(c.source)
            c.connected = bool(c.cap is not None and c.cap.isOpened())
            if c.connected and c.source_kind == 'rtsp':
                try:
                    c.grabber = LatestFrameGrabber(c.cap, name='cam', reopen_src=c.source, reopen_backend=cv2.CAP_FFMPEG).start()
                except Exception:
                    c.grabber = None
            c.prev_t = time.time()

        while not self._stop_evt.is_set():
            frames = [None] * self.num_cams
            tss    = [0.0]  * self.num_cams
            oks    = [False]* self.num_cams

            with self.lock:
                for i in range(self.num_cams):
                    self._apply_pending_source(self.cams[i])

                for i in range(self.num_cams):
                    cam = self.cams[i]
                    if cam.cap is None or not cam.cap.isOpened():
                        cam.connected = False
                        cam.fps = 0.0
                        cam.frame_b64 = None
                        cam.tracks_out = []
                        cam.last_ts_ms = 0
                        try:
                            if cam.grabber is not None:
                                cam.grabber.stop()
                                cam.grabber = None
                        except Exception:
                            pass
                        continue

                    ok, frame, ts = self._capture_latest_frame(cam)
                    oks[i] = bool(ok)
                    frames[i] = frame
                    tss[i] = ts

            with self.lock:
                for i in range(self.num_cams):
                    if oks[i] and frames[i] is not None:
                        self._process_cam_frame(i, self.cams[i], frames[i], tss[i])
                self._packet_id += 1

            time.sleep(0.001)

        for c in self.cams:
            try:
                if c.grabber is not None:
                    c.grabber.stop()
                    c.grabber = None
            except Exception:
                pass
            try:
                if c.cap is not None:
                    c.cap.release()
            except Exception:
                pass
            c.cap = None
            c.connected = False
            c.fps = 0.0
            c.frame_b64 = None
            c.tracks_out = []
            c.last_ts_ms = 0