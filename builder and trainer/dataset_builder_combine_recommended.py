# build_ntu_coco17_2d_dataset_auto_attacker_BYTETRACK_M1_FINAL_COMBINED_SPLIT.py
# =========================================================
# NTU RGB (.avi) -> COCO-17 2D (RTMPose) -> CTR-GCN PKL (M=1 attacker only)
#
# CLEAN COMBINED SPLIT (recommended):
#   - Primary split is PERSON-EXCLUSIVE (by Pxxx) for real generalization
#   - Split happens BEFORE window generation (so no window leakage)
#   - Optionally uses COMMON-SUBJECT pool (subjects that have ALL selected actions)
#   - Balances per-class video counts across train/val/test using a greedy assignment
#
# Also includes:
#   - Offline max-window attacker pick for A050/A051
#   - Camera-bias guard to avoid picking closer victim
#   - Soft quality weight q (keeps dataset size)
#   - TID-LOCK reorder: chosen attacker track-id kept stable
#   - Robust AVI read (skip corrupted frames)
#   - Debug MP4: attacker=RED, other=GREEN + overlay text
# =========================================================

import os, re, json, random, pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter

from ultralytics import YOLO
from mmengine.registry import init_default_scope
from mmpose.apis import init_model as init_mmpose_model, inference_topdown

from ctrgcn_model import COCO17_PARENT

# -------------------------
# PATHS (EDIT)
# -------------------------
NTU_VIDEO_DIR = r"C:\Users\darlene\Downloads\NTU_DATASET"
OUT_DIR       = r"C:\Users\darlene\Downloads\NTU_DATASET_OUTPUT"

RTM_BODY_CFG  = r"C:\Users\darlene\Downloads\Capstoney\CTR_GCN\.venv\Lib\site-packages\mmpose\.mim\configs\body_2d_keypoint\rtmpose\coco\rtmpose-s_8xb256-420e_coco-256x192.py"
RTM_BODY_CKPT = r"C:\Users\darlene\Downloads\Capstoney\mmpose_ckpt\rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
YOLO_WEIGHTS  = "yolov8n.pt"
TRACKER_CFG   = "bytetrack.yaml"

# -------------------------
# SETTINGS
# -------------------------
T_WINDOW      = 36
WINDOW_STEP   = 6
FRAME_STRIDE  = 1
MAX_FRAMES    = 0  # 0 = no cap

CONF_TH       = 0.40
KP_CONF_TH    = 0.20

MAX_PEOPLE_DET = 2
BBOX_EXPAND = 1.25

# Your class set (edit as you add more single-person actions)
SELECTED_ACTION_IDS = [43, 45, 50, 51, 59, 99]
AUTO_ATTACKER_ACTIONS = [50, 51]  # only these need attacker picking

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# CLEAN COMBINED SPLIT OPTIONS
SPLIT_MODE = "person_balanced_common"   # "person_balanced_common" | "person_balanced_all" | "video_only_stratified"
MIN_COMMON_SUBJECTS = 12                # fallback to person_balanced_all if common pool too small
RANDOM_SEED = 1337

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------
# OFFLINE PICKER SETTINGS
# -------------------------
PICK_WIN     = 24
PICK_STEP    = 2
PICK_MARGIN  = 1.10

SCORE_CLOSE_W = 1.40
SCORE_PEAK_W  = 0.60
SCORE_TORSO_W = 1.10

# -------------------------
# CAMERA-BIAS GUARD
# -------------------------
CAMBIAS_ENABLE = True
CAMBIAS_DIAG_RATIO_TH = 1.25
CAMBIAS_PENALTY_W     = 0.60
CAMBIAS_MIN_FRAC      = 0.60

# -------------------------
# SOFT QUALITY WEIGHTING
# -------------------------
QUALITY_ENABLE = True
Q_MIN = 0.05
Q_KP_TH = 0.15
Q_LIMB_MIN_FRAC = 0.35

# -------------------------
# SLOT ASSIGNMENT GATING
# -------------------------
CENTER_GATE_RATIO = 1.6

# -------------------------
# DEBUG VIDEO OUTPUT
# -------------------------
DEBUG_SAVE_VIDEOS = False
DEBUG_MAX_VIDEOS  = 10
DEBUG_VIDEO_DIR   = os.path.join(OUT_DIR, "_debug_tidlock")
os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True)

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda" if device.type == "cuda" else "cpu"

# COCO17 indices
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16
TORSO_JOINTS = [L_HIP, R_HIP, L_SHO, R_SHO]

COCO_EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6),
    (5,11),(6,12),
    (11,12),
    (0,5),(0,6),
    (0,1),(0,2),(1,3),(2,4)
]

# =========================================================
# helpers
# =========================================================
def parse_ntu_filename(fn: str):
    base = os.path.basename(fn)
    mS = re.search(r"S(\d{3})", base)
    mC = re.search(r"C(\d{3})", base)
    mP = re.search(r"P(\d{3})", base)
    mR = re.search(r"R(\d{3})", base)
    mA = re.search(r"A(\d{3})", base)
    if not (mS and mC and mP and mR and mA):
        return None
    return {"S": int(mS.group(1)), "C": int(mC.group(1)), "P": int(mP.group(1)),
            "R": int(mR.group(1)), "A": int(mA.group(1)), "base": base}

def bbox_center(bb):
    x1,y1,x2,y2 = bb.astype(np.float32)
    return np.array([(x1+x2)*0.5, (y1+y2)*0.5], np.float32)

def bbox_diag(bb):
    x1,y1,x2,y2 = bb.astype(np.float32)
    return float(np.hypot(x2-x1, y2-y1) + 1e-6)

def expand_bbox(bb, W, H, scale=1.25):
    x1,y1,x2,y2 = bb.astype(np.float32)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = max(0, cx-w/2), max(0, cy-h/2)
    nx2, ny2 = min(W-1, cx+w/2), min(H-1, cy+h/2)
    return np.array([nx1, ny1, nx2, ny2], np.float32)

def count_valid_frames(kpt_T17_3, kp_conf_th=0.2):
    sc = kpt_T17_3[:, :, 2]
    torso = sc[:, TORSO_JOINTS]
    return int((torso.mean(axis=1) >= kp_conf_th).sum())

def compute_bone_motion_from_joint_CTV(joint_CTV: np.ndarray):
    bone = np.zeros_like(joint_CTV, dtype=np.float32)
    motion = np.zeros_like(joint_CTV, dtype=np.float32)
    for v, p in enumerate(COCO17_PARENT):
        if p >= 0:
            bone[0:2, :, v] = joint_CTV[0:2, :, v] - joint_CTV[0:2, :, p]
    bone[2, :, :] = 0.0
    motion[0:2, 1:, :] = joint_CTV[0:2, 1:, :] - joint_CTV[0:2, :-1, :]
    motion[2, :, :] = 0.0
    return bone, motion

def seq_center_scale_single(seq_T17_3, kp_conf_th=0.2):
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
        if s < 1e-6: s = 1.0
        out[t, :, 0:2] = xy / s
        out[t, :, 2] = np.clip(out[t, :, 2], 0.0, 1.0)
    return out

def _moving_average_1d(x, k=3):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, w, mode="valid").astype(np.float32)

def smooth_seq_xy(seq_T17_3, kp_conf_th=0.15, k=3, vel_clip=0.25):
    out = seq_T17_3.copy().astype(np.float32)
    T = out.shape[0]
    for j in range(17):
        sc = out[:, j, 2]
        good = sc >= kp_conf_th
        if good.sum() < 4:
            continue
        for c in (0, 1):
            x = out[:, j, c].copy()
            xg = x.copy()
            xg[~good] = xg[good][0]
            xs = _moving_average_1d(xg, k=k)
            x[good] = xs[good]
            out[:, j, c] = x

        v = out[1:, j, 0:2] - out[:-1, j, 0:2]
        mag = np.linalg.norm(v, axis=1)
        too = mag > vel_clip
        if np.any(too):
            scale = (vel_clip / (mag[too] + 1e-6)).reshape(-1, 1)
            v[too] *= scale
            out[1:, j, 0:2] = out[:-1, j, 0:2] + v
    return out

def compute_window_quality_single(w_T17_3, action_id, kp_conf_th=0.15):
    sc = w_T17_3[:, :, 2].astype(np.float32)
    torso_vis = (sc[:, TORSO_JOINTS].mean(axis=1) >= kp_conf_th).mean()

    if action_id == 51:
        limb = [L_KNE, R_KNE, L_ANK, R_ANK]
    elif action_id == 50:
        limb = [L_ELB, R_ELB, L_WRI, R_WRI]
    else:
        limb = TORSO_JOINTS

    limb_vis = (sc[:, limb].mean(axis=1) >= kp_conf_th).mean()

    xy = w_T17_3[:, :, 0:2].astype(np.float32)
    torso_xy = xy[:, TORSO_JOINTS, :].mean(axis=1)
    v = torso_xy[1:] - torso_xy[:-1]
    jitter = float(np.linalg.norm(v, axis=1).mean()) if len(v) else 0.0

    jitter_score = 1.0 - np.clip((jitter - 0.03) / (0.08 - 0.03 + 1e-6), 0.0, 1.0)

    q = (0.50 * float(torso_vis)) + (0.35 * float(limb_vis)) + (0.15 * float(jitter_score))
    q = float(np.clip(q, Q_MIN, 1.0))

    if action_id in AUTO_ATTACKER_ACTIONS and float(limb_vis) < float(Q_LIMB_MIN_FRAC):
        q = min(q, 0.12)

    return q, float(torso_vis), float(limb_vis), float(jitter)

# =========================================================
# Pose inference wrapper
# =========================================================
def infer_pose_one(pose_model, frame_bgr, bb_xyxy, bb_score):
    bb_xyxy = bb_xyxy.astype(np.float32)
    try:
        person = [{"bbox": bb_xyxy, "bbox_score": float(bb_score)}]
        with torch.no_grad():
            pose_samples = inference_topdown(pose_model, frame_bgr, person)
        return pose_samples
    except Exception:
        bboxes = bb_xyxy[None, :].astype(np.float32)
        with torch.no_grad():
            pose_samples = inference_topdown(pose_model, frame_bgr, bboxes)
        return pose_samples

def safe_reset_tracker(yolo: YOLO):
    try:
        if hasattr(yolo, "predictor") and yolo.predictor is not None:
            pred = yolo.predictor
            if hasattr(pred, "tracker") and pred.tracker is not None and hasattr(pred.tracker, "reset"):
                pred.tracker.reset()
                return
            if hasattr(pred, "trackers") and pred.trackers:
                for trk in pred.trackers:
                    if hasattr(trk, "reset"):
                        trk.reset()
    except Exception:
        pass

def detect_people_tracks(frame_bgr, yolo: YOLO):
    results = yolo.track(
        frame_bgr, persist=True, verbose=False, device=DEVICE,
        conf=CONF_TH, iou=0.5, classes=[0], max_det=MAX_PEOPLE_DET,
        tracker=TRACKER_CFG,
    )
    if not results: return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    ids = getattr(r.boxes, "id", None)
    if ids is None: return []
    ids_np = ids.detach().cpu().numpy().astype(np.int64)
    xyxy_np = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf_np = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
    dets = [(int(tid), float(cf), bb) for tid, cf, bb in zip(ids_np, conf_np, xyxy_np)]
    dets.sort(key=lambda x: x[1], reverse=True)
    return dets

# =========================================================
# Slot assignment (does NOT define identity; identity comes from tid-lock)
# =========================================================
def assign_two_slots_with_gating(dets, last_center, last_diag, gate_ratio=1.6):
    slot_bbox = [None, None]
    slot_conf = [0.0, 0.0]
    slot_tid  = [None, None]
    if len(dets) == 0:
        return slot_bbox, slot_conf, slot_tid

    centers = [bbox_center(bb) for (_, _, bb) in dets]
    huge = 1e9

    if last_center[0] is None and last_center[1] is None:
        dets2 = sorted(dets, key=lambda x: x[1], reverse=True)[:2]
        for si, (tid, cf, bb) in enumerate(dets2):
            slot_bbox[si] = bb; slot_conf[si] = cf; slot_tid[si] = tid
        return slot_bbox, slot_conf, slot_tid

    costs = [[huge]*len(dets), [huge]*len(dets)]
    for si in (0,1):
        if last_center[si] is None:
            for j in range(len(dets)):
                costs[si][j] = 0.0
        else:
            gate = gate_ratio * float(last_diag[si])
            for j, c in enumerate(centers):
                d = float(np.linalg.norm(c - last_center[si]))
                costs[si][j] = d if d <= gate else huge

    best = (huge, None, None)
    for j0 in range(len(dets)):
        for j1 in range(len(dets)):
            if j1 == j0: continue
            total = costs[0][j0] + costs[1][j1]
            if total < best[0]:
                best = (total, j0, j1)

    total_cost, j0, j1 = best
    if total_cost < huge:
        tid0, cf0, bb0 = dets[j0]
        tid1, cf1, bb1 = dets[j1]
        slot_bbox[0] = bb0; slot_conf[0] = cf0; slot_tid[0] = tid0
        slot_bbox[1] = bb1; slot_conf[1] = cf1; slot_tid[1] = tid1
    else:
        dets2 = sorted(dets, key=lambda x: x[1], reverse=True)[:2]
        for si, (tid, cf, bb) in enumerate(dets2):
            slot_bbox[si] = bb; slot_conf[si] = cf; slot_tid[si] = tid
    return slot_bbox, slot_conf, slot_tid

# =========================================================
# Shared normalization for picking
# =========================================================
def _torso_center_xy(seq, t, kp_conf_th=0.2):
    sc_hips = seq[t, [L_HIP, R_HIP], 2].mean()
    if sc_hips >= kp_conf_th:
        return 0.5 * (seq[t, L_HIP, 0:2] + seq[t, R_HIP, 0:2])
    sc_sho = seq[t, [L_SHO, R_SHO], 2].mean()
    if sc_sho >= kp_conf_th:
        return 0.5 * (seq[t, L_SHO, 0:2] + seq[t, R_SHO, 0:2])
    return np.array([0.0, 0.0], np.float32)

def seq_center_scale_shared_for_pick(seq0, seq1, kp_conf_th=0.2):
    out0 = seq0.copy().astype(np.float32)
    out1 = seq1.copy().astype(np.float32)
    T = min(out0.shape[0], out1.shape[0])
    for t in range(T):
        c0 = _torso_center_xy(out0, t, kp_conf_th)
        c1 = _torso_center_xy(out1, t, kp_conf_th)
        v0 = (out0[t, TORSO_JOINTS, 2].mean() >= kp_conf_th)
        v1 = (out1[t, TORSO_JOINTS, 2].mean() >= kp_conf_th)
        if v0 and v1: center = 0.5*(c0+c1)
        elif v0:      center = c0
        elif v1:      center = c1
        else:         center = np.array([0.0, 0.0], np.float32)

        x0 = out0[t, :, 0:2] - center[None, :]
        x1 = out1[t, :, 0:2] - center[None, :]
        r0 = np.linalg.norm(x0, axis=1)
        r1 = np.linalg.norm(x1, axis=1)
        s = float(max(np.max(r0), np.max(r1)))
        if s < 1e-6: s = 1.0
        out0[t, :, 0:2] = x0 / s
        out1[t, :, 0:2] = x1 / s
        out0[t, :, 2] = np.clip(out0[t, :, 2], 0.0, 1.0)
        out1[t, :, 2] = np.clip(out1[t, :, 2], 0.0, 1.0)
    return out0, out1

def _torso_motion(seq, last_k=24, kp_conf_th=0.2):
    T = seq.shape[0]
    k = min(last_k, T)
    if k < 2: return 0.0
    centers, good = [], []
    for t in range(T-k, T):
        centers.append(_torso_center_xy(seq, t, kp_conf_th))
        good.append(seq[t, TORSO_JOINTS, 2].mean() >= kp_conf_th)
    centers = np.stack(centers, axis=0)
    good = np.array(good, dtype=bool)
    if good.sum() < 2: return 0.0
    centers = centers[good]
    v = centers[1:] - centers[:-1]
    return float(np.linalg.norm(v, axis=1).mean())

def _strike_toward_other_score(seq_self, seq_other, action_id, last_k=24, kp_conf_th=0.2):
    if action_id == 50:
        joints = [L_WRI, R_WRI, L_ELB, R_ELB]
        jw = {L_WRI: 1.6, R_WRI: 1.6, L_ELB: 1.0, R_ELB: 1.0}
    elif action_id == 51:
        joints = [L_ANK, R_ANK, L_KNE, R_KNE]
        jw = {L_ANK: 1.6, R_ANK: 1.6, L_KNE: 1.0, R_KNE: 1.0}
    else:
        joints = TORSO_JOINTS
        jw = {j: 1.0 for j in joints}

    T = seq_self.shape[0]
    k = min(last_k, T)
    if k < 2:
        return 0.0, 0, 0

    t0 = T - k
    per_t, good_frames = [], 0

    for tt in range(t0+1, T):
        c_other = _torso_center_xy(seq_other, tt, kp_conf_th)
        frame_score, conf_ok = 0.0, 0

        for j in joints:
            p_prev = seq_self[tt-1, j, 0:2].astype(np.float32)
            p_cur  = seq_self[tt,   j, 0:2].astype(np.float32)
            v = p_cur - p_prev

            d = (c_other - p_prev).astype(np.float32)
            dn = float(np.linalg.norm(d) + 1e-6)
            d_hat = d / dn

            toward = float(np.dot(v, d_hat))
            if toward <= 0:
                continue

            c1 = float(seq_self[tt-1, j, 2])
            c2 = float(seq_self[tt,   j, 2])
            cw = 0.5*(c1+c2)
            if cw >= kp_conf_th:
                conf_ok += 1

            frame_score += toward * cw * jw.get(j, 1.0)

        if conf_ok > 0:
            good_frames += 1
        per_t.append(frame_score)

    per_t = np.array(per_t, dtype=np.float32)
    total = float(per_t.sum()) if len(per_t) else 0.0
    peak  = int(np.argmax(per_t)) if len(per_t) else 0
    return total, peak, int(good_frames)

def _closing_speed_score(seq_self, seq_other, action_id, last_k=24, kp_conf_th=0.2):
    if action_id == 50:
        joints = [L_WRI, R_WRI]
    elif action_id == 51:
        joints = [L_ANK, R_ANK]
    else:
        return 0.0

    T = seq_self.shape[0]
    k = min(last_k, T)
    if k < 2:
        return 0.0

    t0 = T - k
    total, denom = 0.0, 0.0

    for tt in range(t0+1, T):
        c_other = _torso_center_xy(seq_other, tt, kp_conf_th)
        for j in joints:
            sc1 = float(seq_self[tt-1, j, 2])
            sc2 = float(seq_self[tt,   j, 2])
            cw = 0.5*(sc1+sc2)
            if cw < kp_conf_th:
                continue
            p_prev = seq_self[tt-1, j, 0:2].astype(np.float32)
            p_cur  = seq_self[tt,   j, 0:2].astype(np.float32)
            d_prev = float(np.linalg.norm(c_other - p_prev))
            d_cur  = float(np.linalg.norm(c_other - p_cur))
            closing = max(0.0, d_prev - d_cur)
            total += closing * cw
            denom += cw

    return float(total / (denom + 1e-6))

def _camera_bias_stats(diag0_win, diag1_win, ratio_th=1.25):
    d0 = np.asarray(diag0_win, dtype=np.float32)
    d1 = np.asarray(diag1_win, dtype=np.float32)
    ok = (d0 > 1e-6) & (d1 > 1e-6)
    if ok.sum() < 4:
        return 0.0, 0.0, 1.0
    r01 = d0[ok] / d1[ok]
    frac0 = float((r01 >= ratio_th).mean())
    frac1 = float((r01 <= (1.0 / ratio_th)).mean())
    medr = float(np.median(r01))
    return frac0, frac1, medr

def pick_attacker_offline_maxwindow_with_ratio(
    seq0, seq1, action_id,
    diag0_list=None, diag1_list=None,
    win=PICK_WIN, step=PICK_STEP,
    kp_conf_th=KP_CONF_TH,
    margin=PICK_MARGIN
):
    cam_info = {"enabled": bool(CAMBIAS_ENABLE), "frac0_big": 0.0, "frac1_big": 0.0, "med_ratio": 1.0}

    if action_id not in AUTO_ATTACKER_ACTIONS:
        v0 = count_valid_frames(seq0, kp_conf_th=kp_conf_th)
        v1 = count_valid_frames(seq1, kp_conf_th=kp_conf_th)
        chosen = 0 if v0 >= v1 else 1
        return chosen, 999.0, float(v0), float(v1), cam_info

    T = min(seq0.shape[0], seq1.shape[0])
    if T < 8:
        v0 = count_valid_frames(seq0, kp_conf_th=kp_conf_th)
        v1 = count_valid_frames(seq1, kp_conf_th=kp_conf_th)
        chosen = 0 if v0 >= v1 else 1
        return chosen, 1.0, float(v0), float(v1), cam_info

    win = int(min(win, T))
    win = max(8, win)

    best0 = -1e9
    best1 = -1e9
    min_good = max(6, win // 3)

    have_diag = (diag0_list is not None) and (diag1_list is not None) and (len(diag0_list) >= T) and (len(diag1_list) >= T)

    for s in range(0, max(1, T - win + 1), step):
        w0 = seq0[s:s+win]
        w1 = seq1[s:s+win]
        n0, n1 = seq_center_scale_shared_for_pick(w0, w1, kp_conf_th=kp_conf_th)

        s0, p0, g0 = _strike_toward_other_score(n0, n1, action_id, last_k=win, kp_conf_th=kp_conf_th)
        s1, p1, g1 = _strike_toward_other_score(n1, n0, action_id, last_k=win, kp_conf_th=kp_conf_th)
        if min(g0, g1) < min_good:
            continue

        k_eff = max(1, win - 1)
        ep0 = 1.0 - (p0 / float(k_eff))
        ep1 = 1.0 - (p1 / float(k_eff))

        t0 = _torso_motion(n0, last_k=win, kp_conf_th=kp_conf_th)
        t1 = _torso_motion(n1, last_k=win, kp_conf_th=kp_conf_th)

        c0 = _closing_speed_score(n0, n1, action_id, last_k=win, kp_conf_th=kp_conf_th)
        c1 = _closing_speed_score(n1, n0, action_id, last_k=win, kp_conf_th=kp_conf_th)

        score0 = s0 + SCORE_CLOSE_W*c0 + SCORE_PEAK_W*ep0 - SCORE_TORSO_W*t0
        score1 = s1 + SCORE_CLOSE_W*c1 + SCORE_PEAK_W*ep1 - SCORE_TORSO_W*t1

        if CAMBIAS_ENABLE and have_diag:
            d0w = diag0_list[s:s+win]
            d1w = diag1_list[s:s+win]
            frac0_big, frac1_big, med_ratio = _camera_bias_stats(d0w, d1w, ratio_th=CAMBIAS_DIAG_RATIO_TH)

            if frac0_big >= CAMBIAS_MIN_FRAC:
                score0 -= CAMBIAS_PENALTY_W * (frac0_big)
            if frac1_big >= CAMBIAS_MIN_FRAC:
                score1 -= CAMBIAS_PENALTY_W * (frac1_big)

            if (frac0_big + frac1_big) > (cam_info["frac0_big"] + cam_info["frac1_big"]):
                cam_info = {"enabled": True, "frac0_big": float(frac0_big), "frac1_big": float(frac1_big), "med_ratio": float(med_ratio)}

        best0 = max(best0, score0)
        best1 = max(best1, score1)

    if best0 <= -1e8 and best1 <= -1e8:
        v0 = count_valid_frames(seq0, kp_conf_th=kp_conf_th)
        v1 = count_valid_frames(seq1, kp_conf_th=kp_conf_th)
        chosen = 0 if v0 >= v1 else 1
        return chosen, 1.0, float(v0), float(v1), cam_info

    chosen = 0 if best0 >= best1 else 1

    if chosen == 1 and not (best1 > best0 * float(margin)):
        chosen = 0

    winner, loser = (best0, best1) if chosen == 0 else (best1, best0)
    ratio = float(winner / (abs(loser) + 1e-6)) if loser != 0 else 999.0
    return chosen, ratio, float(best0), float(best1), cam_info

# =========================================================
# Debug draw
# =========================================================
def draw_skeleton(img, kpt17_3, color, conf_th=0.2):
    k = kpt17_3
    for a,b in COCO_EDGES:
        if k[a,2] >= conf_th and k[b,2] >= conf_th:
            cv2.line(img, (int(k[a,0]), int(k[a,1])), (int(k[b,0]), int(k[b,1])), color, 2)
    for j in range(17):
        if k[j,2] >= conf_th:
            cv2.circle(img, (int(k[j,0]), int(k[j,1])), 3, color, -1)

def save_debug_video(frames, out_mp4_path, action_id, ratio_conf, chosen_tid, cam_info):
    if not frames:
        return
    H, W = frames[0][0].shape[:2]
    fps = 30.0
    vw = cv2.VideoWriter(out_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for (frame, atk_k, vic_k) in frames:
        vis = frame
        draw_skeleton(vis, vic_k, (0,255,0), conf_th=KP_CONF_TH)
        draw_skeleton(vis, atk_k, (0,0,255), conf_th=KP_CONF_TH)
        if action_id in AUTO_ATTACKER_ACTIONS:
            txt = f"A{action_id:03d} ATTACKER=RED tid={chosen_tid} ratio={ratio_conf:.2f} cam(fr0={cam_info['frac0_big']:.2f},fr1={cam_info['frac1_big']:.2f},med={cam_info['med_ratio']:.2f})"
        else:
            txt = f"A{action_id:03d} (single/none) tid={chosen_tid}"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
        vw.write(vis)
    vw.release()

# =========================================================
# Extract full sequences + per-frame tids + diags for both slots
# =========================================================
def extract_two_seqs_full_with_tid(video_path, yolo, pose_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None, None, None, None

    safe_reset_tracker(yolo)

    last_center = [None, None]
    last_diag   = [150.0, 150.0]

    seq0, seq1 = [], []
    tid0_list, tid1_list = [], []
    diag0_list, diag1_list = [], []
    last_pose0 = None
    last_pose1 = None

    raw_frames = []

    frame_i = 0
    kept = 0
    bad_read_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            bad_read_count += 1
            if bad_read_count >= 30:
                break
            continue
        bad_read_count = 0

        frame_i += 1
        if FRAME_STRIDE > 1 and ((frame_i - 1) % FRAME_STRIDE != 0):
            continue

        H, W = frame.shape[:2]
        dets = detect_people_tracks(frame, yolo)

        slot_bbox, slot_conf, slot_tid = assign_two_slots_with_gating(
            dets, last_center, last_diag, gate_ratio=CENTER_GATE_RATIO
        )

        out0 = None
        out1 = None
        tid0 = slot_tid[0]
        tid1 = slot_tid[1]

        d0 = bbox_diag(slot_bbox[0]) if slot_bbox[0] is not None else 0.0
        d1 = bbox_diag(slot_bbox[1]) if slot_bbox[1] is not None else 0.0
        diag0_list.append(float(d0))
        diag1_list.append(float(d1))

        for si in (0,1):
            bb = slot_bbox[si]
            if bb is None:
                if si == 0:
                    if last_pose0 is not None:
                        ff = last_pose0.copy(); ff[:,2] = 0.0
                        seq0.append(ff); out0 = ff
                    else:
                        z = np.zeros((17,3), np.float32)
                        seq0.append(z); out0 = z
                    tid0_list.append(None)
                else:
                    if last_pose1 is not None:
                        ff = last_pose1.copy(); ff[:,2] = 0.0
                        seq1.append(ff); out1 = ff
                    else:
                        z = np.zeros((17,3), np.float32)
                        seq1.append(z); out1 = z
                    tid1_list.append(None)
                continue

            last_center[si] = bbox_center(bb)
            last_diag[si]   = bbox_diag(bb)

            bb2 = expand_bbox(bb, W, H, scale=BBOX_EXPAND)
            pose_samples = infer_pose_one(pose_model, frame, bb2, slot_conf[si])

            if pose_samples is None or len(pose_samples) == 0:
                if si == 0:
                    if last_pose0 is not None:
                        ff = last_pose0.copy(); ff[:,2] = 0.0
                        seq0.append(ff); out0 = ff
                    else:
                        z = np.zeros((17,3), np.float32)
                        seq0.append(z); out0 = z
                    tid0_list.append(tid0)
                else:
                    if last_pose1 is not None:
                        ff = last_pose1.copy(); ff[:,2] = 0.0
                        seq1.append(ff); out1 = ff
                    else:
                        z = np.zeros((17,3), np.float32)
                        seq1.append(z); out1 = z
                    tid1_list.append(tid1)
                continue

            pi = pose_samples[0].pred_instances
            kpts = np.asarray(pi.keypoints, dtype=np.float32)[0]
            scs  = np.asarray(pi.keypoint_scores, dtype=np.float32)[0]
            out = np.concatenate([kpts, scs[:,None]], axis=1).astype(np.float32)

            if si == 0:
                seq0.append(out); last_pose0 = out; out0 = out
                tid0_list.append(tid0)
            else:
                seq1.append(out); last_pose1 = out; out1 = out
                tid1_list.append(tid1)

        raw_frames.append((frame.copy(), out0, out1, tid0, tid1))

        kept += 1
        if MAX_FRAMES and kept >= MAX_FRAMES:
            break

    cap.release()

    if len(seq0) == 0:
        return None, None, None, None, None, None, None

    seq0 = np.stack(seq0, axis=0).astype(np.float32)
    seq1 = np.stack(seq1, axis=0).astype(np.float32)
    tid0_list = np.array(tid0_list, dtype=object)
    tid1_list = np.array(tid1_list, dtype=object)
    diag0_list = np.array(diag0_list, dtype=np.float32)
    diag1_list = np.array(diag1_list, dtype=np.float32)
    return seq0, seq1, tid0_list, tid1_list, diag0_list, diag1_list, raw_frames

# =========================================================
# TID lock reorder
# =========================================================
def reorder_by_chosen_tid(seq0, seq1, tid0_list, tid1_list, chosen_tid):
    T = min(seq0.shape[0], seq1.shape[0], len(tid0_list), len(tid1_list))
    attacker = []
    victim = []
    last_atk = None
    last_vic = None

    for t in range(T):
        t0 = tid0_list[t]
        t1 = tid1_list[t]

        if t0 is not None and int(t0) == int(chosen_tid):
            atk = seq0[t]; vic = seq1[t]
        elif t1 is not None and int(t1) == int(chosen_tid):
            atk = seq1[t]; vic = seq0[t]
        else:
            if last_atk is not None:
                atk = last_atk.copy(); atk[:,2] = 0.0
            else:
                atk = np.zeros((17,3), np.float32)

            v0 = float(seq0[t, TORSO_JOINTS, 2].mean())
            v1 = float(seq1[t, TORSO_JOINTS, 2].mean())
            cand = seq0[t] if v0 >= v1 else seq1[t]
            vic = cand if cand is not None else (last_vic if last_vic is not None else np.zeros((17,3), np.float32))

        attacker.append(atk)
        victim.append(vic)
        last_atk = atk
        last_vic = vic

    attacker = np.stack(attacker, axis=0).astype(np.float32)
    victim   = np.stack(victim, axis=0).astype(np.float32)
    return attacker, victim

# =========================================================
# stats
# =========================================================
def compute_masked_mean_std(samples):
    J = np.stack([s["joint"] for s in samples], axis=0)   # (N,C,T,V,1)
    B = np.stack([s["bone"] for s in samples], axis=0)
    Mv= np.stack([s["motion"] for s in samples], axis=0)
    L = np.array([s["length"] for s in samples], dtype=np.int64)

    def masked_stats(X):
        N, C, T, V, Mx = X.shape
        mask = np.zeros((N, 1, T, 1, 1), dtype=np.float32)
        for i, l in enumerate(L):
            mask[i, 0, :min(int(l), T), 0, 0] = 1.0
        m = np.repeat(mask, C, axis=1)
        m = np.repeat(m, V, axis=3)
        m = np.repeat(m, Mx, axis=4)
        denom = m.sum(axis=(0,2,3,4), keepdims=True) + 1e-8
        mu = (X * m).sum(axis=(0,2,3,4), keepdims=True) / denom
        sig = np.sqrt(((X - mu) ** 2 * m).sum(axis=(0,2,3,4), keepdims=True) / denom) + 1e-5
        return mu.astype(np.float32), sig.astype(np.float32)

    mean = {}
    std = {}
    mean["joint"],  std["joint"]  = masked_stats(J)
    mean["bone"],   std["bone"]   = masked_stats(B)
    mean["motion"], std["motion"] = masked_stats(Mv)
    return mean, std

# =========================================================
# SPLIT LOGIC (CLEAN COMBINED APPROACH)
# =========================================================
def build_video_index(video_dir, selected_actions):
    """
    Returns:
      items: list of dict {vp, rel, meta}
      byP: dict P -> list of items
      subjects_byA: dict A -> set(P)
      videos_byA: dict A -> list of items
    """
    items = []
    byP = defaultdict(list)
    subjects_byA = defaultdict(set)
    videos_byA = defaultdict(list)

    for root, _, files in os.walk(video_dir):
        for fn in files:
            if not fn.lower().endswith(".avi"):
                continue
            meta = parse_ntu_filename(fn)
            if not meta:
                continue
            A = int(meta["A"])
            if A not in selected_actions:
                continue
            vp = os.path.join(root, fn)
            rel = os.path.relpath(vp, video_dir)
            it = {"vp": vp, "rel": rel, "meta": meta}
            items.append(it)

            P = int(meta["P"])
            byP[P].append(it)
            subjects_byA[A].add(P)
            videos_byA[A].append(it)

    return items, byP, subjects_byA, videos_byA

def greedy_balanced_subject_split(subject_counts, actions, ratios=(0.8, 0.1, 0.1), seed=1337):
    """
    subject_counts: dict P -> dict A -> count_videos
    actions: list of action ids
    ratios: train/val/test
    Returns: splitP dict P -> "train"/"val"/"test", plus stats dict
    """
    rnd = random.Random(seed)

    # total per action
    totalA = {A: 0 for A in actions}
    for P, ca in subject_counts.items():
        for A in actions:
            totalA[A] += int(ca.get(A, 0))

    # targets per split per action
    splits = ["train", "val", "test"]
    tr, va, te = ratios
    target = {
        "train": {A: totalA[A] * tr for A in actions},
        "val":   {A: totalA[A] * va for A in actions},
        "test":  {A: totalA[A] * te for A in actions},
    }

    current = {sp: {A: 0 for A in actions} for sp in splits}
    splitP = {}

    # subjects sorted by "weight" (most videos first)
    subj_list = list(subject_counts.keys())
    rnd.shuffle(subj_list)
    subj_list.sort(key=lambda P: sum(subject_counts[P].get(A, 0) for A in actions), reverse=True)

    def cost_of_assign(P, sp):
        # squared error delta across all actions
        delta = 0.0
        cP = subject_counts[P]
        for A in actions:
            before = current[sp][A]
            after  = before + int(cP.get(A, 0))
            t = target[sp][A]
            delta += (after - t) ** 2 - (before - t) ** 2
        return delta

    for P in subj_list:
        # choose split with minimal cost increase
        costs = [(cost_of_assign(P, sp), sp) for sp in splits]
        costs.sort(key=lambda x: x[0])
        best_sp = costs[0][1]
        splitP[P] = best_sp

        cP = subject_counts[P]
        for A in actions:
            current[best_sp][A] += int(cP.get(A, 0))

    stats = {"totalA": totalA, "target": target, "current": current}
    return splitP, stats

def print_split_stats(splitP, byP, actions):
    # count subjects + videos per action per split
    splits = ["train", "val", "test"]
    subjs = {sp: [] for sp in splits}
    vidsA = {sp: {A: 0 for A in actions} for sp in splits}
    vids_total = {sp: 0 for sp in splits}

    for P, sp in splitP.items():
        subjs[sp].append(P)
        for it in byP[P]:
            A = int(it["meta"]["A"])
            vidsA[sp][A] += 1
            vids_total[sp] += 1

    print("\n[Split Summary]")
    for sp in splits:
        print(f"  {sp}: subjects={len(subjs[sp])} videos={vids_total[sp]}")
        for A in actions:
            print(f"    A{A:03d}: {vidsA[sp][A]}")

    return subjs, vidsA, vids_total

# =========================================================
# main
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # class map
    selected = sorted(list(set(SELECTED_ACTION_IDS)))
    action_to_class = {a: i for i, a in enumerate(selected)}
    idx_to_name = {i: f"NTU_A{a:03d}" for a, i in action_to_class.items()}
    with open(os.path.join(OUT_DIR, "class_map.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_name, f, indent=2)

    # index videos
    items, byP, subjects_byA, videos_byA = build_video_index(NTU_VIDEO_DIR, set(selected))
    print("Found selected videos:", len(items))
    if not items:
        return

    # Decide subject pool
    actions = selected
    sets = [subjects_byA.get(A, set()) for A in actions]
    common_subjects = set.intersection(*sets) if sets else set()
    print("Subjects per action:", {f"A{A:03d}": len(subjects_byA.get(A, set())) for A in actions})
    print("Common subjects (have all selected actions):", len(common_subjects))

    if SPLIT_MODE == "person_balanced_common":
        if len(common_subjects) >= MIN_COMMON_SUBJECTS:
            subject_pool = sorted(list(common_subjects))
            print(f"[SplitMode] person_balanced_common using common pool of {len(subject_pool)} subjects")
        else:
            subject_pool = sorted(list(byP.keys()))
            print(f"[SplitMode] common pool too small ({len(common_subjects)}). Fallback -> person_balanced_all with {len(subject_pool)} subjects")
    elif SPLIT_MODE == "person_balanced_all":
        subject_pool = sorted(list(byP.keys()))
        print(f"[SplitMode] person_balanced_all using {len(subject_pool)} subjects")
    elif SPLIT_MODE == "video_only_stratified":
        subject_pool = None
        print("[SplitMode] video_only_stratified (not person-exclusive)")
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # Create split mapping for videos
    video_split = {}  # vp -> split

    if SPLIT_MODE == "video_only_stratified":
        # video-only stratified by action (baseline)
        rnd = random.Random(RANDOM_SEED)
        byA = defaultdict(list)
        for it in items:
            A = int(it["meta"]["A"])
            byA[A].append(it)

        def _counts_for_class(n, tr=TRAIN_RATIO, va=VAL_RATIO):
            if n <= 0: return 0, 0, 0
            if n == 1: return 1, 0, 0
            if n == 2: return 1, 1, 0
            n_train = int(round(n * tr))
            n_val   = int(round(n * va))
            n_train = max(1, n_train)
            n_val   = max(1, n_val)
            if n_train + n_val >= n:
                while n_train + n_val >= n and n_train > 1: n_train -= 1
                while n_train + n_val >= n and n_val > 1: n_val -= 1
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_train > 1: n_train -= 1
                else: n_val = max(1, n_val - 1)
            return n_train, n_val, n_test

        print("\n[Split] video-only stratified by action")
        for A in sorted(byA.keys()):
            lst = byA[A]
            rnd.shuffle(lst)
            n_tr, n_va, n_te = _counts_for_class(len(lst))
            print(f"  A{A:03d}: total={len(lst)} train={n_tr} val={n_va} test={n_te}")
            for j, it in enumerate(lst):
                vp = it["vp"]
                if j < n_tr: video_split[vp] = "train"
                elif j < n_tr + n_va: video_split[vp] = "val"
                else: video_split[vp] = "test"

    else:
        # Person-exclusive balanced split
        # Build subject_counts[P][A] as number of videos (selected actions only)
        subject_counts = {}
        for P in subject_pool:
            ca = {A: 0 for A in actions}
            for it in byP[P]:
                A = int(it["meta"]["A"])
                if A in action_to_class:
                    ca[A] += 1
            subject_counts[P] = ca

        splitP, stats = greedy_balanced_subject_split(
            subject_counts=subject_counts,
            actions=actions,
            ratios=(TRAIN_RATIO, VAL_RATIO, TEST_RATIO),
            seed=RANDOM_SEED
        )

        # assign videos by subject
        for P, sp in splitP.items():
            for it in byP[P]:
                A = int(it["meta"]["A"])
                if A in action_to_class:
                    video_split[it["vp"]] = sp

        # print split stats
        subjs, vidsA, vids_total = print_split_stats(splitP, byP, actions)

        # Save subject split list for reproducibility
        split_subjects_path = os.path.join(OUT_DIR, "split_subjects.json")
        with open(split_subjects_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_mode": SPLIT_MODE,
                    "seed": RANDOM_SEED,
                    "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
                    "actions": [int(a) for a in actions],
                    "common_subjects_count": int(len(common_subjects)),
                    "subject_pool_count": int(len(subject_pool)),
                    "subjects": {sp: [int(x) for x in subjs[sp]] for sp in ["train", "val", "test"]},
                    "videos_per_action_per_split": {sp: {f"A{A:03d}": int(vidsA[sp][A]) for A in actions} for sp in ["train","val","test"]},
                    "total_videos_per_split": {sp: int(vids_total[sp]) for sp in ["train","val","test"]},
                },
                f,
                indent=2
            )
        print("Saved subject split:", split_subjects_path)

    def which_split(vp):
        return video_split.get(vp, "train")

    # Load models
    print("Loading YOLO...")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading RTMPose...")
    init_default_scope("mmpose")
    pose_model = init_mmpose_model(RTM_BODY_CFG, RTM_BODY_CKPT, device=DEVICE).eval()

    out_train, out_val, out_test = [], [], []
    debug_saved = 0

    # Shuffle processing order (does not change split)
    rnd = random.Random(RANDOM_SEED)
    items2 = items[:]
    rnd.shuffle(items2)

    for it in tqdm(items2, desc="Building (M=1 combined split)"):
        vp = it["vp"]
        rel = it["rel"]
        meta = it["meta"]

        split = which_split(vp)
        action_id = int(meta["A"])
        label = action_to_class[action_id]

        seq0, seq1, tid0_list, tid1_list, diag0_list, diag1_list, raw_frames = extract_two_seqs_full_with_tid(vp, yolo, pose_model)
        if seq0 is None:
            continue

        chosen_slot, ratio_conf, best0, best1, cam_info = pick_attacker_offline_maxwindow_with_ratio(
            seq0, seq1, action_id,
            diag0_list=diag0_list, diag1_list=diag1_list
        )

        # determine chosen_tid using last frame where chosen slot had a tid
        chosen_tid = None
        if chosen_slot == 0:
            for t in range(len(tid0_list)-1, -1, -1):
                if tid0_list[t] is not None:
                    chosen_tid = int(tid0_list[t]); break
        else:
            for t in range(len(tid1_list)-1, -1, -1):
                if tid1_list[t] is not None:
                    chosen_tid = int(tid1_list[t]); break
        if chosen_tid is None:
            chosen_tid = -1

        # reorder by tid-lock
        if chosen_tid != -1:
            atk_seq, vic_seq = reorder_by_chosen_tid(seq0, seq1, tid0_list, tid1_list, chosen_tid)
        else:
            atk_seq = seq0 if chosen_slot == 0 else seq1
            vic_seq = seq1 if chosen_slot == 0 else seq0

        # debug save
        if DEBUG_SAVE_VIDEOS and debug_saved < DEBUG_MAX_VIDEOS:
            out_mp4 = os.path.join(
                DEBUG_VIDEO_DIR,
                f"{os.path.splitext(os.path.basename(rel))[0]}_A{action_id:03d}_tid{chosen_tid}_r{ratio_conf:.2f}.mp4"
            )
            dbg_frames = []
            Tdbg = min(len(raw_frames), atk_seq.shape[0], vic_seq.shape[0])
            for t in range(Tdbg):
                frame, _, _, _, _ = raw_frames[t]
                dbg_frames.append((frame, atk_seq[t], vic_seq[t]))
            try:
                save_debug_video(dbg_frames, out_mp4, action_id, ratio_conf, chosen_tid, cam_info)
                debug_saved += 1
            except Exception:
                pass

        # Build windows from attacker only
        atk_seq = seq_center_scale_single(atk_seq, kp_conf_th=Q_KP_TH)
        atk_seq = smooth_seq_xy(atk_seq, kp_conf_th=Q_KP_TH, k=3, vel_clip=0.25)

        Tfull = atk_seq.shape[0]
        starts = list(range(0, max(1, Tfull - T_WINDOW + 1), WINDOW_STEP))
        if Tfull >= T_WINDOW and (Tfull - T_WINDOW) not in starts:
            starts.append(Tfull - T_WINDOW)

        for start in starts:
            w = atk_seq[start:start+T_WINDOW]
            if w.shape[0] < T_WINDOW:
                pad = np.zeros((T_WINDOW - w.shape[0], 17, 3), np.float32)
                w = np.concatenate([w, pad], axis=0)

            valid = count_valid_frames(w, kp_conf_th=KP_CONF_TH)
            L = int(max(1, min(valid, T_WINDOW)))

            # quality score
            if QUALITY_ENABLE:
                q, q_torso, q_limb, q_jitter = compute_window_quality_single(w, action_id, kp_conf_th=Q_KP_TH)
                if action_id in AUTO_ATTACKER_ACTIONS and float(ratio_conf) < 1.10:
                    q = max(Q_MIN, q * 0.55)
            else:
                q, q_torso, q_limb, q_jitter = 1.0, 1.0, 1.0, 0.0

            j = w.transpose(2,0,1).astype(np.float32)  # (3,T,17)
            joint_CTVM = j[..., None]                  # (3,T,17,1)

            bone, motion = compute_bone_motion_from_joint_CTV(j)
            bone_CTVM = bone[..., None]
            motion_CTVM = motion[..., None]

            sample = {
                "joint": joint_CTVM.astype(np.float32),
                "bone": bone_CTVM.astype(np.float32),
                "motion": motion_CTVM.astype(np.float32),
                "label": int(label),
                "length": int(L),

                # IMPORTANT: unique and auditable id
                "src": rel,
                "video_id": rel,

                "P": int(meta["P"]),
                "A": int(meta["A"]),
                "S": int(meta["S"]),
                "C": int(meta["C"]),
                "R": int(meta["R"]),

                "chosen_slot": int(chosen_slot),
                "chosen_tid": int(chosen_tid),
                "ratio": float(ratio_conf),

                "q": float(q),
                "q_torso": float(q_torso),
                "q_limb": float(q_limb),
                "q_jitter": float(q_jitter),

                "cam_frac0_big": float(cam_info.get("frac0_big", 0.0)),
                "cam_frac1_big": float(cam_info.get("frac1_big", 0.0)),
                "cam_med_ratio": float(cam_info.get("med_ratio", 1.0)),
            }

            if split == "train": out_train.append(sample)
            elif split == "val": out_val.append(sample)
            else: out_test.append(sample)

    print(f"\nFinal windows: train={len(out_train)}, val={len(out_val)}, test={len(out_test)}")

    with open(os.path.join(OUT_DIR, "train.pkl"), "wb") as f: pickle.dump(out_train, f)
    with open(os.path.join(OUT_DIR, "val.pkl"), "wb") as f: pickle.dump(out_val, f)
    with open(os.path.join(OUT_DIR, "test.pkl"), "wb") as f: pickle.dump(out_test, f)

    if out_train:
        mean, std = compute_masked_mean_std(out_train)
        norm_path = os.path.join(OUT_DIR, f"norm_stats_coco17_2d_T{T_WINDOW}_M1.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump({"mean": mean, "std": std}, f)
        print("Saved norm stats:", norm_path)

    print("Done.")
    if DEBUG_SAVE_VIDEOS:
        print("Debug videos folder:", DEBUG_VIDEO_DIR)

if __name__ == "__main__":
    main()
