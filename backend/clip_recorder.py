# clip_recorder.py
import os, time, json, threading, re
from datetime import datetime

TIME_OFFSET_SEC = int(os.environ.get("GV_TIME_OFFSET_SEC", "4").strip() or "4")

def offset_ts(ts=None):
    base = time.time() if ts is None else float(ts)
    return base + TIME_OFFSET_SEC

def offset_stamp(ts=None):
    return datetime.fromtimestamp(offset_ts(ts)).strftime("%Y%m%d_%H%M%S")

def offset_created_text(ts=None):
    return datetime.fromtimestamp(offset_ts(ts)).strftime("%Y-%m-%d %H:%M:%S")

from dataclasses import dataclass, field
from collections import deque
import cv2
import numpy as np


# ----------------------------
# Action mapping / triggers
# ----------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _is_a050_like(t: str) -> bool:
    # Recognize: A050, A0050, NTU_A050, NTU_A0050, etc + "punch"
    return bool(re.search(r"\ba0*50\b", t)) or ("ntu_a" in t and "50" in t) or ("punch" in t)


def _is_a051_like(t: str) -> bool:
    # Recognize: A051, A0051, NTU_A051, NTU_A0051, etc + "kick"
    return bool(re.search(r"\ba0*51\b", t)) or ("ntu_a" in t and "51" in t) or ("kick" in t)


def map_action(label: str):
    """
    Returns (folder_name, canon_label, action_id) or (None, None, None) if not a trigger.

    REQUIREMENT:
      - A050 clips go to folder "Punching"
      - A051 clips go to folder "Kicking"
    """
    t = _norm(label)

    if _is_a050_like(t):
        return "Punching", "Punching", "A050"
    if _is_a051_like(t):
        return "Kicking", "Kicking", "A051"

    return None, None, None


@dataclass
class CamBuffer:
    frames: deque = field(default_factory=lambda: deque(maxlen=9999))  # stores (ts, frame_small)
    active: bool = False
    until_ts: float = 0.0

    # event info
    event_label_raw: str = ""
    event_label_canon: str = ""
    event_folder: str = ""
    event_action_id: str = ""

    event_conf: float = 0.0
    cooldown_until: float = 0.0
    collect: list = field(default_factory=list)  # list of (ts, frame_small)


class ClipRecorder:
    """
    Usage:
      - push_frame(cam, frame_bgr, ts)
      - on_action(cam, label, conf, ts)  -> arms recording if label matches A050/A051 mapping

    New:
      - on_event callback: called after a clip is successfully saved.
        Signature: on_event(dict) -> None
    """

    def __init__(
        self,
        out_dir="records",
        fps=15,
        pre_sec=2,     # CHANGED DEFAULT: 2s pre
        post_sec=5,    # CHANGED DEFAULT: 5s post
        min_conf=0.85,
        cooldown_sec=2.0,
        target_wh=(640, 360),
        on_event=None,
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.fps = float(fps)
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.min_conf = float(min_conf)
        self.cooldown_sec = float(cooldown_sec)
        self.target_wh = target_wh

        self.on_event = on_event  # optional callback

        self.lock = threading.Lock()
        self.cams = {0: CamBuffer(), 1: CamBuffer(), 2: CamBuffer()}

    def _shrink(self, frame_bgr):
        if self.target_wh is None:
            return frame_bgr
        w, h = self.target_wh
        return cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

    def push_frame(self, cam: int, frame_bgr: np.ndarray, ts: float):
        fr = self._shrink(frame_bgr)
        with self.lock:
            cb = self.cams.get(cam)
            if cb is None:
                cb = CamBuffer()
                self.cams[cam] = cb

            cb.frames.append((ts, fr))

            # If recording active, collect frames until deadline
            if cb.active:
                cb.collect.append((ts, fr))
                if ts >= cb.until_ts:
                    frames_to_write = cb.collect[:]
                    label_raw = cb.event_label_raw
                    label_canon = cb.event_label_canon
                    folder = cb.event_folder
                    action_id = cb.event_action_id
                    conf = cb.event_conf

                    cb.active = False
                    cb.collect.clear()
                    cb.cooldown_until = ts + self.cooldown_sec

                    # write on a separate thread so detection loop stays fast
                    threading.Thread(
                        target=self._write_clip,
                        args=(cam, frames_to_write, label_raw, label_canon, folder, action_id, conf),
                        daemon=True,
                    ).start()

            # prune old frames beyond pre_sec
            cutoff = ts - self.pre_sec
            while cb.frames and cb.frames[0][0] < cutoff:
                cb.frames.popleft()

    def on_action(self, cam: int, label: str, conf: float, ts: float):
        label_raw = (label or "").strip()
        conf = float(conf or 0.0)

        folder, canon, action_id = map_action(label_raw)
        if folder is None:
            return
        if conf < self.min_conf:
            return

        with self.lock:
            cb = self.cams.get(cam)
            if cb is None:
                cb = CamBuffer()
                self.cams[cam] = cb

            if cb.active:
                return
            if ts < cb.cooldown_until:
                return

            # start event: take pre-buffer frames + start collecting
            cb.active = True
            cb.until_ts = ts + self.post_sec

            cb.event_label_raw = label_raw
            cb.event_label_canon = canon        # "Punching"/"Kicking"
            cb.event_folder = folder            # folder name
            cb.event_action_id = action_id      # "A050"/"A051"
            cb.event_conf = conf

            cb.collect = list(cb.frames)  # pre-buffer

    def _write_clip(self, cam: int, frames_ts, label_raw: str, label_canon: str,
                    folder: str, action_id: str, conf: float):
        if not frames_ts:
            return

        frames = [f for (_, f) in frames_ts]
        h, w = frames[0].shape[:2]

        # filename: include action_id always (A050/A051)
        tstamp = offset_stamp()
        fname = f"cam{cam}_{tstamp}_{action_id}_{int(conf * 100)}.mp4"

        if folder not in ("Punching", "Kicking") or action_id not in ("A050", "A051"):
            return

        # action folder: Punching / Kicking
        clip_dir = os.path.join(self.out_dir, folder)
        os.makedirs(clip_dir, exist_ok=True)
        out_path = os.path.join(clip_dir, fname)

        vw = None
        for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            vw = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
            if vw.isOpened():
                break

        if vw is None or not vw.isOpened():
            return

        for fr in frames:
            vw.write(fr)
        vw.release()

        # derive clip time span (epoch seconds)
        try:
            ts_start = offset_ts(frames_ts[0][0])
            ts_end = offset_ts(frames_ts[-1][0])
        except Exception:
            ts_start = offset_ts()
            ts_end = ts_start

        # sidecar metadata
        rel = os.path.relpath(out_path, self.out_dir).replace("\\", "/")
        meta = {
            "file": rel,                  # e.g. "Punching/cam0_....mp4"
            "cam": cam,
            "folder": folder,             # "Punching" / "Kicking"
            "action_id": action_id,       # "A050" / "A051"
            "label": label_canon,         # "Punching" / "Kicking" (for UI)
            "label_raw": label_raw,       # original model output (could be ntu_a0050)
            "conf": float(conf),
            "created_ts": offset_ts(),
            "created": offset_created_text(),
            "fps": float(self.fps),
            "w": int(w), "h": int(h),
            "num_frames": int(len(frames)),
            "clip_ts_start": int(ts_start),
            "clip_ts_end": int(ts_end),
        }
        with open(out_path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # optional callback (DB logging, timeline UI, etc.)
        if callable(self.on_event):
            try:
                self.on_event({
                    "cam": int(cam),
                    "label": str(label_canon),
                    "conf": float(conf),
                    "ts_start": int(ts_start),
                    "ts_end": int(ts_end),
                    "clip_rel": rel,
                    "action_id": str(action_id),
                })
            except Exception:
                pass
