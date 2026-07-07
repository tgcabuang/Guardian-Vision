import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict

_FILENAME_RE = re.compile(r"^(\d{8})_(\d{6})\.mp4$", re.IGNORECASE)
_SEGMENT_CACHE: Dict[str, tuple] = {}
_SEGMENT_CACHE_TTL = float(os.environ.get("GV_SEGMENT_CACHE_TTL", "10"))

def parse_segment_start_ts(filename: str) -> Optional[int]:
    """Parse YYYYmmdd_HHMMSS.mp4 -> epoch seconds using local time."""
    m = _FILENAME_RE.match(filename or "")
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return int(time.mktime(dt.timetuple()))
    except Exception:
        return None

def _cache_key(cam_dir: str, segment_seconds: int) -> str:
    return f"{os.path.abspath(cam_dir)}|{int(segment_seconds)}"

def clear_segment_cache() -> None:
    _SEGMENT_CACHE.clear()

def list_segments(cam_dir: str, segment_seconds: int) -> List[Dict[str, int]]:
    """
    List continuous recording segments sorted by ts_start.

    Important fixes:
    - Uses a short cache so old timeline/export requests do not rescan thousands of files each time.
    - Does NOT stretch a segment until the next file if the next file is far away.
      That avoids false green coverage and avoids FFmpeg seeking many hours into a short file.
    """
    segs: List[Dict[str, int]] = []
    if not os.path.isdir(cam_dir):
        return segs

    segment_seconds = max(1, int(segment_seconds or 10))
    max_gap = int(os.environ.get("GV_SEGMENT_MAX_GAP_SEC", str(max(8, segment_seconds * 3))))
    key = _cache_key(cam_dir, segment_seconds)

    try:
        dir_mtime = os.path.getmtime(cam_dir)
    except Exception:
        dir_mtime = 0

    cached = _SEGMENT_CACHE.get(key)
    now = time.time()
    if cached:
        cached_at, cached_mtime, cached_items = cached
        if (now - cached_at) <= _SEGMENT_CACHE_TTL and cached_mtime == dir_mtime:
            return list(cached_items)

    tmp = []
    try:
        names = os.listdir(cam_dir)
    except Exception:
        names = []

    for fn in names:
        if not fn.lower().endswith(".mp4"):
            continue
        path = os.path.join(cam_dir, fn)
        ts0 = parse_segment_start_ts(fn)
        if ts0 is None:
            try:
                ts0 = int(os.path.getmtime(path))
            except Exception:
                continue
        tmp.append({"filename": fn, "ts_start": int(ts0)})

    tmp.sort(key=lambda s: s["ts_start"])

    for i, s in enumerate(tmp):
        ts0 = int(s["ts_start"])
        ts_end = ts0 + segment_seconds
        if i + 1 < len(tmp):
            next_ts = int(tmp[i + 1]["ts_start"])
            gap = next_ts - ts0
            # If next file is close, use its start as the current end.
            # If next file is far away, use only the configured segment length.
            if 0 < gap <= max_gap:
                ts_end = next_ts
        segs.append({
            "filename": s["filename"],
            "ts_start": int(ts0),
            "ts_end": int(ts_end),
        })

    _SEGMENT_CACHE[key] = (now, dir_mtime, tuple(segs))
    return segs

def resolve_ts_to_segment(cam_dir: str, ts: int, segment_seconds: int) -> Optional[Tuple[str, int, int]]:
    """Return (filename, offset_seconds, segment_start_ts) for timestamp ts, or None if no real coverage."""
    target = int(ts)
    for s in list_segments(cam_dir, segment_seconds):
        if int(s["ts_start"]) <= target < int(s["ts_end"]):
            return s["filename"], int(target - int(s["ts_start"])), int(s["ts_start"])
    return None
