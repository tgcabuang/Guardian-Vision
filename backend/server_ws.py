# backend/server_ws.py
import asyncio
import os
import sys
import glob
import json
import hmac
import hashlib
import sqlite3
import secrets
import threading
import time
import shutil
import csv
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from collections import Counter
from openpyxl import Workbook, load_workbook

from fastapi import FastAPI, WebSocket, Request, Depends, HTTPException, Header, Query
from starlette.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# =========================================================
# Host/Port
# =========================================================
HOST = os.environ.get("GV_HOST", "127.0.0.1").strip()
PORT = int(os.environ.get("GV_PORT", "8766").strip())

app = FastAPI()

TIME_OFFSET_SEC = int(os.environ.get("GV_TIME_OFFSET_SEC", "4").strip() or "4")

def offset_datetime(dt: datetime) -> datetime:
    return dt + timedelta(seconds=TIME_OFFSET_SEC)

def now_offset() -> datetime:
    return offset_datetime(datetime.now())

def timestamp_offset(ts: Optional[Union[float, int]]) -> int:
    if ts is None:
        return int(time.time()) + TIME_OFFSET_SEC
    try:
        return int(float(ts) + TIME_OFFSET_SEC)
    except Exception:
        return int(time.time()) + TIME_OFFSET_SEC


# =========================================================
# Packaging-safe paths
# =========================================================
def app_root() -> str:
    # Nuitka / PyInstaller-safe: sys.argv[0] points to exe location
    try:
        p = os.path.abspath(os.path.dirname(sys.argv[0]))
        if p and os.path.isdir(p):
            return p
    except Exception:
        pass
    return os.path.abspath(os.path.dirname(__file__))

APP_ROOT = app_root()

def pjoin(*parts) -> str:
    return os.path.join(APP_ROOT, *parts)

# Allow overriding root via env (useful for packaged builds)
if os.environ.get("GV_APP_ROOT", "").strip():
    APP_ROOT = os.environ["GV_APP_ROOT"].strip().strip('"')

# =========================================================
# Records dir
# =========================================================
def _read_records_dir_from_cfg(cfg_path: str) -> str:
    try:
        if not cfg_path:
            return ""
        if not os.path.exists(cfg_path):
            return ""
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        d = (cfg.get("recordsDir") or "").strip()
        return d
    except Exception:
        return ""

def resolve_record_dir():
    env_dir = os.environ.get("GV_RECORD_DIR", "").strip().strip('"')
    if env_dir:
        return os.path.abspath(env_dir)

    env_cfg_path = os.environ.get("GV_CONFIG_PATH", "").strip().strip('"')
    cfg_dir = _read_records_dir_from_cfg(env_cfg_path)
    if cfg_dir:
        return os.path.abspath(cfg_dir)

    bundled_cfg = os.path.join(APP_ROOT, "gv_config.json")
    cfg_dir = _read_records_dir_from_cfg(bundled_cfg)
    if cfg_dir:
        return os.path.abspath(cfg_dir)

    # safer fallback, not inside backend folder
    return os.path.join(os.path.expanduser("~"), "Videos", "GuardianVision")

RECORD_DIR = resolve_record_dir()
print("APP_ROOT =", APP_ROOT)
print("RECORD_DIR =", RECORD_DIR)
print("CFG_PATH =", os.path.join(APP_ROOT, "gv_config.json"))
os.makedirs(RECORD_DIR, exist_ok=True)

# =========================================================
# Tabular Logging (per server run)
# - Creates CSV + XLSX files each time the server starts.
# - Appends one row per recorded event / action sample.
# - Captures every visible track, not only the top track.
# =========================================================
CSV_LOCK = threading.Lock()
CSV_LOG_PATH = None
XLSX_LOG_PATH = None

LOG_HEADERS = [
    "date",
    "time",
    "cam",
    "track_id",
    "action",
    "confidence",
]

def _xlsx_autofit(ws):
    try:
        for col in ws.columns:
            max_len = 0
            letter = col[0].column_letter
            for cell in col:
                try:
                    max_len = max(max_len, len(str(cell.value or "")))
                except Exception:
                    pass
            ws.column_dimensions[letter].width = min(max(12, max_len + 2), 28)
    except Exception:
        pass

def _csv_init_log():
    """Create a new CSV log file for this server run."""
    global CSV_LOG_PATH, XLSX_LOG_PATH
    try:
        log_dir = os.path.join(RECORD_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)

        stamp = now_offset().strftime("%Y%m%d_%H%M%S")
        CSV_LOG_PATH = os.path.join(log_dir, f"detections_{stamp}.csv")
        XLSX_LOG_PATH = None  # keep runtime logging CSV-only for better FPS

        with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)
    except Exception:
        CSV_LOG_PATH = None
        XLSX_LOG_PATH = None

def _append_tabular_row(row: List[Any]):
    if not CSV_LOG_PATH:
        return
    try:
        with CSV_LOCK:
            with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
    except Exception:
        return

def _csv_append_event(evt: Dict[str, Any]):
    if not CSV_LOG_PATH:
        return
    try:
        cam0 = int(evt.get("cam", 0))
        cam_disp = cam0 + 1
        label = str(evt.get("label", "")).strip()
        conf = float(evt.get("conf", 0.0) or 0.0)
        track_id = int(evt.get("track_id", -1) or -1)
        ts0 = evt.get("ts_start", None)
        try:
            dt = offset_datetime(datetime.fromtimestamp(int(ts0))) if ts0 is not None else now_offset()
        except Exception:
            dt = now_offset()

        if not label:
            return

        row = [
            dt.strftime("%Y-%m-%d"),
            dt.strftime("%H:%M:%S"),
            f"Cam {cam_disp}",
            track_id if track_id >= 0 else "",
            label,
            f"{conf:.4f}",
        ]
        _append_tabular_row(row)
    except Exception:
        return

# =========================================================
# WS Continuous Action Logging
# - Logs every visible track once per interval (default: 1 second).
# - CSV-only during runtime to minimize FPS impact.
# =========================================================
WS_LOG_ENABLED = True
WS_LOG_MIN_INTERVAL = float(os.environ.get('GV_WS_LOG_MIN_INTERVAL', '1.00'))
WS_LOG_ONLY_CONNECTED = True
WS_LOG_INCLUDE_UNKNOWN = os.environ.get('GV_WS_LOG_INCLUDE_UNKNOWN', '1').strip() == '1'
WS_LOG_INCLUDE_NO_TRACKS = os.environ.get('GV_WS_LOG_INCLUDE_NO_TRACKS', '0').strip() == '1'

_WS_LAST_LOG: Dict[tuple, Dict[str, Any]] = {}

def _csv_append_action_row(cam_idx: int, track_id: Optional[int], label: str, conf: float):
    if not CSV_LOG_PATH:
        return
    try:
        cam_disp = int(cam_idx) + 1
        label = str(label or '').strip()
        if label == '':
            return
        conf = float(conf or 0.0)
        dt = now_offset()
        row = [
            dt.strftime('%Y-%m-%d'),
            dt.strftime('%H:%M:%S'),
            f'Cam {cam_disp}',
            '' if track_id is None else int(track_id),
            label,
            f'{conf:.4f}',
        ]
        _append_tabular_row(row)
    except Exception:
        return
def _maybe_log_from_ws_packet(pkt: Dict[str, Any]):
    if not WS_LOG_ENABLED or not CSV_LOG_PATH or not pkt:
        return
    try:
        now_s = time.time()
        packet_id = pkt.get('id')
        for cam_i in range(3):
            cam_obj = pkt.get(f'cam{cam_i}') or {}
            connected = bool(cam_obj.get('connected', False))
            if WS_LOG_ONLY_CONNECTED and not connected:
                continue

            ts_ms = int(cam_obj.get('ts') or pkt.get('ts') or cam_obj.get('last_ts_ms') or 0) or None
            tracks = cam_obj.get('tracks') or []

            if not tracks:
                if WS_LOG_INCLUDE_NO_TRACKS:
                    key = (cam_i, 'NO_TRACKS')
                    prev = _WS_LAST_LOG.get(key)
                    if prev is None or (now_s - float(prev.get('t', 0.0))) >= WS_LOG_MIN_INTERVAL:
                        _WS_LAST_LOG[key] = {'t': now_s}
                        _csv_append_action_row(cam_i, None, 'NO_TRACKS', 0.0)
                continue

            for tr in tracks:
                try:
                    track_id = int(tr.get('tid')) if tr.get('tid') is not None else None
                except Exception:
                    track_id = None
                label = str(tr.get('label', '')).strip()
                conf = float(tr.get('conf', 0.0) or 0.0)

                if not label:
                    continue
                if (not WS_LOG_INCLUDE_UNKNOWN) and label.upper() == 'UNKNOWN':
                    continue

                key = (cam_i, track_id if track_id is not None else f'anon_{label}')
                prev = _WS_LAST_LOG.get(key)
                if prev is None or (now_s - float(prev.get('t', 0.0))) >= WS_LOG_MIN_INTERVAL:
                    _WS_LAST_LOG[key] = {'t': now_s}
                    _csv_append_action_row(cam_i, track_id, label, conf)
    except Exception:
        return

app.mount("/records", StaticFiles(directory=RECORD_DIR), name="records")

# Continuous recordings (for timeline playback)
CONTINUOUS_DIR = os.path.join(RECORD_DIR, "continuous")
os.makedirs(CONTINUOUS_DIR, exist_ok=True)
app.mount("/continuous", StaticFiles(directory=CONTINUOUS_DIR), name="continuous")

# =========================================================
# CORS (Electron file:// can call localhost)
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local app; acceptable for localhost thesis
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Globals (set in startup)
# =========================================================
runner = None
continuous_mgr = None
recorder = None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
VIOLENT_LABELS = {"punching", "kicking"}

# =========================================================
# AUTH DB (SQLite) - stdlib only
# =========================================================
DB_PATH = os.path.join(APP_ROOT, "gv_users.db")
_DB_LOCK = threading.Lock()

PBKDF2_ITERS = 150_000
RECOVERY_ITERS = 120_000

# Recovery security: limit brute-force attempts per IP/client.
RECOVERY_MAX_FAILS = 5
RECOVERY_WINDOW_SEC = 15 * 60
_RECOVERY_FAILS: Dict[str, List[float]] = {}

# =========================================================
# TOKEN EXPIRY CONTROL
# - Set GV_TOKEN_TTL_SEC=0   -> NEVER expires (default in your file)
# - Set GV_TOKEN_TTL_SEC=43200 -> 12 hours
# =========================================================
TOKEN_TTL_SEC = int(os.environ.get("GV_TOKEN_TTL_SEC", "0").strip() or "0")
FAR_FUTURE_ISO = "9999-12-31T23:59:59Z"

def _db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _pbkdf2_hash(password: str, salt: bytes, iters: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)

def _random_recovery_code() -> str:
    # 16 chars from a no-confusing alphabet. Hyphens are display-only.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    raw = "".join(secrets.choice(alphabet) for _ in range(16))
    return "-".join(raw[i:i+4] for i in range(0, len(raw), 4))

def _normalize_recovery_code(code: str) -> str:
    # Accept pasted codes with spaces/hyphens, but verify only normalized characters.
    return "".join(ch for ch in (code or "").strip().upper() if ch.isalnum())

def _recovery_rate_key(request: Request, client_id: Optional[str], email: str = "") -> str:
    ip = request.client.host if request and request.client else "unknown-ip"
    return f"{ip}|{client_id or ''}|{email.lower().strip()}"

def _check_recovery_rate_limit(key: str):
    now = time.time()
    fails = [t for t in _RECOVERY_FAILS.get(key, []) if now - t < RECOVERY_WINDOW_SEC]
    _RECOVERY_FAILS[key] = fails
    if len(fails) >= RECOVERY_MAX_FAILS:
        raise HTTPException(status_code=429, detail="Too many recovery attempts. Please wait a few minutes and try again.")

def _record_recovery_fail(key: str):
    now = time.time()
    fails = [t for t in _RECOVERY_FAILS.get(key, []) if now - t < RECOVERY_WINDOW_SEC]
    fails.append(now)
    _RECOVERY_FAILS[key] = fails

def _clear_recovery_fails(key: str):
    _RECOVERY_FAILS.pop(key, None)

def _mask_email(email: str) -> str:
    try:
        name, domain = email.split("@", 1)
        if len(name) <= 2:
            masked = name[0] + "*"
        else:
            masked = name[0] + "*" * max(1, len(name) - 2) + name[-1]
        return masked + "@" + domain
    except Exception:
        return email

def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    return col in cols

def _db_init_and_migrate():
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                pw_salt BLOB NOT NULL,
                pw_hash BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                client_id TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_seen TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_id INTEGER,
                action TEXT NOT NULL,
                client_id TEXT,
                ip TEXT,
                details TEXT
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS gv_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start INTEGER NOT NULL,
                ts_end INTEGER NOT NULL,
                cam INTEGER NOT NULL,
                label TEXT NOT NULL,
                conf REAL,
                action_id TEXT,
                clip_rel TEXT,
                created_at TEXT NOT NULL
            );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_gv_events_cam_ts ON gv_events (cam, ts_start);")

            conn.commit()

            # migrations
            if not _table_has_column(conn, "users", "must_change_pw"):
                cur.execute("ALTER TABLE users ADD COLUMN must_change_pw INTEGER NOT NULL DEFAULT 0;")
            if not _table_has_column(conn, "users", "recovery_salt"):
                cur.execute("ALTER TABLE users ADD COLUMN recovery_salt BLOB;")
            if not _table_has_column(conn, "users", "recovery_hash"):
                cur.execute("ALTER TABLE users ADD COLUMN recovery_hash BLOB;")

            conn.commit()
        finally:
            conn.close()

def _audit(request_ip: Optional[str], user_id: Optional[int], action: str, client_id: Optional[str], details: Optional[dict]):
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO audit_log (ts, user_id, action, client_id, ip, details) VALUES (?,?,?,?,?,?)",
                (_now_iso(), user_id, action, (client_id or None), (request_ip or None), json.dumps(details or {}, ensure_ascii=False))
            )
            conn.commit()
        finally:
            conn.close()

# =========================================================
# Timeline events (DB)
# =========================================================
def _timeline_insert_event(evt: Dict[str, Any]):
    try:
        cam = int(evt.get("cam", 0))
        ts_start = int(evt.get("ts_start"))
        ts_end = int(evt.get("ts_end"))
        label = str(evt.get("label", "")).strip()
        conf = float(evt.get("conf", 0.0) or 0.0)
        action_id = (evt.get("action_id") or None)
        clip_rel = (evt.get("clip_rel") or None)
    except Exception:
        return

    if not label:
        return
    if ts_end < ts_start:
        ts_start, ts_end = ts_end, ts_start

    merged_existing = False
    # Merge repeated detections into one event duration.
    # Example: Punching detected every second should appear as ONE row whose
    # ts_end keeps extending until the action stops, instead of many rows.
    MERGE_GAP_SEC = 8

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, ts_start, ts_end, conf
                   FROM gv_events
                   WHERE cam=? AND label=? AND clip_rel IS NULL
                   ORDER BY ts_end DESC
                   LIMIT 1""",
                (cam, label),
            )
            last = cur.fetchone()

            if last is not None and int(last["ts_end"] or 0) >= int(ts_start) - MERGE_GAP_SEC:
                row_id = int(last["id"])
                new_start = min(int(last["ts_start"] or ts_start), ts_start)
                new_end = max(int(last["ts_end"] or ts_end), ts_end)
                new_conf = max(float(last["conf"] or 0.0), conf)
                cur.execute(
                    """UPDATE gv_events
                       SET ts_start=?, ts_end=?, conf=?, action_id=?, created_at=?
                       WHERE id=?""",
                    (new_start, new_end, new_conf, action_id, _now_iso(), row_id),
                )
                merged_existing = True
            else:
                cur.execute(
                    "INSERT INTO gv_events (ts_start, ts_end, cam, label, conf, action_id, clip_rel, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (ts_start, ts_end, cam, label, conf, action_id, clip_rel, _now_iso())
                )
            conn.commit()
        finally:
            conn.close()

    # Only append brand-new events to CSV. Continuous extensions should not spam logs.
    if not merged_existing:
        _csv_append_event(evt)


class TimelineEventMarkerRecorder:
    """
    Records detection events only.
    It does NOT save automatic clips.
    Continuous video is still recorded by ContinuousRecordingManager;
    users manually export any selected range from the timeline.

    Timeline event duration is based on the actual detection duration:
    repeated Punching/Kicking detections are merged by _timeline_insert_event.
    Example: if Punching is detected for about 10 seconds, the timeline stamp
    becomes about 10 seconds, not a forced 1-minute marker.
    """
    def __init__(self, on_event=None, cooldown_sec: float = 1.0, event_len_sec: int = 1):
        self.on_event = on_event
        self.cooldown_sec = float(cooldown_sec)
        self.event_len_sec = max(1, int(event_len_sec))
        self._last = {}

    def push_frame(self, cam: int, frame, ts: float):
        # No frame buffering here. This intentionally prevents auto clip saving.
        return None

    def on_action(self, cam: int, label: str, conf: float, ts: float):
        label = str(label or "").strip()
        if label not in ("Punching", "Kicking", "A050", "A051"):
            return

        canon = {"A050": "Punching", "A051": "Kicking"}.get(label, label)
        now = float(ts or time.time())
        key = (int(cam), canon)
        last = float(self._last.get(key, 0.0) or 0.0)
        if (now - last) < self.cooldown_sec:
            return
        self._last[key] = now

        start_ts = int(max(0, now))
        end_ts = int(max(start_ts + 1, now + self.event_len_sec))
        evt = {
            "cam": int(cam),
            "label": canon,
            "conf": float(conf or 0.0),
            "ts_start": start_ts,
            "ts_end": end_ts,
            "action_id": "A050" if canon == "Punching" else "A051",
            "clip_rel": None,
            "timeline_only": True,
        }
        if self.on_event:
            try:
                self.on_event(evt)
            except Exception:
                pass

# Duplicate WS logging block removed; see primary definition above.

def _create_user(email: str, full_name: str, role: str, password: str, must_change_pw: int = 0) -> Dict[str, Any]:
    salt = secrets.token_bytes(16)
    pw_hash = _pbkdf2_hash(password, salt, PBKDF2_ITERS)

    rec_code = _random_recovery_code()
    rec_salt = secrets.token_bytes(16)
    rec_hash = _pbkdf2_hash(_normalize_recovery_code(rec_code), rec_salt, RECOVERY_ITERS)

    now = _now_iso()
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO users
                   (email, full_name, role, pw_salt, pw_hash, created_at, updated_at, must_change_pw, recovery_salt, recovery_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (email.lower().strip(), full_name.strip(), role.strip(), salt, pw_hash, now, now, int(must_change_pw), rec_salt, rec_hash)
            )
            conn.commit()
            user_id = int(cur.lastrowid)
        finally:
            conn.close()

    return {"id": user_id, "email": email.lower().strip(), "recovery_code": rec_code}

def _get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

def _get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?", (int(user_id),))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

def _verify_password(user_row: Dict[str, Any], password: str) -> bool:
    salt = user_row["pw_salt"]
    expected = user_row["pw_hash"]
    got = _pbkdf2_hash(password, salt, PBKDF2_ITERS)
    return hmac.compare_digest(expected, got)

def _verify_recovery(user_row: Dict[str, Any], recovery_code: str) -> bool:
    salt = user_row.get("recovery_salt")
    expected = user_row.get("recovery_hash")
    if not salt or not expected:
        return False
    got = _pbkdf2_hash(_normalize_recovery_code(recovery_code), salt, RECOVERY_ITERS)
    return hmac.compare_digest(expected, got)

def _find_user_by_recovery_code(recovery_code: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_recovery_code(recovery_code)
    if not normalized:
        return None
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users")
            rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    for row in rows:
        if _verify_recovery(row, normalized):
            return row
    return None

def _set_password(user_id: int, new_password: str, must_change_pw: int = 0):
    salt = secrets.token_bytes(16)
    pw_hash = _pbkdf2_hash(new_password, salt, PBKDF2_ITERS)
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET pw_salt=?, pw_hash=?, must_change_pw=?, updated_at=? WHERE id=?",
                (salt, pw_hash, int(must_change_pw), _now_iso(), int(user_id))
            )
            conn.commit()
        finally:
            conn.close()

def _clear_must_change(user_id: int):
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET must_change_pw=0, updated_at=? WHERE id=?",
                (_now_iso(), int(user_id))
            )
            conn.commit()
        finally:
            conn.close()

def _rotate_recovery_code(user_id: int) -> str:
    rec_code = _random_recovery_code()
    rec_salt = secrets.token_bytes(16)
    rec_hash = _pbkdf2_hash(_normalize_recovery_code(rec_code), rec_salt, RECOVERY_ITERS)
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET recovery_salt=?, recovery_hash=?, updated_at=? WHERE id=?",
                (rec_salt, rec_hash, _now_iso(), int(user_id))
            )
            conn.commit()
        finally:
            conn.close()
    return rec_code

def _issue_token(user_id: int, client_id: Optional[str]) -> str:
    token = secrets.token_urlsafe(32)
    now_iso = _now_iso()

    if TOKEN_TTL_SEC and TOKEN_TTL_SEC > 0:
        expires_dt = datetime.utcnow() + timedelta(seconds=int(TOKEN_TTL_SEC))
        exp_iso = expires_dt.isoformat(timespec="seconds") + "Z"
    else:
        exp_iso = FAR_FUTURE_ISO

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO sessions (token, user_id, client_id, created_at, expires_at, last_seen) VALUES (?,?,?,?,?,?)",
                (token, int(user_id), (client_id or "").strip() or None, now_iso, exp_iso, now_iso)
            )
            conn.commit()
        finally:
            conn.close()

    return token

def _delete_token(token: str):
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()

def _get_session(token: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM sessions WHERE token = ?", (token,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

def _touch_session(token: str):
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("UPDATE sessions SET last_seen = ? WHERE token = ?", (_now_iso(), token))
            conn.commit()
        finally:
            conn.close()

def _ensure_default_admin():
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM users")
            n = int(cur.fetchone()["n"])
        finally:
            conn.close()

    if n == 0:
        default_email = os.environ.get("GV_DEFAULT_EMAIL", "admin@gv.com")
        default_password = os.environ.get("GV_DEFAULT_PASSWORD", "pocartisweat")
        created = _create_user(
            email=default_email,
            full_name="System Administrator",
            role="System Administrator",
            password=default_password
        )
        print(f"[auth] Created default admin user: {default_email} / {default_password}")
        print(f"[auth] Admin RECOVERY CODE (save this!): {created['recovery_code']}")

# =========================================================
# Auth dependency (HTTP)
#   PATCH MERGE: allow ?token=... for browser + Electron tests
# =========================================================
def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None

async def require_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_client_id: Optional[str] = Header(None),
    token_q: Optional[str] = Query(None, alias="token"),
) -> Dict[str, Any]:
    token = _extract_bearer(authorization) or ((token_q or "").strip() or None)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    sess = _get_session(token)
    if not sess:
        raise HTTPException(status_code=401, detail="Invalid token")

    if TOKEN_TTL_SEC and TOKEN_TTL_SEC > 0:
        try:
            exp = datetime.fromisoformat(str(sess["expires_at"]).replace("Z", ""))
            if datetime.utcnow() > exp:
                _delete_token(token)
                raise HTTPException(status_code=401, detail="Token expired")
        except HTTPException:
            raise
        except Exception:
            _delete_token(token)
            raise HTTPException(status_code=401, detail="Invalid session")

    user = _get_user_by_id(int(sess["user_id"]))
    if not user:
        _delete_token(token)
        raise HTTPException(status_code=401, detail="User not found")

    _touch_session(token)
    request.state.auth_token = token
    request.state.client_id = x_client_id or sess.get("client_id")
    return user

def _is_admin(user: Dict[str, Any]) -> bool:
    return str(user.get("role") or "").strip().lower() in ["system administrator", "admin", "administrator"]

async def require_admin(request: Request, user=Depends(require_user)) -> Dict[str, Any]:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin only")
    return user

# =========================================================
# Asset fix for packaging
# =========================================================
def _fix_assets_for_packaging():
    try:
        assets_dir = os.environ.get("GV_ASSETS_DIR", "").strip().strip('"')
        if not assets_dir:
            assets_dir = os.path.join(APP_ROOT, "assets")

        if not os.path.isdir(assets_dir):
            return

        bad = os.path.join(assets_dir, "rtmpose_cfg.py.py")
        good = os.path.join(assets_dir, "rtmpose_cfg.py")

        if os.path.isfile(bad) and not os.path.isfile(good):
            shutil.copyfile(bad, good)
            print("[assets] Fixed rtmpose_cfg.py.py -> rtmpose_cfg.py")
    except Exception:
        pass

# =========================================================
# Startup/shutdown
# =========================================================
@app.on_event("startup")
def _startup():
    global runner, continuous_mgr, recorder
    
    _db_init_and_migrate()
    _ensure_default_admin()

    _fix_assets_for_packaging()

    # init per-run CSV log
    _csv_init_log()

    # delayed imports (safer when packaged)
    from detection_core import MultiCamDetectionRunner
    from continuous_recorder import ContinuousRecordingManager
    from timeline_utils import list_segments, resolve_ts_to_segment

    # Timeline-only marker recorder:
    # Violent actions create red/yellow timeline stamps only.
    # No automatic 10-second Punching/Kicking clip is saved.
    recorder = TimelineEventMarkerRecorder(
        on_event=_timeline_insert_event,
        cooldown_sec=float(os.environ.get("GV_TIMELINE_EVENT_COOLDOWN_SEC", "2")),
        event_len_sec=int(os.environ.get("GV_TIMELINE_EVENT_LEN_SEC", "1")),
    )

    # Continuous recording source used for manual timeline export.
    try:
        segment_sec = int(os.environ.get("GV_SEGMENT_SEC", "60"))
        keep_hours = int(os.environ.get("GV_KEEP_HOURS", str(int(os.environ.get("GV_KEEP_DAYS", "30")) * 24)))  # default 30 days
        continuous_mgr = ContinuousRecordingManager(
            num_cams=3,
            root_dir=CONTINUOUS_DIR,
            segment_seconds=segment_sec,
            keep_hours=keep_hours,
        )
    except Exception:
        continuous_mgr = None

    runner = MultiCamDetectionRunner(num_cams=3, recorder=recorder)
    runner.start()

@app.on_event("shutdown")
def _shutdown():
    global runner, continuous_mgr
    try:
        if runner:
            runner.stop()
    except Exception:
        pass
    try:
        if continuous_mgr:
            continuous_mgr.stop_all()
    except Exception:
        pass

# =========================================================
# Auth routes
# =========================================================
@app.post("/auth/login")
async def auth_login(payload: dict, request: Request, x_client_id: Optional[str] = Header(None)):
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    user = _get_user_by_email(email)
    if not user or not _verify_password(user, password):
        _audit(request.client.host if request.client else None, None, "login_failed", x_client_id, {"email": email})
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _issue_token(int(user["id"]), x_client_id)
    _audit(request.client.host if request.client else None, int(user["id"]), "login_ok", x_client_id, {})

    return {
        "ok": True,
        "token": token,
        "must_change_pw": int(user.get("must_change_pw") or 0),
        "user": {
            "id": int(user["id"]),
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        }
    }

@app.post("/auth/logout")
async def auth_logout(request: Request, user=Depends(require_user)):
    token = getattr(request.state, "auth_token", None)
    if token:
        _delete_token(token)
    _audit(request.client.host if request.client else None, int(user["id"]), "logout", getattr(request.state, "client_id", None), {})
    return {"ok": True}

@app.get("/auth/me")
async def auth_me(request: Request, user=Depends(require_user)):
    return {
        "ok": True,
        "must_change_pw": int(user.get("must_change_pw") or 0),
        "user": {
            "id": int(user["id"]),
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        }
    }

@app.put("/auth/profile")
async def auth_update_profile(payload: dict, request: Request, user=Depends(require_user)):
    new_full_name = (payload.get("full_name") or "").strip()
    new_email = (payload.get("email") or "").strip().lower()
    new_role = (payload.get("role") or "").strip()

    if not new_full_name:
        raise HTTPException(status_code=400, detail="Full name required")
    if not new_email or "@" not in new_email:
        raise HTTPException(status_code=400, detail="Valid email required")
    if not new_role:
        new_role = user["role"]

    if new_email != user["email"]:
        existing = _get_user_by_email(new_email)
        if existing and int(existing["id"]) != int(user["id"]):
            raise HTTPException(status_code=409, detail="Email already in use")

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET email=?, full_name=?, role=?, updated_at=? WHERE id=?",
                (new_email, new_full_name, new_role, _now_iso(), int(user["id"]))
            )
            conn.commit()
        finally:
            conn.close()

    _audit(request.client.host if request.client else None, int(user["id"]), "profile_update",
           getattr(request.state, "client_id", None),
           {"email": new_email, "full_name": new_full_name, "role": new_role})

    return {"ok": True}

@app.put("/auth/change_password")
async def auth_change_password(payload: dict, request: Request, user=Depends(require_user)):
    current_password = payload.get("current_password") or ""
    new_password = payload.get("new_password") or ""

    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="Current and new password required")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    full_user = _get_user_by_id(int(user["id"]))
    if not full_user or not _verify_password(full_user, current_password):
        _audit(request.client.host if request.client else None, int(user["id"]), "password_change_failed",
               getattr(request.state, "client_id", None), {})
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    _set_password(int(user["id"]), new_password, must_change_pw=0)
    _audit(request.client.host if request.client else None, int(user["id"]), "password_change_ok",
           getattr(request.state, "client_id", None), {})

    return {"ok": True}

@app.put("/auth/recover_reset")
async def auth_recover_reset(payload: dict, request: Request, x_client_id: Optional[str] = Header(None)):
    # Email is optional. If the user forgot the email too, the recovery code can identify the account.
    email = (payload.get("email") or "").strip().lower()
    recovery_code = _normalize_recovery_code(payload.get("recovery_code") or "")
    new_password = (payload.get("new_password") or "")

    rate_key = _recovery_rate_key(request, x_client_id, email)
    _check_recovery_rate_limit(rate_key)

    if not recovery_code or not new_password:
        raise HTTPException(status_code=400, detail="Recovery code and new password are required")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    # If email is provided, verify against that account. If not, search by recovery code.
    user = _get_user_by_email(email) if email else _find_user_by_recovery_code(recovery_code)
    if not user or not _verify_recovery(user, recovery_code):
        _record_recovery_fail(rate_key)
        _audit(request.client.host if request.client else None, None, "recover_failed", x_client_id, {"email": email or "forgotten"})
        raise HTTPException(status_code=401, detail="Invalid recovery details")

    _set_password(int(user["id"]), new_password, must_change_pw=0)

    # Make recovery code one-time after successful use. Show this once so the user can save it.
    new_recovery_code = _rotate_recovery_code(int(user["id"]))
    _clear_recovery_fails(rate_key)

    _audit(request.client.host if request.client else None, int(user["id"]), "recover_ok", x_client_id, {"email_was_provided": bool(email)})

    return {
        "ok": True,
        "email": user["email"],
        "masked_email": _mask_email(user["email"]),
        "new_recovery_code": new_recovery_code
    }

# =========================================================
# Admin routes
# =========================================================
@app.get("/auth/admin/users")
async def admin_list_users(request: Request, user=Depends(require_admin)):
    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, email, full_name, role, created_at, updated_at, must_change_pw FROM users ORDER BY id ASC")
            rows = cur.fetchall()
        finally:
            conn.close()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "email": r["email"],
            "full_name": r["full_name"],
            "role": r["role"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "must_change_pw": int(r["must_change_pw"] or 0),
        })
    return {"ok": True, "items": out}


@app.post("/auth/admin/create_user")
async def admin_create_user(payload: dict, request: Request, user=Depends(require_admin)):
    email = (payload.get("email") or "").strip().lower()
    full_name = (payload.get("full_name") or "").strip()
    role = (payload.get("role") or "Security Personnel").strip()
    password = payload.get("password") or ""
    must_change = int(payload.get("must_change_pw") if payload.get("must_change_pw") is not None else 1)

    if not full_name:
        raise HTTPException(status_code=400, detail="Full name required")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Temporary password must be at least 8 characters")
    if not role:
        role = "Security Personnel"

    existing = _get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")

    try:
        created = _create_user(
            email=email,
            full_name=full_name,
            role=role,
            password=password,
            must_change_pw=must_change,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already exists")

    _audit(request.client.host if request.client else None, int(user["id"]), "admin_create_user",
           getattr(request.state, "client_id", None), {"target_user_id": int(created["id"]), "email": email, "role": role})

    return {
        "ok": True,
        "user": {"id": int(created["id"]), "email": email, "full_name": full_name, "role": role},
        "recovery_code": created["recovery_code"],
    }


def _delete_user_by_admin(target_user_id: int, acting_user_id: int, request: Request):
    if target_user_id <= 0:
        raise HTTPException(status_code=400, detail="user_id required")
    if target_user_id == int(acting_user_id):
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account while logged in")

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()

            cur.execute("SELECT id, email, full_name, role FROM users WHERE id=?", (target_user_id,))
            target = cur.fetchone()
            if not target:
                raise HTTPException(status_code=404, detail="User not found")

            target_info = {
                "target_user_id": int(target["id"]),
                "email": target["email"],
                "full_name": target["full_name"],
                "role": target["role"],
            }

            # Prevent removing the last administrator account.
            target_role = (target["role"] or "").strip().lower()
            if target_role in ("admin", "administrator", "system administrator") or "administrator" in target_role:
                cur.execute("SELECT COUNT(*) AS n FROM users WHERE LOWER(role) LIKE '%administrator%' OR LOWER(role)='admin'")
                admin_count = int(cur.fetchone()["n"] or 0)
                if admin_count <= 1:
                    raise HTTPException(status_code=400, detail="Cannot delete the last administrator account")

            # Remove active sessions first so the deleted user is immediately logged out.
            cur.execute("DELETE FROM sessions WHERE user_id=?", (target_user_id,))
            cur.execute("DELETE FROM users WHERE id=?", (target_user_id,))
            if cur.rowcount <= 0:
                raise HTTPException(status_code=404, detail="User not found or already removed")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    _audit(request.client.host if request.client else None, int(acting_user_id), "admin_delete_user",
           getattr(request.state, "client_id", None), target_info)
    return {"ok": True, "deleted_user_id": target_user_id}


@app.post("/auth/admin/delete_user")
async def admin_delete_user_post(payload: dict, request: Request, user=Depends(require_admin)):
    target_user_id = int(payload.get("user_id") or 0)
    return _delete_user_by_admin(target_user_id, int(user["id"]), request)


@app.delete("/auth/admin/delete_user")
async def admin_delete_user_delete(payload: dict, request: Request, user=Depends(require_admin)):
    # Keep DELETE support for compatibility, but the Admin Users page uses POST because
    # some browser/Electron setups are inconsistent with DELETE requests that contain JSON bodies.
    target_user_id = int(payload.get("user_id") or 0)
    return _delete_user_by_admin(target_user_id, int(user["id"]), request)


@app.post("/auth/admin/reset_password")
async def admin_reset_password(payload: dict, request: Request, user=Depends(require_admin)):
    target_user_id = int(payload.get("user_id") or 0)
    new_password = (payload.get("new_password") or "").strip()
    must_change = int(payload.get("must_change_pw") or 1)

    if target_user_id <= 0 or not new_password:
        raise HTTPException(status_code=400, detail="user_id and new_password required")

    _set_password(target_user_id, new_password, must_change_pw=must_change)
    _audit(request.client.host if request.client else None, int(user["id"]), "admin_reset_password",
           getattr(request.state, "client_id", None), {"target_user_id": target_user_id, "must_change_pw": must_change})
    return {"ok": True}

@app.post("/auth/admin/rotate_recovery")
async def admin_rotate_recovery(payload: dict, request: Request, user=Depends(require_admin)):
    target_user_id = int(payload.get("user_id") or 0)
    if target_user_id <= 0:
        raise HTTPException(status_code=400, detail="user_id required")

    new_code = _rotate_recovery_code(target_user_id)
    _audit(request.client.host if request.client else None, int(user["id"]), "admin_rotate_recovery",
           getattr(request.state, "client_id", None), {"target_user_id": target_user_id})
    return {"ok": True, "recovery_code": new_code}

# =========================================================
# Protected API
# =========================================================
@app.get("/status")
def status(request: Request, user=Depends(require_user)):
    if not runner:
        return {"ok": False, "error": "runner_not_started"}
    return runner.get_status()

@app.post("/config/cam0")
def config_cam0(payload: dict, request: Request, user=Depends(require_user)):
    if not runner:
        return {"ok": False, "error": "runner_not_started"}
    src = payload.get("source", None)
    _audit(request.client.host if request.client else None, int(user["id"]), "config_cam0",
           getattr(request.state, "client_id", None), {"source": src})
    res = runner.set_source(0, src)
    if continuous_mgr:
        try:
            res["continuous"] = continuous_mgr.set_source(0, src)
        except Exception:
            res["continuous"] = {"ok": False}
    return res

@app.post("/config/cam1")
def config_cam1(payload: dict, request: Request, user=Depends(require_user)):
    if not runner:
        return {"ok": False, "error": "runner_not_started"}
    src = payload.get("source", None)
    _audit(request.client.host if request.client else None, int(user["id"]), "config_cam1",
           getattr(request.state, "client_id", None), {"source": src})
    res = runner.set_source(1, src)
    if continuous_mgr:
        try:
            res["continuous"] = continuous_mgr.set_source(1, src)
        except Exception:
            res["continuous"] = {"ok": False}
    return res

@app.post("/config/cam2")
def config_cam2(payload: dict, request: Request, user=Depends(require_user)):
    if not runner:
        return {"ok": False, "error": "runner_not_started"}
    src = payload.get("source", None)
    _audit(request.client.host if request.client else None, int(user["id"]), "config_cam2",
           getattr(request.state, "client_id", None), {"source": src})
    res = runner.set_source(2, src)
    if continuous_mgr:
        try:
            res["continuous"] = continuous_mgr.set_source(2, src)
        except Exception:
            res["continuous"] = {"ok": False}
    return res

# =========================================================
# Timeline API
# =========================================================
@app.get("/timeline/segments")
def timeline_segments(cam: int = 0, hours: int = 24, request: Request = None, user=Depends(require_user)):
    from timeline_utils import list_segments
    cam = int(cam)
    hours = max(1, min(int(hours), 24 * 31))

    cam_dir = os.path.join(CONTINUOUS_DIR, f"cam{cam}")

    # Use the real segment size (same as continuous recorder)
    seg_sec = 60
    try:
        if continuous_mgr is not None:
            seg_sec = int(getattr(continuous_mgr, "segment_seconds", 60))
    except Exception:
        pass

    items = list_segments(cam_dir, segment_seconds=seg_sec)

    # Filter to last N hours
    now_ts = int(time.time())
    window_start = now_ts - hours * 3600
    items = [s for s in items if int(s.get("ts_end", 0)) >= window_start]

    #return {"ok": True, "cam": cam, "hours": hours, "segment_seconds": seg_sec, "cam_dir": cam_dir, "items": items}
    return {"ok": True, "cam": cam, "hours": hours, "segment_seconds": seg_sec, "items": items}

@app.get("/timeline/events")
def timeline_events(cam: int = 0, hours: int = 24, request: Request = None, user=Depends(require_user)):
    cam = int(cam)
    hours = int(hours)
    if hours < 1:
        hours = 1
    if hours > 24 * 31:
        hours = 24 * 31

    now_ts = int(time.time())
    window_end = now_ts
    window_start = int(now_ts - hours * 3600)

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, ts_start, ts_end, cam, label, conf, action_id, clip_rel, created_at
                   FROM gv_events
                   WHERE cam=? AND ts_end>=?
                   ORDER BY ts_start ASC""",
                (cam, window_start),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

    events = []
    for r in rows:
        clip_rel = r["clip_rel"]
        events.append({
            "id": int(r["id"]),
            "ts_start": int(r["ts_start"]),
            "ts_end": int(r["ts_end"]),
            "cam": int(r["cam"]),
            "label": r["label"],
            "conf": float(r["conf"] or 0.0),
            "action_id": r["action_id"],
            "clip_rel": clip_rel,
            "clip_url": (f"/records/{clip_rel}" if clip_rel else None),
            "created_at": r["created_at"],
        })

    # Match records.html expectations:
    # - window_start/window_end
    # - events array
    # Also keep "items" for backward compatibility (if other pages use it)
    return {
        "ok": True,
        "cam": cam,
        "hours": hours,
        "window_start": window_start,
        "window_end": window_end,
        "events": events,
        "items": events,
    }


@app.delete("/timeline/events")
def timeline_clear_events(cam: int = 0, hours: int = 24, all_events: bool = False, user=Depends(require_user)):
    """Clear detected timeline events so the Detected Events list does not keep building up.

    This only deletes event markers from gv_events. It does NOT delete continuous recordings
    or manually saved clips.
    """
    cam = int(cam)
    hours = int(hours)
    if hours < 1:
        hours = 1
    if hours > 24 * 31:
        hours = 24 * 31

    params = [cam]
    where = "cam=?"
    if not bool(all_events):
        window_start = int(time.time()) - hours * 3600
        where += " AND ts_end>=?"
        params.append(window_start)

    with _DB_LOCK:
        conn = _db_connect()
        try:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM gv_events WHERE {where}", tuple(params))
            deleted = int(cur.rowcount or 0)
            conn.commit()
        finally:
            conn.close()

    return {"ok": True, "cam": cam, "hours": hours, "all_events": bool(all_events), "deleted": deleted}

@app.get("/timeline/resolve")
def timeline_resolve(cam: int = 0, ts: int = 0, request: Request = None, user=Depends(require_user)):
    from timeline_utils import resolve_ts_to_segment
    cam = int(cam)
    ts = int(ts)
    # If frontend ever passes JS Date.now() (ms), convert to seconds
    if ts > 10_000_000_000:
        ts = ts // 1000

    seg_sec = 60
    try:
        if continuous_mgr is not None:
            seg_sec = int(getattr(continuous_mgr, "segment_seconds", 60))
    except Exception:
        seg_sec = 60

    cam_dir = os.path.join(CONTINUOUS_DIR, f"cam{cam}")
    r = resolve_ts_to_segment(cam_dir, ts, seg_sec)
    if not r:
        return {"ok": False, "error": "segment_not_found"}

    filename, offset_s, seg_start = r
    return {
        "ok": True,
        "cam": cam,
        "ts": int(ts),
        "segment_seconds": int(seg_sec),
        "segment_start": int(seg_start),
        "offset_seconds": int(offset_s),
        "segment_url": f"/continuous/cam{cam}/{filename}",
    }


@app.post("/timeline/export")
async def timeline_export(request: Request, user=Depends(require_user)):
    """
    Export a manually selected timeline range into one saved MP4 clip.

    Performance fixes:
    - Run FFmpeg work in a background thread so FastAPI/camera endpoints do not freeze.
    - Do NOT join a huge temporary video before trimming.
    - For one segment, trim directly from that segment.
    - For multiple segments, use concat inpoint/outpoint so FFmpeg only reads the selected range.
    - Prefer fast stream-copy first, then fallback to ultrafast re-encode only if copy fails.
    - Reject suspiciously large gaps instead of seeking many hours into a short segment.
    """
    from timeline_utils import list_segments
    from continuous_recorder import resolve_ffmpeg_path

    try:
        body = await request.json()
    except Exception:
        body = {}

    cam = int(body.get("cam", 0))
    ts_start = int(float(body.get("ts_start", 0)))
    ts_end = int(float(body.get("ts_end", 0)))
    label = str(body.get("label", "manual") or "manual").strip().replace("/", "_").replace("\\", "_")

    if ts_start > 10_000_000_000:
        ts_start //= 1000
    if ts_end > 10_000_000_000:
        ts_end //= 1000
    if ts_end <= ts_start:
        return {"ok": False, "error": "end_must_be_after_start"}

    # Avoid accidental very long exports.
    max_export_sec = int(os.environ.get("GV_MAX_EXPORT_SEC", str(60 * 60)))
    if (ts_end - ts_start) > max_export_sec:
        return {"ok": False, "error": f"range_too_long_max_{max_export_sec}_seconds"}

    seg_sec = 10
    try:
        if continuous_mgr is not None:
            seg_sec = int(getattr(continuous_mgr, "segment_seconds", 10))
    except Exception:
        pass
    seg_sec = max(1, int(seg_sec or 10))

    cam_dir = os.path.join(CONTINUOUS_DIR, f"cam{cam}")
    segs = list_segments(cam_dir, segment_seconds=seg_sec)

    # Only use real overlapping coverage. timeline_utils caps large gaps, so false long coverage is avoided.
    hits = [s for s in segs if int(s["ts_end"]) > ts_start and int(s["ts_start"]) < ts_end]
    if not hits:
        return {"ok": False, "error": "no_video_coverage_for_selected_range"}

    # Extra safety: if there is a big gap between selected files, do not let FFmpeg hang.
    max_gap = int(os.environ.get("GV_EXPORT_MAX_SEGMENT_GAP_SEC", str(max(8, seg_sec * 3))))
    for prev, cur in zip(hits, hits[1:]):
        gap = int(cur["ts_start"]) - int(prev["ts_end"])
        if gap > max_gap:
            return {
                "ok": False,
                "error": "selected_range_has_recording_gap",
                "detail": f"Gap of {gap}s between {prev.get('filename')} and {cur.get('filename')}",
            }

    export_dir = os.path.join(RECORD_DIR, "manual_exports")
    os.makedirs(export_dir, exist_ok=True)

    stamp1 = datetime.fromtimestamp(ts_start).strftime("%Y%m%d_%H%M%S")
    stamp2 = datetime.fromtimestamp(ts_end).strftime("%H%M%S")
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("_", "-"))[:32] or "manual"
    display_cam = int(cam) + 1
    out_name = f"cam{display_cam}_{safe_label}_{stamp1}_to_{stamp2}.mp4"
    out_path = os.path.join(export_dir, out_name)

    ffmpeg = resolve_ffmpeg_path(None)
    duration = max(1, int(ts_end - ts_start))

    def _run_export_sync():
        # Windows: run FFmpeg with lower priority when possible.
        creationflags = 0
        try:
            if os.name == "nt":
                creationflags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        except Exception:
            creationflags = 0

        def run_cmd(cmd, timeout=None):
            return subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                timeout=timeout,
            )

        timeout_sec = int(os.environ.get("GV_EXPORT_TIMEOUT_SEC", "120"))

        # Fast path: selected clip is inside one continuous recording segment.
        if len(hits) == 1:
            h = hits[0]
            src_path = os.path.join(cam_dir, h["filename"])
            offset = max(0, int(ts_start) - int(h["ts_start"]))

            # 1st try: stream copy, fastest and lowest CPU.
            copy_cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(offset), "-i", src_path,
                "-t", str(duration),
                "-map", "0:v:0", "-an",
                "-c:v", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                out_path,
            ]

            try:
                run_cmd(copy_cmd, timeout=timeout_sec)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                    return
            except Exception:
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except Exception:
                    pass

            # Fallback: ultrafast re-encode. More compatible, still lighter than before.
            reenc_cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(offset), "-i", src_path,
                "-t", str(duration),
                "-map", "0:v:0", "-an",
                "-vf", "fps=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-threads", "1",
                "-movflags", "+faststart",
                out_path,
            ]
            run_cmd(reenc_cmd, timeout=timeout_sec)
            return

        # Multi-file path: concat only the required portion using inpoint/outpoint.
        with tempfile.TemporaryDirectory() as td:
            list_path = os.path.join(td, "concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for h in hits:
                    src_path = os.path.abspath(os.path.join(cam_dir, h["filename"]))
                    # concat demuxer needs forward slashes or escaped backslashes on Windows.
                    safe_path = src_path.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

                    inpoint = max(0, int(ts_start) - int(h["ts_start"]))
                    outpoint = min(int(h["ts_end"]), int(ts_end)) - int(h["ts_start"])
                    if inpoint > 0:
                        f.write(f"inpoint {inpoint}\n")
                    if outpoint > 0:
                        f.write(f"outpoint {outpoint}\n")

            # 1st try: stream copy from trimmed concat. Fastest.
            copy_cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-map", "0:v:0", "-an",
                "-c:v", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                out_path,
            ]
            try:
                run_cmd(copy_cmd, timeout=timeout_sec)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                    return
            except Exception:
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except Exception:
                    pass

            # Fallback re-encode, still with trimmed concat; no huge temporary joined file.
            reenc_cmd = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-map", "0:v:0", "-an",
                "-vf", "fps=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-threads", "1",
                "-movflags", "+faststart",
                out_path,
            ]
            run_cmd(reenc_cmd, timeout=timeout_sec)

    try:
        await asyncio.to_thread(_run_export_sync)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffmpeg_export_timeout", "detail": "Export took too long. Try a shorter range or check for damaged recording segments."}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": "ffmpeg_export_failed", "detail": (e.stderr or b"").decode("utf-8", "ignore")[-800:]}
    except Exception as e:
        return {"ok": False, "error": "export_failed", "detail": str(e)[-800:]}

    if not os.path.exists(out_path) or os.path.getsize(out_path) <= 1024:
        return {"ok": False, "error": "export_output_empty"}

    rel = os.path.relpath(out_path, RECORD_DIR).replace("\\", "/")
    try:
        with open(out_path + ".json", "w", encoding="utf-8") as mf:
            json.dump({
                "label": label,
                "created_ts": int(time.time()) + TIME_OFFSET_SEC,
                "created": now_offset().strftime("%Y-%m-%d %H:%M:%S"),
                "cam": cam,
                "ts_start": int(ts_start),
                "ts_end": int(ts_end),
                "duration_seconds": int(duration),
                "manual_export": True,
            }, mf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {"ok": True, "file": out_name, "url": f"/records/{rel}", "ts_start": int(ts_start), "ts_end": int(ts_end), "cam": cam}

@app.get("/records_list")
def records_list(request: Request, user=Depends(require_user)):
    files = sorted(
        glob.glob(os.path.join(RECORD_DIR, "**", "*.mp4"), recursive=True),
        key=os.path.getmtime,
        reverse=True
    )

    out = []
    count = 0

    for p in files:
        rel = os.path.relpath(p, RECORD_DIR).replace("\\", "/")
        # Only show manually saved clips in Records.
        # Auto Punching/Kicking clips are disabled; old auto clips are hidden from this list.
        if not rel.startswith("manual_exports/"):
            continue
        mtime = os.path.getmtime(p)

        meta_path = p + ".json"
        meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = None

        if not isinstance(meta, dict):
            meta = {}

        try:
            meta.setdefault("label", rel.split("/", 1)[0])
        except Exception:
            meta.setdefault("label", "Recorded")

        meta.setdefault("created_ts", int(mtime) + TIME_OFFSET_SEC)
        meta.setdefault("created", offset_datetime(datetime.fromtimestamp(mtime)).strftime("%Y-%m-%d %H:%M:%S"))

        out.append({"file": rel, "url": f"/records/{rel}", "meta": meta})

        count += 1
        if count >= 200:
            break

    return {"items": out}

# =========================================================
# WebSocket (token via query param ?token=.)
# =========================================================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = (ws.query_params.get("token") or "").strip()
    client_id = (ws.query_params.get("client_id") or "").strip() or None

    if not token:
        await ws.close(code=1008)
        return

    sess = _get_session(token)
    if not sess:
        await ws.close(code=1008)
        return

    if TOKEN_TTL_SEC and TOKEN_TTL_SEC > 0:
        try:
            exp = datetime.fromisoformat(str(sess["expires_at"]).replace("Z", ""))
            if datetime.utcnow() > exp:
                _delete_token(token)
                await ws.close(code=1008)
                return
        except Exception:
            _delete_token(token)
            await ws.close(code=1008)
            return

    user = _get_user_by_id(int(sess["user_id"]))
    if not user:
        _delete_token(token)
        await ws.close(code=1008)
        return

    _touch_session(token)
    await ws.accept()

    ip = ws.client.host if ws.client else None
    _audit(ip, int(user["id"]), "ws_connect", client_id or sess.get("client_id"), {})

    last_id = -1
    try:
        while True:
            await asyncio.sleep(0.05)
            if not runner:
                continue
            pkt = runner.get_latest()
            # continuous CSV logging based on WS output
            _maybe_log_from_ws_packet(pkt)
            if not pkt:
                continue
            if pkt.get("id", -1) == last_id:
                continue
            last_id = pkt["id"]
            await ws.send_json(pkt)
    except WebSocketDisconnect:
        _audit(ip, int(user["id"]), "ws_disconnect", client_id or sess.get("client_id"), {})
    except Exception as e:
        _audit(ip, int(user["id"]), "ws_error", client_id or sess.get("client_id"), {"err": repr(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
