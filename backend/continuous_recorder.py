import os
import time
import threading
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class ContinuousRecorderConfig:
    cam_dirname: str                 # e.g. 'cam0'
    source: str                      # rtsp/http/file path
    root_dir: str                    # e.g. <records>/continuous
    segment_seconds: int = 10        # default 10s for faster playback/coverage
    keep_hours: int = 24 * 30

    # If True: copy stream (low CPU). If False: re-encode (more CPU but cleaner segments).
    use_copy: bool = True

    # Encoding knobs (used only when use_copy=False)
    vcodec: str = "libx264"
    crf: int = 28
    preset: str = "veryfast"
    fps: Optional[int] = None        # e.g. 15 to reduce storage
    scale: Optional[str] = None      # e.g. "1280:720"

    # Optional: override ffmpeg path (absolute path preferred)
    ffmpeg_path: Optional[str] = None

    # Restart ffmpeg if it crashes (RTSP drops are common)
    auto_restart: bool = True
    restart_delay_sec: float = 1.0


def resolve_ffmpeg_path(explicit: Optional[str] = None) -> str:
    """
    Resolve ffmpeg.exe in this priority order:
      1) explicit argument (cfg.ffmpeg_path) if it exists
      2) env var GV_FFMPEG if it exists
      3) bundled backend/bin/ffmpeg.exe (relative to this file)
      4) ffmpeg in PATH
    """
    if explicit:
        p = explicit.strip().strip('"')
        if p and os.path.isfile(p):
            return p

    envp = (os.environ.get("GV_FFMPEG", "") or "").strip().strip('"')
    if envp and os.path.isfile(envp):
        return envp

    here = os.path.dirname(os.path.abspath(__file__))  # .../backend
    bundled = os.path.join(here, "bin", "ffmpeg.exe")
    if os.path.isfile(bundled):
        return bundled

    which = shutil.which("ffmpeg")
    if which:
        return which

    raise FileNotFoundError(
        "FFmpeg not found.\n"
        "Fix one of these:\n"
        "  - Put ffmpeg at: backend/bin/ffmpeg.exe\n"
        "  - OR set env var GV_FFMPEG=C:\\path\\to\\ffmpeg.exe\n"
        "  - OR add ffmpeg to PATH\n"
    )


class ContinuousRecorder:
    """Records a single camera source into segmented MP4 files."""

    def __init__(self, cfg: ContinuousRecorderConfig):
        self.cfg = cfg
        self.stop_evt = threading.Event()
        self.proc: Optional[subprocess.Popen] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None

        self.cam_dir = os.path.join(self.cfg.root_dir, self.cfg.cam_dirname)
        os.makedirs(self.cam_dir, exist_ok=True)

        self._log_fh = None  # file handle for ffmpeg_error.log

    def _ffmpeg(self) -> str:
        return resolve_ffmpeg_path(self.cfg.ffmpeg_path)

    def _log_path(self) -> str:
        return os.path.join(self.cam_dir, "ffmpeg_error.log")

    def _append_log(self, text: str) -> None:
        try:
            with open(self._log_path(), "a", encoding="utf-8", errors="ignore") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    def _build_cmd(self) -> List[str]:
        seg = max(1, int(self.cfg.segment_seconds))
        out_pattern = os.path.join(self.cam_dir, "%Y%m%d_%H%M%S.mp4")

        cmd: List[str] = [
            self._ffmpeg(),
            "-hide_banner",
            "-loglevel", "error",

            # These are the exact flags that made your TP-Link RTSP stream produce stable MP4 segments.
            "-use_wallclock_as_timestamps", "1",
            "-fflags", "+genpts+igndts",
            "-avoid_negative_ts", "make_zero",
        ]

        # RTSP knobs (SAFE: do NOT use -stimeout / -rw_timeout because your ffmpeg rejects them)
        if isinstance(self.cfg.source, str) and self.cfg.source.lower().startswith("rtsp://"):
            cmd += [
                "-rtsp_transport", "tcp",
                "-max_delay", "500000",
            ]

        cmd += ["-i", self.cfg.source]

        # Important: record VIDEO ONLY to avoid non-monotonic DTS issues from audio streams
        cmd += ["-map", "0:v:0", "-an"]

        if self.cfg.use_copy:
            cmd += ["-c:v", "copy"]
        else:
            vf = []
            if self.cfg.scale:
                vf.append(f"scale={self.cfg.scale}")
            if self.cfg.fps:
                vf.append(f"fps={int(self.cfg.fps)}")
            if vf:
                cmd += ["-vf", ",".join(vf)]

            cmd += [
                "-c:v", self.cfg.vcodec,
                "-preset", self.cfg.preset,
                "-crf", str(int(self.cfg.crf)),
            ]

        # segment muxer
        cmd += [
            "-f", "segment",
            "-segment_time", str(seg),
            "-segment_format", "mp4",
            "-segment_format_options", "movflags=+faststart",
            "-reset_timestamps", "1",
            "-strftime", "1",
            out_pattern,
        ]

        return cmd

    def _spawn_ffmpeg(self):
        cmd = self._build_cmd()

        # log the exact command used
        self._append_log("\n\n===== FFmpeg spawn =====")
        self._append_log("TIME: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        self._append_log("CMD: " + " ".join(cmd))

        creationflags = 0
        if os.name == "nt":
            try:
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            except Exception:
                creationflags = 0

        # IMPORTANT: do NOT use stderr=PIPE (it can block/hang). Write to ffmpeg_error.log instead.
        self._log_fh = open(self._log_path(), "a", encoding="utf-8", errors="ignore")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._log_fh,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def start(self):
        if self.proc and self.proc.poll() is None:
            return

        self.stop_evt.clear()
        self._spawn_ffmpeg()

        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        if bool(self.cfg.auto_restart):
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self.watchdog_thread.start()

    def stop(self):
        self.stop_evt.set()

        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2.0)
            except Exception:
                pass

        self.proc = None

        try:
            if self._log_fh:
                self._log_fh.flush()
                self._log_fh.close()
        except Exception:
            pass
        self._log_fh = None

    def _watchdog_loop(self):
        while not self.stop_evt.is_set():
            p = self.proc
            if p is None:
                break

            code = p.poll()
            if code is None:
                self.stop_evt.wait(0.5)
                continue

            # process exited
            self._append_log(f"FFmpeg exited with code: {code}")

            if self.stop_evt.is_set():
                break

            # small delay then restart
            try:
                time.sleep(float(self.cfg.restart_delay_sec or 1.0))
            except Exception:
                time.sleep(1.0)

            # close old log handle before respawn
            try:
                if self._log_fh:
                    self._log_fh.flush()
                    self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

            try:
                self._spawn_ffmpeg()
            except Exception as e:
                self._append_log("Respawn failed: " + repr(e))
                self.stop_evt.wait(2.0)

    def _cleanup_loop(self):
        while not self.stop_evt.is_set():
            try:
                self.cleanup_old_segments()
            except Exception:
                pass
            self.stop_evt.wait(60)

    def cleanup_old_segments(self):
        if int(self.cfg.keep_hours or 0) <= 0:
            return

        keep_seconds = int(self.cfg.keep_hours) * 3600
        cutoff = time.time() - keep_seconds

        for fn in os.listdir(self.cam_dir):
            if not fn.lower().endswith(".mp4"):
                continue
            p = os.path.join(self.cam_dir, fn)
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                continue
            if mtime < cutoff:
                try:
                    os.remove(p)
                except Exception:
                    pass


class ContinuousRecordingManager:
    """Manages up to N ContinuousRecorder instances (one per camera index)."""

    def __init__(self, root_dir: str, num_cams: int = 3, segment_seconds: int = 10, keep_hours: int = 24 * 30):
        self.root_dir = root_dir
        self.num_cams = int(num_cams)
        self.segment_seconds = int(segment_seconds)
        self.keep_hours = int(keep_hours)
        os.makedirs(self.root_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._recorders: List[Optional[ContinuousRecorder]] = [None] * self.num_cams
        self._sources: List[Optional[str]] = [None] * self.num_cams

    def stop_all(self):
        with self._lock:
            for i in range(self.num_cams):
                self._stop_i(i)

    def _stop_i(self, cam_idx: int):
        r = self._recorders[cam_idx]
        if r is not None:
            try:
                r.stop()
            except Exception:
                pass
        self._recorders[cam_idx] = None
        self._sources[cam_idx] = None

    def set_source(self, cam_idx: int, source) -> Dict[str, object]:
        cam_idx = int(cam_idx)
        if cam_idx < 0 or cam_idx >= self.num_cams:
            return {"ok": False, "error": "bad_cam_index"}

        src_str = None
        if isinstance(source, str):
            s = source.strip()
            if s:
                src_str = s

        with self._lock:
            if src_str is None:
                self._stop_i(cam_idx)
                return {"ok": True, "cam": cam_idx, "recording": False}

            if self._sources[cam_idx] == src_str and self._recorders[cam_idx] is not None:
                return {"ok": True, "cam": cam_idx, "recording": True, "source": src_str}

            self._stop_i(cam_idx)

            cfg = ContinuousRecorderConfig(
                cam_dirname=f"cam{cam_idx}",
                source=src_str,
                root_dir=self.root_dir,
                segment_seconds=self.segment_seconds,
                keep_hours=self.keep_hours,
                use_copy=True,
                ffmpeg_path=None,
                auto_restart=True,
                restart_delay_sec=1.0,
            )

            r = ContinuousRecorder(cfg)
            r.start()
            self._recorders[cam_idx] = r
            self._sources[cam_idx] = src_str

            return {"ok": True, "cam": cam_idx, "recording": True, "source": src_str}


# Backwards-compatible alias
ContinuousRecorderManager = ContinuousRecordingManager
