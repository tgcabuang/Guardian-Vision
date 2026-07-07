# backend/check_packaging.py
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DIST = HERE / "build" / "server_ws.dist"

def fail(msg: str):
    print("\n[PACKAGING CHECK FAILED]\n" + msg + "\n", file=sys.stderr)
    sys.exit(1)

def ok(msg: str):
    print("[OK] " + msg)

def main():
    if not DIST.exists():
        fail(f"Dist folder not found: {DIST}\nBuild first via build_backend.bat")

    # --- required root files (same level as server_ws.exe) ---
    required_root_files = [
        "gv_users.db",
        "gv_config.json",
    ]

    missing_root = [f for f in required_root_files if not (DIST / f).exists()]
    if missing_root:
        fail(
            "Missing required backend files in dist root:\n"
            + "\n".join([f" - {DIST / f}" for f in missing_root])
            + "\n\nFix: add --include-data-file for these in Nuitka build."
        )
    else:
        ok("Found gv_users.db + gv_config.json in dist root.")

    # --- required assets ---
    assets_dir = DIST / "assets"
    if not assets_dir.exists():
        fail(f"Missing assets folder in dist: {assets_dir}\nFix: add --include-data-dir=assets=assets")

    # NOTE: allow rtmpose_cfg.py OR rtmpose_cfg.py.py (we patch server_ws.py to fallback)
    required_assets_any = {
        "rtmpose_cfg": ["rtmpose_cfg.py", "rtmpose_cfg.py.py"],
    }
    required_assets_exact = [
        "yolov8n.pt",
        "rtmpose.pth",
        "class_map.json",
        "ctrgcn_bundle.pth",
    ]

    # exact assets
    missing_assets = [f for f in required_assets_exact if not (assets_dir / f).exists()]
    if missing_assets:
        fail(
            "Missing required assets:\n"
            + "\n".join([f" - {assets_dir / f}" for f in missing_assets])
            + "\n\nFix: ensure backend/assets contains these, then rebuild."
        )
    else:
        ok("Found required exact assets.")

    # any-of assets
    for key, candidates in required_assets_any.items():
        if not any((assets_dir / c).exists() for c in candidates):
            fail(
                f"Missing required asset for {key}. Expected one of:\n"
                + "\n".join([f" - {assets_dir / c}" for c in candidates])
                + "\n\nFix: rename file or include it."
            )
    ok("Found RTMPose config asset (one of accepted filenames).")

    ok("Packaging checks passed ✅")

if __name__ == "__main__":
    main()
