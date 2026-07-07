// main.js
const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron/main");
const path = require("node:path");
const fs = require("node:fs");
const { spawn } = require("child_process");

let win = null;
let pyProc = null;

// -------------------------------
// Config helpers (stored in userData)
// -------------------------------
function configPath() {
  return path.join(app.getPath("userData"), "gv_config.json");
}

function readConfig() {
  try {
    const p = configPath();
    if (!fs.existsSync(p)) return {};
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return {};
  }
}

function writeConfig(cfg) {
  const p = configPath();
  fs.writeFileSync(p, JSON.stringify(cfg, null, 2), "utf-8");
}

// Optional: also write next to backend/server_ws.py so backend can read it too
function writeBackendSideConfig(cfg) {
  try {
    const p = path.join(__dirname, "backend", "gv_config.json");
    fs.writeFileSync(p, JSON.stringify(cfg, null, 2), "utf-8");
  } catch {
    // ignore
  }
}

// -------------------------------
// Backend control
// -------------------------------
function startBackend() {
  const cfg = readConfig();
  const recordsDir = (cfg.recordsDir || "").trim();

  // DEV build output (Nuitka standalone)
  const devBackendExe = path.join(__dirname, "backend", "build", "server_ws.dist", "server_ws.exe");

  // PACKAGED app extraResources path
  const packagedBackendExe = path.join(process.resourcesPath, "backend", "server_ws.exe");

  const backendExe = app.isPackaged ? packagedBackendExe : devBackendExe;

  // ✅ check exe exists (very common failure)
  if (!fs.existsSync(backendExe)) {
    dialog.showErrorBox(
      "Backend missing",
      `Cannot find backend exe:\n${backendExe}\n\nFix:\n- Rebuild backend\n- Or check electron-builder extraResources.\n`
    );
    return;
  }

  // Resolve bundled ffmpeg path (dev vs packaged)
  const devFfmpeg = path.join(__dirname, "backend", "bin", "ffmpeg.exe");
  const packagedFfmpeg = path.join(process.resourcesPath, "backend", "bin", "ffmpeg.exe");
  const ffmpegExe = app.isPackaged ? packagedFfmpeg : devFfmpeg;

  const logPath = path.join(app.getPath("userData"), "backend.log");
  const log = (msg) => {
    try { fs.appendFileSync(logPath, msg + "\n", "utf-8"); } catch {}
  };

  log("------------------------------------------------------");
  log(`[START] isPackaged=${app.isPackaged}`);
  log(`[START] backendExe=${backendExe}`);
  log(`[START] cwd=${path.dirname(backendExe)}`);
  log(`[START] recordsDir=${recordsDir || "(default backend resolve_record_dir)"}`);
  log(`[START] ffmpegExe=${fs.existsSync(ffmpegExe) ? ffmpegExe : "(missing)"} `);

  // IMPORTANT: pass env vars for your requested behavior
  const env = {
    ...process.env,
    GV_RECORD_DIR: recordsDir,        // records base folder
    GV_HOST: "127.0.0.1",
    GV_PORT: "8766",

    // ✅ Your requested changes:
    GV_TOKEN_TTL_SEC: "0",            // no expiration
    GV_KEEP_HOURS: "0",               // no auto-delete for continuous segments
    GV_SEGMENT_SEC: "10",             // segment length

    // ✅ Use bundled ffmpeg if present (no PATH install)
    ...(fs.existsSync(ffmpegExe) ? { GV_FFMPEG: ffmpegExe } : {}),
  };

  pyProc = spawn(backendExe, [], {
    cwd: path.dirname(backendExe),
    windowsHide: true,
    env,
  });

  pyProc.stdout.on("data", (d) => log("[STDOUT] " + d.toString()));
  pyProc.stderr.on("data", (d) => log("[STDERR] " + d.toString()));
  pyProc.on("close", (code) => log("[CLOSE] code=" + code));
  pyProc.on("error", (err) => log("[ERROR] " + String(err)));
}

// Force-kill process tree (Windows) so restart is real
function stopBackend() {
  return new Promise((resolve) => {
    if (!pyProc) return resolve();

    const p = pyProc;
    const pid = p.pid;
    pyProc = null;

    const done = () => resolve();
    p.once("close", done);

    try {
      if (process.platform === "win32" && pid) {
        // Kill full tree (/T) force (/F) — avoids old backend lingering and causing 404/port issues.
        spawn("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true })
          .on("close", () => resolve());
      } else {
        p.kill();
        setTimeout(resolve, 800);
      }
    } catch {
      resolve();
    }
  });
}

// -------------------------------
// Window
// -------------------------------
function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    icon: path.join(__dirname, "image/icon.png"),
    frame: true,
    autoHideMenuBar: true,
    titleBarStyle: "hiddenInset",
    title: "",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  win.webContents.on("did-fail-load", (_e, code, desc, url) => {
    console.error("[did-fail-load]", code, desc, url);
  });

  // Always start at the login page.
  // login.html decides whether to restore the session based on the Remember me checkbox.
  win.loadFile(path.join(__dirname, "login.html"));
  win.setTitle("");
}

app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// -------------------------------
// IPC window controls
// -------------------------------
ipcMain.on("minimize", () => win && win.minimize());
ipcMain.on("maximize", () => {
  if (!win) return;
  win.isMaximized() ? win.unmaximize() : win.maximize();
});
ipcMain.on("close", () => win && win.close());

// -------------------------------
// IPC: Records folder (Option B)
// -------------------------------
ipcMain.handle("gv:getConfig", async () => {
  return { ok: true, config: readConfig() };
});

ipcMain.handle("gv:pickRecordsDir", async () => {
  if (!win) return { ok: false, error: "window_not_ready" };
  const res = await dialog.showOpenDialog(win, {
    title: "Select Records Folder",
    properties: ["openDirectory", "createDirectory"],
  });
  if (res.canceled || !res.filePaths?.length) return { ok: true, dir: "" };
  return { ok: true, dir: res.filePaths[0] };
});


ipcMain.handle("gv:setRecordsDir", async (_evt, dir) => {
  try {
    const d = (dir || "").trim();
    const cfg = readConfig();
    cfg.recordsDir = d;
    writeConfig(cfg);
    writeBackendSideConfig(cfg); // optional but helpful

    // restart backend so it uses the new GV_RECORD_DIR
    await stopBackend();
    startBackend();

    return { ok: true };
  } catch (e) {
    return { ok: false, error: String(e?.message || e) };
  }
});

ipcMain.handle("gv:openPath", async (_evt, p) => {
  try {
    await shell.openPath(String(p || ""));
    return { ok: true };
  } catch (e) {
    return { ok: false, error: String(e?.message || e) };
  }
});
