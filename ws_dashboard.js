const WS_URL = "ws://127.0.0.1:8765/ws";

let ws;
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => console.log("[WS] connected");
  ws.onclose = () => setTimeout(connect, 1000);
  ws.onerror = (e) => console.log("[WS] error", e);

  ws.onmessage = (evt) => {
    const pkt = JSON.parse(evt.data);

    // frame (base64 JPEG)
    const img = document.getElementById("dashCam0");
    if (img && pkt.frame0) {
      img.src = "data:image/jpeg;base64," + pkt.frame0;
    }

    // tracks (gesture predictions)
    const tracks = pkt.cam0?.tracks ?? [];
    const box = document.getElementById("dashTracks");
    if (box) {
      box.innerHTML = tracks
        .map(t => `ID ${t.tid}: <b>${t.label}</b> (${Math.round(t.conf * 100)}%)`)
        .join("<br/>");
    }

    // optional fps (you draw FPS on the frame already, but you can also show it here if you add it)
    const fpsEl = document.getElementById("dashFps");
    if (fpsEl && pkt.fps != null) fpsEl.textContent = pkt.fps;
  };
}
connect();
