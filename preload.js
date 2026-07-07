// preload.js
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  // window controls
  minimize: () => ipcRenderer.send("minimize"),
  maximize: () => ipcRenderer.send("maximize"),
  close: () => ipcRenderer.send("close"),

  // option B: records dir + backend restart
  getConfig: () => ipcRenderer.invoke("gv:getConfig"),
  pickRecordsDir: () => ipcRenderer.invoke("gv:pickRecordsDir"),
  setRecordsDir: (dir) => ipcRenderer.invoke("gv:setRecordsDir", dir),
  openPath: (p) => ipcRenderer.invoke("gv:openPath", p),
});
