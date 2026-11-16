// Yap browser client: mic -> server, server -> captions + translated voice.
const $ = (s) => document.querySelector(s);
const COLORS = ["#ff7a3d", "#3ddc97", "#6aa8ff", "#ffb547", "#d98bff", "#ff5a8a"];
let lang = "hi", demo = null, voiceB64 = null, ws = null, micCtx = null, node = null;
let micOn = true, recording = null, me = null, people = [];
const out = new (window.AudioContext || window.webkitAudioContext)();
let playEnd = 0;                       // when the last scheduled translated clip finishes
const liveEnd = {};                    // per-speaker schedule for same-language pass-through

// ------------------------------------------------------------ join screen
document.querySelectorAll("[data-lang]").forEach((b) => b.onclick = () => {
  lang = b.dataset.lang;
  document.querySelectorAll("[data-lang]").forEach((x) => x.classList.toggle("on", x === b));
  $("#readtxt").textContent = lang === "hi"
    ? "नमस्ते, मैं आज की मीटिंग में हूँ। मैं अपनी आवाज़ रिकॉर्ड कर रहा हूँ ताकि दूसरे लोग मुझे अपनी भाषा में, मेरी ही आवाज़ में सुन सकें।"
    : "Hi, I'm joining today's meeting. I'm recording my voice so everyone else can hear me in their own language, but still in my voice.";
});
document.querySelectorAll("[data-demo]").forEach((b) => b.onclick = () => {
  demo = b.dataset.demo; voiceB64 = null;
  document.querySelectorAll("[data-demo],#rec").forEach((x) => x.classList.toggle("on", x === b));
  $("#voicehint").textContent = `Using ${b.textContent.replace("Demo: ", "")}'s made-up voice.`;
  $("#go").disabled = false;
});

async function startMic() {
  if (micCtx) return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
  micCtx = new AudioContext({ sampleRate: 16000 });
  await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  node = new AudioWorkletNode(micCtx, "mic");
  micCtx.createMediaStreamSource(stream).connect(node);
  node.port.onmessage = (e) => onMic(e.data);
}

function onMic({ pcm, peak }) {
  $("#lvl").style.width = Math.min(100, peak * 160) + "%";
  if (recording) { recording.push(new Int16Array(pcm)); return; }
  if (!ws || ws.readyState !== 1 || !micOn) return;
  // half duplex: while a translation is playing, don't send the mic (stops echo loops)
  if (out.currentTime < playEnd + 0.25) { if (!onMic.muted) { ws.send(JSON.stringify({ type: "mute" })); onMic.muted = true; } return; }
  onMic.muted = false;
  ws.send(pcm);
}

$("#rec").onclick = async () => {
  await startMic();
  demo = null;
  document.querySelectorAll("[data-demo]").forEach((x) => x.classList.remove("on"));
  $("#rec").classList.add("on");
  recording = [];
  let left = 10;
  $("#rec").textContent = `● Recording… ${left}`;
  const t = setInterval(() => {
    left--;
    $("#rec").textContent = left > 0 ? `● Recording… ${left}` : "✓ Recorded";
    if (left <= 0) {
      clearInterval(t);
      const n = recording.reduce((a, b) => a + b.length, 0), all = new Int16Array(n);
      let o = 0; for (const c of recording) { all.set(c, o); o += c.length; }
      recording = null;
      voiceB64 = b64(all.buffer);
      $("#voicehint").textContent = "Got your voice. It stays on this laptop.";
      $("#go").disabled = false;
    }
  }, 1000);
};

$("#go").onclick = async () => {
  await startMic(); await out.resume();
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "join", name: $("#name").value.trim() || "Guest", lang, demo, voice: voiceB64 }));
  ws.onmessage = (e) => onMsg(JSON.parse(e.data));
  ws.onclose = () => setStatus("disconnected", false);
  $("#join").style.display = "none"; $("#call").style.display = "block";
};

$("#mic").onclick = () => {
  micOn = !micOn;
  $("#mic").textContent = micOn ? "🎙 Mic on" : "🔇 Mic off";
  $("#mic").classList.toggle("live", micOn);
};
$("#mic").classList.add("live");

// ------------------------------------------------------------ server messages
function onMsg(m) {
  if (m.type === "joined") { me = m.id; setStatus(m.ready ? "models ready" : "warming up models…", m.ready); }
  if (m.type === "status") setStatus("models ready", true);
  if (m.type === "roster") { people = m.people; drawPeople(); }
  if (m.type === "speaking") document.querySelector(`[data-pid="${m.pid}"]`)?.classList.toggle("talk", m.on);
  if (m.type === "live") playLive(m);
  if (m.type === "caption") addCaption(m);
  if (m.type === "caption_tr") addTranslation(m);
  if (m.type === "timing") addTiming(m);
  if (m.type === "audio") playClip(m);
}

function setStatus(t, ok) { $("#status").textContent = t; $("#status").classList.toggle("ok", !!ok); }

function drawPeople() {
  $("#people").innerHTML = people.map((p, i) => `
    <div class="person" data-pid="${p.id}">
      <div class="av" style="background:${COLORS[i % COLORS.length]}">${esc(p.name[0] || "?").toUpperCase()}</div>
      <div><div class="n">${esc(p.name)}${p.id === me ? " (you)" : ""}</div>
      <div class="l lang-${p.lang}">${p.lang === "hi" ? "हिन्दी" : "English"}</div></div>
    </div>`).join("");
}

function addCaption(m) {
  $("#empty")?.remove();
  const div = document.createElement("div");
  div.className = "cap"; div.id = m.cid;
  div.innerHTML = `
    <div class="who"><span><b>${esc(m.speaker)}</b> · <span class="lang-${m.lang}">${m.lang === "hi" ? "हिन्दी" : "English"}</span></span>
      <span class="chip" data-t>…</span></div>
    <div class="orig">${esc(m.heard)}</div>
    <div data-tr></div>
    ${m.locked !== m.heard ? `<div class="lock">🔒 numbers locked: ${esc(m.locked)}</div>` : ""}`;
  $("#feed").appendChild(div);
  $("#feed").scrollTop = 1e9;
}

function addTranslation(m) {
  const box = document.querySelector(`#${m.cid} [data-tr]`);
  if (!box) return;
  const mine = people.find((p) => p.id === me);
  if (mine && mine.lang !== m.lang && box.childElementCount) return;   // show only my language when I have one
  box.insertAdjacentHTML("beforeend", `<div class="tr lang-${m.lang}">${esc(m.text)}</div>`);
}

function addTiming(m) {
  const chip = document.querySelector(`#${m.cid} [data-t]`);
  if (!chip) return;
  chip.textContent = `${(m.first_audio_ms / 1000).toFixed(1)} s`;
  chip.classList.toggle("fast", m.first_audio_ms < 3000);
  chip.title = "first sound after they stopped · " + Object.entries(m.ms).map(([k, v]) => `${k} ${v} ms`).join(" · ");
}

function pcmToBuffer(b64s, sr) {
  const bytes = Uint8Array.from(atob(b64s), (c) => c.charCodeAt(0));
  const i16 = new Int16Array(bytes.buffer);
  const buf = out.createBuffer(1, i16.length, sr);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768;
  return buf;
}

function playClip(m) {
  const buf = pcmToBuffer(m.pcm, m.sr);
  const src = out.createBufferSource(); src.buffer = buf; src.connect(out.destination);
  const at = Math.max(out.currentTime + 0.05, playEnd);
  src.start(at); playEnd = at + buf.duration;
  $("#duplex").textContent = `🔊 ${m.speaker} (${(m.latency_ms / 1000).toFixed(1)} s after they stopped)`;
  setTimeout(() => { if (out.currentTime >= playEnd - 0.05) $("#duplex").textContent = ""; }, (playEnd - out.currentTime) * 1000 + 300);
}

function playLive(m) {
  const buf = pcmToBuffer(m.pcm, 16000);
  const src = out.createBufferSource(); src.buffer = buf; src.connect(out.destination);
  const at = Math.max(out.currentTime + 0.08, liveEnd[m.pid] || 0);
  src.start(at); liveEnd[m.pid] = at + buf.duration;
}

function b64(ab) { let s = ""; const u = new Uint8Array(ab); for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
