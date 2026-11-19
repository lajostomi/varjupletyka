// Varjúpletyka — listens through the mic, detects crows with YAMNet (TensorFlow.js),
// and shows a message bubble each time a crow is heard.
// States: intro -> listening -> active -> error  (set on #app[data-state])

const app = document.getElementById('app');
const earBtn = document.getElementById('ear');
const errorEl = document.getElementById('error');
const CORNERS = ['tl', 'tr', 'bl', 'br'];   // top/bottom + left/right

// ---- Settings (tweak here) ----
const params = new URLSearchParams(location.search);
const THRESHOLD = parseFloat(params.get('t')) || 0.2;   // per-window score that counts as a "hit" (quiet-room noise peaked at 0.05, clear crows 0.3-1.0, weaker/farther ones 0.2-0.3; override: ?t=0.3)
const HITS_NEEDED = 2, HITS_OF = 3;  // fire when HITS_NEEDED of the last HITS_OF windows are hits
const DEMO = params.has('demo');     // ?demo: fake detections at random intervals, no microphone
const CALIB = params.has('calibrate');   // ?calibrate: label what you hear (crow / no crow) and get a suggested threshold
const DEBUG = params.has('debug') || CALIB;   // ?debug: live crow scores on screen (audio still works)
const PANEL = params.has('panel');   // ?panel: button panel to jump between states and spawn crows (hidden by default)
// Level boost: the mic is asked NOT to apply automatic gain, so quiet or distant sounds arrive very faint.
// Before the model sees a window we scale it up to a steady loudness (never in near-silence, never more than MAX_GAIN).
const TARGET_RMS = 0.05;            // loudness the window is scaled to
const MAX_GAIN = params.has('gain') ? parseFloat(params.get('gain')) : 20;   // ?gain=1 switches the boost off
const SILENCE_RMS = 0.0015;         // quieter than this = silence: not boosted (would only amplify mic hiss)
const AGC = params.has('agc');      // ?agc: let the browser apply its own automatic gain instead (test option)
const WINDOW = 15600;             // samples YAMNet needs (0.975 s @ 16 kHz)
const HOP_MS = 500;               // how often we run the model
const COOLDOWN_MS = 2500;         // min. time between two 'crowDetected' events
const FADE_AFTER_MS = 12000;     // no new detection for this long -> crows and bubbles fade out
const MAX_SPEAKERS = 2;           // at most this many bubbles (speaking crows) at once; the oldest goes quiet
const FADE_MS = 1000;             // fade duration (keep in sync with style.css)
const LISTEN_TIMEOUT_MS = 25000;  // listening with no crow for this long -> error screen (keeps listening)

const ERR_NO_CROWS = 'Most nincsenek varjak\na közeledben.';
const ERR_MIC_LOST = 'A mikrofon kapcsolata\nmegszakadt. Koppints\na fülre újra.';
const ERR_NO_MIC = 'Az oldalnak szüksége van\na mikrofonra, hogy\nmeghallja a varjakat.';

let messages = ['*varjú üzenet*'];
let bag = [], lastMsg;            // messages not shown yet in this round
let speakers = [];                // corners whose bubble is showing, oldest first
let live = {};                    // corner -> { crow, bubble } for crows currently on screen
let fadeTimer, fadeEnd;
let stereo = false, sides = [];   // stereo mic? recent [left, right] energies per audio block
let ctx, stream, node, model, timer;
let ring = new Float32Array(0);   // recent 16 kHz audio
let lastCrow = 0, listenStart = 0, busy = false;
let classNames = [];              // YAMNet display names, index = class id
let crowIds = [112, 113];         // overwritten from the class map ("Crow", "Caw")
let hits = [];                    // last HITS_OF window results (true = score over threshold)

// ---- YAMNet class map: find the ids of "Crow" and "Caw" by name ----
fetch('models/yamnet/yamnet_class_map.csv').then(r => r.text()).then(t => {
  t.split(/\r?\n/).slice(1).forEach(line => {
    const m = line.match(/^(\d+),[^,]*,(.*)$/);
    if (m) classNames[+m[1]] = m[2].replace(/^"|"$/g, '');
  });
  const ids = ['Crow', 'Caw'].map(n => classNames.indexOf(n)).filter(i => i >= 0);
  if (ids.length) crowIds = ids;
}).catch(() => {});

// ---- Layout unit: 1 Figma px (frame 393x852) scaled to the screen ----
function setUnit() {
  const u = Math.min(Math.min(innerWidth, 520) / 393, innerHeight / 852);
  document.documentElement.style.setProperty('--u', u + 'px');
}
setUnit();
addEventListener('resize', () => { setUnit(); fitAll(); });
if (document.fonts) document.fonts.ready.then(() => fitAll());   // text width changes once the web font is in


// The one place that changes state. Leaving 'active' clears crows and bubbles.
function setState(s, errorText) {
  if (s !== 'active') clearCrows();
  if (s === 'intro' && app.dataset.state !== 'intro') {   // plug the AirPod back in (reverse of the take-out)
    app.classList.remove('airpod-out');
    app.classList.add('airpod-in');
    setTimeout(() => app.classList.remove('airpod-in'), AIRPOD_MS);
  }
  if (s === 'error') errorEl.textContent = errorText || ERR_NO_CROWS;
  app.dataset.state = s;
  earBtn.setAttribute('aria-label', s === 'intro' ? 'Koppints a fülhallgatóra, és hallgasd a varjakat'
    : s === 'error' ? 'Koppints ide az újrakezdéshez' : 'Koppints ide a hallgatás befejezéséhez');
}

// ---- Messages: one per line in messages.txt (blank lines and lines starting with # are ignored) ----
// cache: 'no-store' so an edited file shows up on the next reload
fetch('messages.txt', { cache: 'no-store' }).then(r => r.text()).then(t => {
  const lines = t.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
  if (lines.length) { messages = lines; bag = []; }
}).catch(() => {});

// ---- Start ----
// Intro: tap the AirPod -> it slides out (~800 ms) -> ask for the mic.
// Error: tap the ear to try again. Keyboard activation of the ear button also starts.
const airpod = document.querySelector('.airpod');
const AIRPOD_MS = 900;   // keep in sync with the airpod animations in style.css
const sleep = ms => new Promise(r => setTimeout(r, ms));
let starting = false, running = false, ending = false, wakeLock = null, demoTimer;

airpod.addEventListener('click', e => { e.stopPropagation(); start(); });
// Tap the ear during a session -> end it (crows fade out, AirPod plugs back in). Tap the AirPod -> start again.
// Error without a session (mic denied etc.): tap the ear to retry. Keyboard activation in the intro also starts.
earBtn.addEventListener('click', e => {
  if (running && app.dataset.state !== 'intro') endSession();
  else if (app.dataset.state === 'error' || e.detail === 0) start();
});

async function start() {
  const from = app.dataset.state;
  if ((from !== 'intro' && from !== 'error') || starting) return;
  starting = true;

  // iOS Safari: the AudioContext must be created and resumed synchronously inside the tap
  if (!DEMO) {
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC();
    ctx.resume();
    // iOS may interrupt audio (call, Siri, lock screen): try to resume when we are visible again
    ctx.onstatechange = () => { if (ctx && running && !document.hidden && ctx.state !== 'running') ctx.resume().catch(() => {}); };
  }

  if (from === 'intro') {                       // animate the AirPod out of the ear first
    app.classList.remove('airpod-in');
    app.classList.add('airpod-out');
    await sleep(AIRPOD_MS);
  }

  if (DEMO) {                                   // presentation mode: no mic, fake detections
    setState('listening'); starting = false; running = true;
    listenStart = performance.now();
    requestWake(); scheduleDemo();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: AGC,
               channelCount: { ideal: 2 } }     // ask for stereo; most phones still give mono
    });
  } catch (e) { return fail(ERR_NO_MIC); }     // denied, or no mic / insecure page
  stereo = (stream.getAudioTracks()[0].getSettings().channelCount || 1) === 2;
  sides = [];
  // Mic unplugged / taken away by another app -> error screen
  stream.getAudioTracks().forEach(t => t.addEventListener('ended', () => { if (stream) fail(ERR_MIC_LOST); }));

  setState('listening');
  starting = false;
  running = true;
  requestWake();

  try {
    model = model || await tf.loadGraphModel('models/yamnet/model.json');
  } catch (e) { console.error(e); return fail(ERR_NO_CROWS); }

  // Collect mic samples (ScriptProcessor: deprecated but works on iOS Safari + Android Chrome)
  const src = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(4096, 2, 1);
  node.onaudioprocess = e => {
    const b = e.inputBuffer;
    push(b.getChannelData(0), ctx.sampleRate);
    if (stereo) trackSides(b.getChannelData(0), b.getChannelData(1));
  };
  const mute = ctx.createGain(); mute.gain.value = 0;   // keep node alive without echoing the mic
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);

  ring = new Float32Array(0);
  listenStart = performance.now();
  lastCrow = 0;
  hits = [];
  timer = setInterval(analyse, HOP_MS);
  running = true;
}

// Downsample to 16 kHz (simple averaging) and keep the last WINDOW samples
function push(buf, rate) {
  const ratio = rate / 16000;
  const n = Math.floor(buf.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let s = 0; for (let j = a; j < b; j++) s += buf[j];
    out[i] = s / (b - a);
  }
  const merged = new Float32Array(Math.min(WINDOW, ring.length + n));
  const all = new Float32Array(ring.length + n);
  all.set(ring); all.set(out, ring.length);
  merged.set(all.subarray(all.length - merged.length));
  ring = merged;
}

// Scale a window to TARGET_RMS (clipped to +-1). Returns the scaled copy and what was done, for ?debug.
let lastLevel = 0, lastGain = 1;
function boost(x) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  const rms = Math.sqrt(sum / x.length);
  lastLevel = rms; lastGain = 1;
  if (rms < SILENCE_RMS || MAX_GAIN <= 1) return x;
  const g = lastGain = Math.min(MAX_GAIN, Math.max(1, TARGET_RMS / rms));
  if (g === 1) return x;
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = Math.max(-1, Math.min(1, x[i] * g));
  return y;
}

// ---- Run YAMNet on the latest window ----
async function analyse() {
  if (busy || document.hidden || !ctx || ctx.state !== 'running' || ring.length < WINDOW) return;
  busy = true;
  try {
    const scores = tf.tidy(() => {
      let out = model.predict(tf.tensor1d(boost(ring)));
      if (Array.isArray(out)) out = out[0];       // [frames, 521]
      return out.max(0);                          // best score per class
    });
    const data = await scores.data();
    scores.dispose();
    const crowScores = crowIds.map(c => data[c]);
    const best = Math.max(...crowScores);
    const now = performance.now();

    // 2-of-3 rule: keep the last HITS_OF results, fire when enough of them are hits
    hits.push(best >= THRESHOLD);
    if (hits.length > HITS_OF) hits.shift();
    if (DEBUG) showDebug(crowScores, best, data);
    if (CALIB) calibRecord(best);

    if (hits.filter(Boolean).length >= HITS_NEEDED && now - lastCrow > COOLDOWN_MS) {
      lastCrow = now; hits = [];
      document.dispatchEvent(new CustomEvent('crowDetected', { detail: { score: best } }));
    } else if (app.dataset.state === 'listening' && now - listenStart > LISTEN_TIMEOUT_MS) setState('error', ERR_NO_CROWS);   // keep listening
  } catch (e) { console.error(e); }
  busy = false;
}

// ---- Live scores (?debug): crow/caw scores, last windows, threshold, loudest class ----
let dbg, peak = 0;
function showDebug(crowScores, best, data) {
  if (!dbg) {
    dbg = document.createElement('pre');
    dbg.id = 'scores';
    app.append(dbg);
  }
  peak = Math.max(peak, best);
  const top = data.indexOf(Math.max(...data));
  const names = crowIds.map(c => classNames[c] || '#' + c);
  dbg.textContent =
    names.map((n, i) => `${n} ${crowScores[i].toFixed(2)}`).join('  ') + `  peak ${peak.toFixed(2)}
` +
    `thr ${THRESHOLD}  hits [${hits.map(h => h ? 'x' : '.').join('')}] need ${HITS_NEEDED}/${HITS_OF}
` +
    `top: ${classNames[top] || '#' + top} ${data[top].toFixed(2)}\n` +
    `level ${lastLevel.toFixed(4)}  gain x${lastGain.toFixed(1)}  (max x${MAX_GAIN})`;
}

// ---- Stereo: remember left/right energy of the last ~1 s (only if the mic gives 2 channels) ----
function trackSides(l, r) {
  let a = 0, b = 0;
  for (let i = 0; i < l.length; i++) { a += l[i] * l[i]; b += r[i] * r[i]; }
  sides.push([a, b]);
  if (sides.length > 12) sides.shift();           // 12 blocks of 4096 samples ~ 1 s
}
function heardSide() {                            // 'l' | 'r' | null (mono or no clear difference)
  if (!stereo || !sides.length) return null;
  const L = sides.reduce((s, x) => s + x[0], 0), R = sides.reduce((s, x) => s + x[1], 0);
  if (Math.abs(L - R) < 0.1 * (L + R)) return null;
  return L > R ? 'l' : 'r';
}

// Free corner: on the louder side if known (top/bottom random), otherwise any free corner
function pickCorner() {
  const free = CORNERS.filter(c => !live[c]);
  if (!free.length) return null;
  const side = heardSide();
  const onSide = side ? free.filter(c => c[1] === side) : free;
  const pool = onSide.length ? onSide : free;
  return pool[Math.floor(Math.random() * pool.length)];
}

// Random message, none repeated until every message has been shown
function nextMessage() {
  if (!bag.length) bag = [...messages];
  let i = Math.floor(Math.random() * bag.length);
  if (bag[i] === lastMsg && bag.length > 1) i = (i + 1) % bag.length;   // no repeat across rounds
  lastMsg = bag.splice(i, 1)[0];
  return lastMsg;
}

const rnd = (a, b) => a + Math.random() * (b - a);

// Bubble hugs its text: if the text wrapped, trim the box to the longest line so the padding is the same on every side
function fitBubble(b) {
  b.style.width = '';
  const txt = b.firstChild;
  const range = document.createRange();
  range.selectNodeContents(txt);
  const widest = Math.max(0, ...[...range.getClientRects()].map(r => r.width));
  const cs = getComputedStyle(b);
  if (widest) b.style.width = Math.ceil(widest + parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight)) + 'px';
}
function fitAll() {
  Object.values(live).forEach(o => { if (o.bubble) fitBubble(o.bubble); });
  speakers.forEach(c => live[c] && live[c].bubble && avoidOverlap(c));
}

// Two speaking bubbles must never overlap: measure them and push the new one towards the ear (top row down, bottom row up)
function avoidOverlap(corner) {
  const me = live[corner].bubble;
  const box = (el, dy = 0) => ({ l: el.offsetLeft, r: el.offsetLeft + el.offsetWidth, t: el.offsetTop + dy, b: el.offsetTop + el.offsetHeight + dy });
  const gap = 10 * (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--u')) || 1);
  const others = speakers.filter(c => c !== corner && live[c] && live[c].bubble)
    .map(c => box(live[c].bubble));          // their offsets already include their own --dy
  const down = corner[0] === 't';
  let dy = 0;
  for (let pass = 0; pass < others.length + 1; pass++) {
    const m = box(me, dy);
    const hit = others.find(o => m.l < o.r && m.r > o.l && m.t < o.b && m.b > o.t);
    if (!hit) break;
    dy += down ? hit.b + gap - m.t : hit.t - gap - m.b;
  }
  me.style.setProperty('--dy', dy + 'px');
}

// A crow stops speaking: its bubble fades out, the crow itself stays
function silence(corner) {
  const o = live[corner];
  speakers = speakers.filter(c => c !== corner);
  if (!o || !o.bubble) return;
  const b = o.bubble; o.bubble = null;
  b.classList.remove('on');
  setTimeout(() => b.remove(), 600);
}

// ---- Place a crow (and its bubble) in a corner; max 4 crows, one per corner, max 2 speaking ----
function spawnCrow(corner, text) {
  if (ending) return;
  setState('active');
  if (!live[corner]) {
    while (speakers.length >= MAX_SPEAKERS) silence(speakers[0]);   // new rule: only 2 bubbles at once

    // wrapper covers the screen and pops (scales up) from the corner; the image inside keeps its Figma pose
    const crow = document.createElement('div');
    crow.className = 'crow-wrap ' + corner;
    const img = new Image();
    img.src = 'assets/crow-head.png'; img.alt = ''; img.className = 'crow ' + corner;
    crow.append(img);

    // Bubble = text + zig-zag tail (assets/bubble-tail.svg) pointing at the crow's beak
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + corner;
    const txt = document.createElement('span');
    txt.textContent = text || nextMessage();
    const tail = new Image();
    tail.src = 'assets/bubble-tail.svg'; tail.alt = ''; tail.className = 'tail';
    bubble.append(txt, tail);

    // Every crow looks a little different: tilt, size, offset, and when / how fast it pops in
    const u = n => `calc(${n.toFixed(1)} * var(--u))`;
    const delay = rnd(0, 350);
    const jx = u(rnd(-8, 8)), jy = u(rnd(-8, 8));
    for (const el of [img, bubble]) { el.style.setProperty('--jx', jx); el.style.setProperty('--jy', jy); }   // bubble follows its crow
    img.style.setProperty('--jr', rnd(-4, 4).toFixed(1) + 'deg');
    img.style.setProperty('--js', rnd(0.95, 1.06).toFixed(2));
    crow.style.setProperty('--delay', delay.toFixed(0) + 'ms');
    crow.style.setProperty('--dur', rnd(300, 500).toFixed(0) + 'ms');

    app.append(crow, bubble);
    live[corner] = { crow, bubble };
    speakers.push(corner);
    fitBubble(bubble);
    avoidOverlap(corner);
    requestAnimationFrame(() => requestAnimationFrame(() => crow.classList.add('on')));   // next frame: pop in
    setTimeout(() => { if (live[corner] && live[corner].bubble === bubble) bubble.classList.add('on'); }, delay + 350);   // speaks once it has popped in
  }
  scheduleFade();
}

// 'crowDetected' (fired by analyse): new crow in a free corner; if all 4 are taken just keep them alive
document.addEventListener('crowDetected', crowHeard);
function crowHeard() {
  const c = pickCorner();
  if (c) spawnCrow(c); else scheduleFade();
}

// ---- Fade everything out FADE_AFTER_MS after the last detection, then back to 'listening' ----
function scheduleFade() {
  clearTimeout(fadeTimer);
  fadeTimer = setTimeout(() => {
    fadeCrows();
    fadeEnd = setTimeout(() => {
      if (app.dataset.state === 'active') { setState('listening'); listenStart = performance.now(); }
      else clearCrows();
    }, FADE_MS + 50);
  }, FADE_AFTER_MS);
}

function fadeCrows() {
  Object.values(live).forEach(o => { o.crow.classList.add('fade'); if (o.bubble) o.bubble.classList.add('fade'); });
}

function clearCrows() {
  clearTimeout(fadeTimer); clearTimeout(fadeEnd);
  Object.values(live).forEach(o => { o.crow.remove(); if (o.bubble) o.bubble.remove(); });
  live = {}; speakers = [];
}

// ---- Error screen: stop everything, tap the ear to try again ----
function fail(text) {
  starting = false;
  stop();
  setState('error', text);
}

// ---- Screen Wake Lock: keep the screen on while listening (re-acquired when the tab returns) ----
async function requestWake() {
  if (wakeLock && !wakeLock.released) return;
  try {
    if ('wakeLock' in navigator) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => { wakeLock = null; });   // the browser drops it when the tab is hidden
    }
  } catch (e) {}
}

// Tab hidden/shown: re-acquire the wake lock, resume audio, check the mic is still alive
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState !== 'visible' || !running) return;
  if (stream && stream.getAudioTracks().every(t => t.readyState === 'ended')) return fail(ERR_MIC_LOST);
  if (ctx && ctx.state !== 'running') { try { await ctx.resume(); } catch (e) {} }
  listenStart = performance.now();               // time in the background doesn't count as "no crows"
  hits = [];
  requestWake();
});

// ---- ?demo: pretend crows are around (first one after ~2 s, then every 3-10 s) ----
function scheduleDemo(first = true) {
  demoTimer = setTimeout(() => {
    if (!document.hidden) document.dispatchEvent(new CustomEvent('crowDetected', { detail: { score: 0.9, demo: true } }));
    scheduleDemo(false);
  }, first ? 2000 : 3000 + Math.random() * 7000);
}

// End the session on request: stop listening right away, fade crows and bubbles out, then plug the AirPod back in
async function endSession() {
  if (ending) return;
  ending = true;
  clearTimeout(fadeTimer); clearTimeout(fadeEnd);
  fadeCrows();
  stop();
  await sleep(FADE_MS + 50);
  setState('intro');                            // clears what is left and plays the plug-in animation
  ending = false;
}

function stop() {
  running = false;
  clearInterval(timer); clearTimeout(demoTimer);
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  if (node) { node.disconnect(); node.onaudioprocess = null; node = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (ctx) { ctx.close(); ctx = null; }
}

// ---- ?panel: jump between states, spawn crows (hidden unless the address has ?panel) ----
if (PANEL) {
  const bar = document.createElement('div');
  bar.id = 'debug';
  const btn = (label, fn) => {
    const b = document.createElement('button');
    b.textContent = label; b.type = 'button';
    b.addEventListener('click', e => { e.stopPropagation(); fn(); });
    bar.append(b);
  };
  ['intro', 'listening', 'active', 'error'].forEach(n => btn(n, () => (n === 'intro' && running ? endSession() : setState(n))));
  [['tl', '↖'], ['tr', '↗'], ['bl', '↙'], ['br', '↘']].forEach(([c, a]) => btn('crow ' + a, () => spawnCrow(c)));
  btn('crowDetected', crowHeard);                // same path as a real detection
  btn('clear', () => setState('listening'));
  app.append(bar);
}

// ---- ?calibrate: label what you hear, get a suggested THRESHOLD ----
// Press "crow" while a crow sound is playing, "no crow" for background (speech, traffic, silence), "pause" in between.
// Every analysed window is stored with the current label; bursts of the same label form one segment.
// The suggestion replays the real 2-of-3 rule on the stored scores for thresholds 0.05 - 0.95.
let calib = [], calibLabel = null, calibSeg = 0, calibOut;

function calibRecord(score) {
  if (calibLabel) calib.push({ label: calibLabel, seg: calibSeg, s: score });
  calibShow();
}

// the real rule on one segment's scores: true where >= HITS_NEEDED of the last HITS_OF windows are over thr
function calibRule(scores, thr) {
  const h = [];
  return scores.map(v => { h.push(v >= thr); if (h.length > HITS_OF) h.shift(); return h.filter(Boolean).length >= HITS_NEEDED; });
}

function calibSegments() {
  const segs = {};
  calib.forEach(w => { (segs[w.seg] = segs[w.seg] || { label: w.label, scores: [] }).scores.push(w.s); });
  return Object.values(segs);
}

function calibSuggest() {
  const segs = calibSegments();
  const noise = segs.filter(g => g.label === 'noise'), crow = segs.filter(g => g.label === 'crow');
  if (!noise.length || !crow.length) return null;
  const rows = [];
  for (let t = 5; t <= 95; t++) {
    const thr = t / 100;
    const fp = noise.reduce((n, g) => n + calibRule(g.scores, thr).filter(Boolean).length, 0);   // noise windows that would fire
    const hit = crow.filter(g => calibRule(g.scores, thr).some(Boolean)).length;                 // crow bursts that would be caught
    rows.push({ thr, fp, hit });
  }
  const bestFp = Math.min(...rows.map(r => r.fp));
  const bestHit = Math.max(...rows.filter(r => r.fp === bestFp).map(r => r.hit));
  const ok = rows.filter(r => r.fp === bestFp && r.hit === bestHit);      // all equally good thresholds
  return { thr: ok[Math.floor(ok.length / 2)].thr, lo: ok[0].thr, hi: ok[ok.length - 1].thr, fp: bestFp, hit: bestHit, crowSegs: crow.length };
}

const pct = (arr, p) => arr.length ? [...arr].sort((a, b) => a - b)[Math.min(arr.length - 1, Math.floor(p * arr.length))] : 0;

function calibShow() {
  if (!calibOut) return;
  const line = l => {
    const v = calib.filter(w => w.label === l).map(w => w.s);
    return `${l.padEnd(5)} ${String(v.length).padStart(4)} windows  max ${Math.max(0, ...v).toFixed(2)}  p95 ${pct(v, .95).toFixed(2)}  median ${pct(v, .5).toFixed(2)}`;
  };
  const sg = calibSuggest();
  calibOut.textContent = `label: ${calibLabel || 'pause'}\n${line('crow')}\n${line('noise')}\n` +
    (sg ? `suggested threshold ${sg.thr.toFixed(2)} (works from ${sg.lo.toFixed(2)} to ${sg.hi.toFixed(2)}): ` +
          `crow bursts caught ${sg.hit}/${sg.crowSegs}, false fires in noise ${sg.fp}\nuse ?t=${sg.thr.toFixed(2)}`
        : 'need at least one crow burst and some noise');
}

if (CALIB) {
  calibOut = document.createElement('pre');
  calibOut.id = 'calib-out';
  const bar = document.createElement('div');
  bar.className = 'calib-bar';
  const btn = (label, fn) => {
    const b = document.createElement('button');
    b.textContent = label; b.type = 'button';
    b.addEventListener('click', e => { e.stopPropagation(); fn(); calibShow(); });
    bar.append(b);
  };
  const setLabel = l => () => { calibLabel = l; calibSeg++; };           // a new burst starts with every label change
  btn('crow', setLabel('crow'));
  btn('no crow', setLabel('noise'));
  btn('pause', setLabel(null));
  btn('reset', () => { calib = []; calibSeg = 0; });
  btn('copy data', () => navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(calib)));
  app.append(calibOut, bar);
  calibShow();
}
