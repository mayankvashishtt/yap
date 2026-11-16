// Runs in the audio thread. The AudioContext is created at 16 kHz, so the browser resamples the mic.
// We batch 512 samples (32 ms) and hand them to the page as Int16.
class Mic extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(512); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let peak = 0;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      peak = Math.max(peak, Math.abs(s));
      this.buf[this.n++] = s * 32767;
      if (this.n === 512) { this.port.postMessage({ pcm: this.buf.buffer.slice(0), peak }); this.n = 0; peak = 0; }
    }
    return true;
  }
}
registerProcessor("mic", Mic);
