// Minimal reproduction: VoxtralRealtimeForConditionalGeneration.generate() runs one decoder
// step AFTER the audio chunk iterator is exhausted, with no audio embedding added, and streams
// the token sampled at that step. Python transformers stops before any such step.
//
//   node repro.mjs            tiny-random model (fast; proves the extra step by accounting)
//   node repro.mjs --real     onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX q4 (shows the junk token)
//   node repro.mjs --fix      add a StoppingCriteria that ends the run on the last audio position
//
// Set VX_MODEL_DIR to a directory holding <model id>/ to load from disk instead of the Hub.
import { readFileSync } from 'node:fs';
import {
  env, BaseStreamer, StoppingCriteria,
  VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor,
} from '@huggingface/transformers';

const REAL = process.argv.includes('--real');
const FIX = process.argv.includes('--fix');
const MODEL = REAL
  ? 'onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX'
  : 'onnx-internal-testing/tiny-random-VoxtralRealtimeForConditionalGeneration';
const DTYPE = REAL ? 'q4' : 'fp32';
if (process.env.VX_MODEL_DIR) {
  env.localModelPath = process.env.VX_MODEL_DIR;
  env.allowRemoteModels = false;
}

// jfk.wav (16 kHz mono, public domain): speech up to 3.36 s — "And so, my fellow Americans," and
// the first 80 ms of "ask" — then 7 tokens (560 ms) of digital zeros, the way a VAD-driven
// client pads an utterance end. The tiny-random model gets a 1 s sine, like the unit test.
function wav16k(path) {
  const b = readFileSync(path);
  let off = 12, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), size = b.readUInt32LE(off + 4);
    if (id === 'data') { data = { start: off + 8, size }; break; }
    off += 8 + size + (size & 1);
  }
  const n = data.size / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = b.readInt16LE(data.start + i * 2) / 32768;
  return out;
}
function audioFor(processor) {
  const sr = processor.feature_extractor.config.sampling_rate;
  if (!REAL) return Float32Array.from({ length: sr }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / sr));
  const clip = wav16k(new URL('./jfk.wav', import.meta.url));
  const speech = clip.subarray(0, Math.round(3.36 * sr));
  const out = new Float32Array(Math.round(0.5 * sr) + speech.length + 7 * processor.raw_audio_length_per_tok);
  out.set(speech, Math.round(0.5 * sr));
  return out;
}

const model = await VoxtralRealtimeForConditionalGeneration.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });
const processor = await VoxtralRealtimeProcessor.from_pretrained(MODEL);
const tokenizer = processor.tokenizer;
const audio = audioFor(processor);

// The streaming recipe from the transformers docs / transformers.js tests: a left-padded first
// chunk, then one chunk per audio token, each starting one STFT half-window before its frames.
const { hop_length, n_fft } = processor.feature_extractor.config;
const winHalf = Math.floor(n_fft / 2);
const first = await processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true, is_first_audio_chunk: true });
const prefillTokens = first.input_ids.dims[1];

let encoderTokens = 0; // audio tokens the encoder produced
let lastChunkYielded = false;
const enc = model.sessions['audio_encoder'];
const encRun = enc.run.bind(enc);
enc.run = async (...a) => { const r = await encRun(...a); encoderTokens += r.audio_embeds.dims[1]; return r; };

async function* chunks() {
  yield first.input_features;
  let melFrameIdx = processor.num_mel_frames_first_audio_chunk;
  while (true) {
    const start = melFrameIdx * hop_length - winHalf;
    const end = start + processor.num_samples_per_audio_chunk;
    if (audio.length < end) break;
    const c = await processor(audio.slice(start, end), { is_streaming: true, is_first_audio_chunk: false });
    melFrameIdx += c.input_features.dims[2];
    if (audio.length < melFrameIdx * hop_length - winHalf + processor.num_samples_per_audio_chunk) lastChunkYielded = true;
    yield c.input_features;
  }
  lastChunkYielded = true;
}

let forwards = 0, lastTop = null;
const forward = model.forward.bind(model);
model.forward = async (inputs) => {
  forwards++;
  const out = await forward(inputs);
  const [, seq, vocab] = out.logits.dims;
  const row = out.logits.data.subarray((seq - 1) * vocab, seq * vocab);
  let mx = -Infinity; for (let i = 0; i < row.length; i++) if (row[i] > mx) mx = row[i];
  let s = 0; for (let i = 0; i < row.length; i++) s += Math.exp(row[i] - mx);
  const lse = mx + Math.log(s);
  lastTop = Array.from({ length: row.length }, (_, i) => i).sort((a, b) => row[b] - row[a]).slice(0, 5).map((i) => [i, +(row[i] - lse).toFixed(2)]);
  return out;
};

const streamed = [];
let generated = 0;
const streamer = new (class extends BaseStreamer {
  put(v) { streamed.push(...v[0]); if (streamed.length > prefillTokens) generated++; }
  end() {}
})();

// What Python transformers does: stop once the token of the last audio position is out. The
// prefill forward consumes `prefillTokens` positions and samples the first token; each later
// token consumes one more.
const stopOnLastAudio = new (class extends StoppingCriteria {
  _call(ids) { return ids.map(() => lastChunkYielded && prefillTokens - 1 + generated >= encoderTokens); }
})();

await model.generate({
  input_ids: first.input_ids,
  input_features: chunks(),
  max_new_tokens: 4096,
  streamer,
  ...(FIX ? { stopping_criteria: stopOnLastAudio } : {}),
});

const gen = streamed.slice(prefillTokens);
const consumed = prefillTokens - 1 + gen.length; // audio positions the decoder steps consumed
const piece = (id) => JSON.stringify(tokenizer.decode([BigInt(id)], { skip_special_tokens: false }));
console.log(`model: ${MODEL} (${DTYPE})${FIX ? ', with the stop criterion' : ''}`);
console.log(`audio tokens encoded: ${encoderTokens} (prefill ${prefillTokens})`);
console.log(`decoder forwards: ${forwards}, generated tokens: ${gen.length}, audio positions consumed: ${consumed}`);
if (consumed > encoderTokens) {
  console.log(`=> the last ${consumed - encoderTokens} step(s) ran with NO audio embedding; the token sampled there was streamed: ${piece(Number(gen[gen.length - 1]))}`);
} else {
  console.log('=> every generated token came from a step that had an audio embedding');
}
console.log(`last step's top-5: ${lastTop.map(([i, lp]) => `${piece(i)}:${lp}`).join('  ')}`);
console.log(`text: ${JSON.stringify(tokenizer.decode(gen, { skip_special_tokens: true }))}`);
