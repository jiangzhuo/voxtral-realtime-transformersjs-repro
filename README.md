# transformers.js × Voxtral Realtime: one decoder step too many

`VoxtralRealtimeForConditionalGeneration.generate()` in `@huggingface/transformers` (4.2.0, 4.3.1
and `main` at 511bb61a) runs **one decoder step after the audio chunk iterator is exhausted**. That
step has no audio embedding, and the token it samples is streamed before the library's own
`AudioExhaustedCriteria` stops the loop. The Python implementation never runs such a step.

With the real model that step is a near-tie between `[STREAMING_PAD]`, punctuation and frequent
word-piece suffixes, so a VAD-segmented transcript ends with junk such as `ished`, `ous`, `ized`,
`ch` glued to the last word — or, when the preceding sentence was already published, as a
one-word segment of its own.

## Where

`packages/transformers/src/models/voxtral_realtime/modeling_voxtral_realtime.js`:

- `forward()` calls `fillAudioBuffer()`, which pulls chunks **until the iterator reports `done`**
  and only then sets `stream_exhausted`.
- `addAudioEmbeddings()` then **returns without adding anything when the queue is empty**
  (`if (s.audio_embed_queue.length === 0) return;`), and the decoder runs on the text embedding
  alone.
- `generate()` (`modeling_utils.js`) is forward → sample → `streamer.put()` → stopping criteria, so
  `AudioExhaustedCriteria` fires one step late and the audio-less token is already out.

Python transformers pulls the next chunk after each step and stops on `StopIteration`
(`_update_model_kwargs_for_generation` / `_has_unfinished_sequences`).

## Run

```sh
npm install
npm run repro          # tiny-random model (~15 MB): proves the extra step by accounting
npm run repro:fix      # same, with a StoppingCriteria that stops on the last audio position
npm run repro:real     # onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX q4 (3.2 GB): shows the junk
npm run repro:real:fix
```

Expected (tiny-random, 1 s of audio = 44 audio tokens after the 39-token prefill):

```
audio tokens encoded: 44 (prefill 39)
decoder forwards: 7, generated tokens: 7, audio positions consumed: 45
=> the last 1 step(s) ran with NO audio embedding; the token sampled there was streamed: " apresenta"
```

With `--fix`: 6 forwards, 6 tokens, 44 positions consumed — every token had an audio embedding.

Expected (real model, CPU, greedy — deterministic), the clip cut inside the onset of "ask":

```
text: " And so, my fellow Americans,ished"
last step's top-5: "ished":-1.52  "[STREAMING_PAD]":-2.36  "ous":-2.57  "ized":-2.85  ",":-3.31
```

and with `--fix`: `" And so, my fellow Americans,"`.

## Fix in the caller

Count audio tokens per yielded chunk and sampled tokens in the streamer, and pass a
`StoppingCriteria` that returns true once `prefillTokens - 1 + generated >= audioTokens` and no
further chunk is coming (see `stopOnLastAudio` in `repro.mjs`). The library fix would be to pull
the next chunk after each step and end generation when the iterator is exhausted, as Python does.

`jfk.wav` is the public-domain clip shipped with whisper.cpp / ggml.
