# Audio Input (Alpha)

Granite Switch can accept **audio input** through a single vLLM model load — no
separate speech server, no change to how developers deploy or call the model.

This is an **alpha**: a speech-to-text *cascade*. Audio is transcribed to text by
a small ASR model and the transcript is fed to the LLM as ordinary tokens. It is
intentionally simple and requires no training. The "proper" upgrade (feeding a
trained projection of a speech encoder's embeddings straight into the LLM) reuses
the same hooks — see [Design](#design) below.

## Building an audio-enabled checkpoint

Add `--enable-audio` when composing:

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.0-micro \
  --built-in-adapters core \
  --enable-audio \
  --output ./granite-switch-audio
```

This adds the `<|audio|>` marker token to the tokenizer and writes the audio
settings into `config.json` so the checkpoint is self-describing:

```json
{ "asr_enabled": true, "asr_model_id": null, "asr_device": "cpu" }
```

- `asr_model_id` — HF id of the speech-to-text model (default: a small built-in
  `distil-whisper/distil-small.en`). Override with `--asr-model <hf-id>`, e.g.
  `openai/whisper-small` for multilingual.
- `asr_device` — `cpu` (default) keeps vLLM's GPU KV-cache budget clean; set
  `--asr-device cuda:0` to run transcription on GPU (watch GPU memory).

Audio capability is **gated per checkpoint** by `asr_enabled`: a checkpoint built
without `--enable-audio` reports no audio modality and never loads the ASR model.

## Calling it

### Python (offline)

```python
from granite_switch.vllm import register; register()
from vllm import LLM, SamplingParams
import soundfile as sf

llm = LLM(model="./granite-switch-audio")          # one model load
audio, sr = sf.read("question.wav")                # numpy array + sample rate

out = llm.generate({
    "prompt": "Transcript of the audio: <|audio|>\nAnswer:",
    "multi_modal_data": {"audio": [(audio, sr)]},
}, SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

The `<|audio|>` marker is where the transcript is spliced in.

### OpenAI-compatible server / chat API

```bash
vllm serve ./granite-switch-audio --port 8000
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
resp = client.chat.completions.create(
    model="granite-switch-audio",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Answer the question in the audio."},
        {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}},
    ]}],
)
print(resp.choices[0].message.content)
```

The chat template emits the `<|audio|>` marker for audio content parts
(`audio` / `input_audio` / `audio_url`), so the processor splices the transcript
in automatically — callers send standard chat messages, no manual marker needed.

## Design

Per request, before the scheduler allocates KV cache:

1. vLLM's multimodal pipeline hands the audio to our processor
   (`granite_switch.vllm.audio`).
2. The processor runs ASR → transcript → token ids.
3. A `PromptReplacement` swaps the `<|audio|>` marker for those transcript token
   ids. The scheduler then sizes KV for the **real** length — the audio "window"
   is variable and decided at runtime, not reserved in advance.
4. The model's `embed_multimodal` supplies embeddings for those positions. In the
   alpha that is simply the transcript's own token embeddings (identical to
   embedding them as text). **This is the seam the future encoder reuses:** swap
   `embed_multimodal` to return `projection(speech_encoder(audio))` and the rest
   of the machinery is unchanged.

The decoder, switch, and LoRA paths are untouched — they only ever see text
tokens.

## Limitations (alpha)

- **Cascade, not end-to-end.** Prosody/emotion/uncertainty are lost; ASR errors
  propagate to the LLM. Two models run sequentially (ASR then LLM).
- **English by default** (`distil-whisper/distil-small.en`). Use `--asr-model`
  for multilingual.
- **Audio runs on the base model.** Combining audio with an adapter (adapter
  control tokens in an audio request) is future work.
- One audio clip per request.

## Tests

- `tests/unit/test_asr.py` — CPU unit tests for the ASR backend (audio coercion,
  resampling, transcription with a mocked pipeline). No GPU/vLLM required.
- End-to-end (GPU): compose an `--enable-audio` checkpoint, then an audio request
  through vLLM produces an answer and text-only requests are unaffected.
