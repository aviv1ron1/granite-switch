# Workflow: Run Inference

[← Quickstart](../quickstart.md)

There are three ways to invoke adapter functions. **Mellea + vLLM is the
recommended path** — it handles control tokens, prompt rewriting, and constrained
decoding so you work with typed function calls, not raw tokens.

## Option 1 — vLLM + Mellea (recommended, production)

Start the server:

```bash
pip install "granite-switch[vllm]" mellea
python -m vllm.entrypoints.openai.api_server \
  --model ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

Call an adapter function:

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.chat import Message
from mellea.stdlib.components.intrinsic.guardian import guardian_check
from mellea.stdlib.context import ChatContext

backend = OpenAIBackend(
    model_id="ibm-granite/granite-switch-4.1-3b-preview",
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
backend.register_embedded_adapter_model("ibm-granite/granite-switch-4.1-3b-preview")

ctx = ChatContext().add(Message("user", "Group X people are all lazy."))
score = guardian_check(ctx, backend, "social_bias", scoring_schema="user_prompt")
print(f"social_bias score: {score:.3f}")   # => 0.964
```

Mellea maps the requested capability (`social_bias`) to its control token,
places it correctly (ALoRA vs LoRA placement), and enforces the output schema at
the token level. See the guide
[`tutorials/guides/mellea_with_granite_switch.md`](../../tutorials/guides/mellea_with_granite_switch.md).

## Option 2 — HuggingFace (prototyping)

```python
from granite_switch.hf import GraniteSwitchForCausalLM
from transformers import AutoTokenizer

model = GraniteSwitchForCausalLM.from_pretrained("./my-model")
tok = AutoTokenizer.from_pretrained("./my-model")

# The chat template accepts adapter_name= to place the control token for you.
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "..."}],
    adapter_name="answerability",
    add_generation_prompt=True,
    tokenize=False,
)
```

The chat template (built by `composer/tokenizer_setup.py`) inserts the right
control token given `adapter_name=`. Inspect `model._last_adapter_indices` after
a forward to see the per-token selection.

## Option 3 — raw control tokens

You can place a control token directly in the input; the switch will detect it.
This is the lowest-level path and usually only used in tests. Token ids live in
`config.adapter_token_ids` and the name mapping in `adapter_index.json`.

## Control-token placement rules

From the chat template logic (`tokenizer_setup.configure_chat_template`):

- **ALoRA adapters**: control token placed immediately before the adapter's
  `alora_invocation_tokens` (matched inside the user message) or right before the
  generation prompt.
- **LoRA adapters**: control token placed at sequence position 0 (prefix).

Both cases are made KV-safe by [token exchange](../architecture/token-exchange.md).

## Pre-download for faster startup

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 hf download ibm-granite/granite-switch-4.1-3b-preview
```

## Tutorials to try

- [Hello Mellea](../../tutorials/notebooks/hello_mellea.ipynb) — 5 min
- [RAG Flow](../../tutorials/notebooks/rag_flow.ipynb) — rewrite + answerability + citations + guardians
- [aLoRA vs LoRA race](../../tutorials/notebooks/alora_vs_lora_race.ipynb) — throughput comparison

See [`tutorials/README.md`](../../tutorials/README.md) for the full list and
guided learning paths.
