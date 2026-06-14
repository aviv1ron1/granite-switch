# SPDX-License-Identifier: Apache-2.0
"""Audio (ASR) preprocessing for the Granite Switch vLLM backend.

This package implements the **alpha** audio pathway: a speech-to-text *cascade*.
Audio is transcribed to text by a small ASR model and the transcript tokens are
spliced into the prompt before the decoder runs. The Granite Switch decoder is
unchanged and only ever sees text tokens — there is no in-model audio encoder yet
(that is deferred future work: a trained projection from Granite Speech encoder
embeddings into Granite token space).

The ASR backend (:mod:`asr`) deliberately has no vLLM dependency so it can be
unit-tested on CPU and reused by either the multimodal-processor integration or a
fallback entrypoint wrapper.
"""

from .asr import ASRTranscriber, DEFAULT_ASR_MODEL_ID, transcribe

__all__ = [
    "ASRTranscriber",
    "DEFAULT_ASR_MODEL_ID",
    "transcribe",
]
