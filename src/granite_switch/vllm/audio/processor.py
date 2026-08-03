# SPDX-License-Identifier: Apache-2.0
"""vLLM multimodal processor for the audio cascade.

``_call_hf_processor`` runs ASR and tokenizes the transcript;
``_get_prompt_updates`` then replaces the ``<|audio|>`` marker with the real
transcript token ids via ``PromptReplacement``, so the scheduler sizes KV for the
runtime-determined length rather than a fixed audio window.

Modeled on vLLM 0.19.1's ``ultravox.py``. Audio is answered by the base model —
no adapter control tokens are placed, so the switch is not involved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from .asr import (
    DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
    DEFAULT_ASR_MODEL_ID,
    get_transcriber,
    resolve_generate_kwargs,
)

AUDIO_MARKER = "<|audio|>"
_TARGET_SR = 16_000
# Keeps the transcript budget finite if max_model_len cannot be read.
_FALLBACK_CONTEXT_LEN = 8192
_DUMMY_AUDIO_SECONDS = 5


class GraniteSwitchASRProcessingInfo(BaseProcessingInfo):
    """Static info vLLM needs about the audio modality."""

    def _asr_enabled(self) -> bool:
        return bool(getattr(self.get_hf_config(), "asr_enabled", False))

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # No modalities on a non-audio checkpoint, so vLLM never loads ASR.
        if not self._asr_enabled():
            return {}
        # Finite so vLLM can size KV for the worst case. --limit-mm-per-prompt
        # may lower this ceiling, not raise it.
        return {"audio": self._asr_max_audio_clips()}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        if not self._asr_enabled():
            return {}
        # Sizes the encoder cache and profiling pass only — does NOT bound
        # requests (vLLM's prompt-length check does). A clip cannot exceed the
        # whole context, so the per-clip share of it is the honest upper bound.
        count = mm_counts.get("audio", 1) or 1
        return {"audio": max(1, seq_len // count)}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=_TARGET_SR)

    # --- ASR config resolved from the model's GraniteSwitchConfig ---

    def _asr_model_id(self) -> str:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_model_id", None) or DEFAULT_ASR_MODEL_ID

    def _asr_device(self) -> str:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_device", "cpu") or "cpu"

    def _asr_dtype(self) -> str | None:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_dtype", None)

    def _asr_pipeline_kwargs(self) -> Mapping[str, object]:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_pipeline_kwargs", None) or {}

    def _asr_generate_kwargs(self) -> Mapping[str, object]:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_generate_kwargs", None) or {}

    def _asr_max_audio_clips(self) -> int:
        cfg = self.get_hf_config()
        return int(getattr(cfg, "asr_max_audio_clips", 32) or 32)

    def _asr_self_chunks(self) -> bool:
        cfg = self.get_hf_config()
        return bool(getattr(cfg, "asr_self_chunks", True))

    def _asr_chunk_length_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_length_s", 30.0) or 30.0)

    def _asr_chunk_overlap_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_overlap_s", 5.0) or 0.0)

    def _max_model_len(self) -> int:
        """The served context window, or a safe fallback.

        ``_call_hf_processor`` needs this at request time to size the transcript
        budget; vLLM's profiling ``seq_len`` is not available there.
        """
        model_config = getattr(self.ctx, "model_config", None)
        max_len = getattr(model_config, "max_model_len", None)
        if not max_len:
            max_len = getattr(self.get_hf_config(), "max_position_embeddings", None)
        return int(max_len) if max_len else _FALLBACK_CONTEXT_LEN


class GraniteSwitchASRDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteSwitchASRProcessingInfo]
):
    """Synthetic inputs for vLLM's startup memory-profiling pass."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_MARKER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        length = _DUMMY_AUDIO_SECONDS * _TARGET_SR
        audio = torch.zeros(length, dtype=torch.float32).numpy()
        return {"audio": [audio] * num_audios}


class GraniteSwitchASRMultiModalProcessor(
    BaseMultiModalProcessor[GraniteSwitchASRProcessingInfo]
):
    """Runs ASR and splices the transcript tokens into the prompt."""

    def _transcribe(
        self,
        audio,
        generate_kwargs: Mapping[str, object] | None = None,
    ) -> list[int]:
        """Transcribe one audio item to token ids. Never truncated here — an
        oversized prompt is rejected by vLLM's own length check."""
        transcriber = get_transcriber(
            model_id=self.info._asr_model_id(),
            device=self.info._asr_device(),
            pipeline_kwargs=self.info._asr_pipeline_kwargs(),
            dtype=self.info._asr_dtype(),
        )
        # The data parser already resampled to _TARGET_SR.
        text = transcriber.transcribe(
            audio,
            sampling_rate=_TARGET_SR,
            generate_kwargs=generate_kwargs or None,
            self_chunks=self.info._asr_self_chunks(),
            chunk_length_s=self.info._asr_chunk_length_s(),
            chunk_overlap_s=self.info._asr_chunk_overlap_s(),
        )
        tokenizer = self.info.get_tokenizer()
        return tokenizer.encode(text, add_special_tokens=False)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        audios = mm_data.get("audios", []) or []

        if not audios:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature(dict(input_ids=[input_ids]), tensor_type="pt")

        # Resolved once, then applied to every audio item in this request.
        generate_kwargs = resolve_generate_kwargs(
            self.info._asr_generate_kwargs(),
            mm_kwargs,
            DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
        )

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Concatenated flat, with per-item sizes to split them back.
        per_item_ids = [self._transcribe(a, generate_kwargs) for a in audios]
        sizes = [len(ids) for ids in per_item_ids]
        flat_ids = [tid for ids in per_item_ids for tid in ids]

        return BatchFeature(
            dict(
                input_ids=[input_ids],
                audio_token_ids=torch.tensor(flat_ids, dtype=torch.long),
                audio_num_tokens=torch.tensor(sizes, dtype=torch.long),
            ),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_tokens = hf_inputs.get("audio_num_tokens", torch.zeros(0))
        return dict(
            audio_token_ids=MultiModalFieldConfig.flat_from_sizes("audio", num_tokens),
            audio_num_tokens=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out = out_mm_kwargs.get_data()
        num_tokens = out.get("audio_num_tokens", torch.zeros(0))
        starts = torch.cumsum(num_tokens, dim=0, dtype=torch.long)
        starts = torch.cat([torch.tensor([0], dtype=torch.long), starts])
        all_ids = out.get("audio_token_ids", torch.zeros(0, dtype=torch.long))

        def replacement(item_idx: int):
            s = int(starts[item_idx])
            e = int(starts[item_idx + 1])
            return [int(t) for t in all_ids[s:e]]

        return [
            PromptReplacement(
                modality="audio",
                target=AUDIO_MARKER,
                replacement=replacement,
            )
        ]
