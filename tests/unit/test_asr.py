# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the audio ASR backend (granite_switch.vllm.audio.asr).

The module under test has no vLLM dependency, but it lives under the
``granite_switch.vllm`` package whose ``__init__`` imports vLLM. To keep this a
fast CPU-tier unit test that runs without the vLLM extra installed, we load the
leaf module directly by file path rather than through the package.
"""

import importlib.util
import pathlib
from unittest import mock

import numpy as np
import pytest

# Load asr.py directly (bypasses granite_switch.vllm.__init__ -> vLLM import).
_ASR_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/granite_switch/vllm/audio/asr.py"
)
_spec = importlib.util.spec_from_file_location("gs_asr_under_test", _ASR_PATH)
asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr)


class TestCoerceAudio:
    def test_array_plus_rate(self):
        a = np.zeros(1600, dtype=np.float32)
        arr, sr = asr._coerce_audio(a, 16000)
        assert sr == 16000 and arr is a

    def test_tuple_form(self):
        a = np.zeros(800, dtype=np.float32)
        arr, sr = asr._coerce_audio((a, 8000), None)
        assert sr == 8000 and arr is a

    def test_list_input_becomes_ndarray(self):
        arr, sr = asr._coerce_audio([0.0] * 10, 16000)
        assert isinstance(arr, np.ndarray) and sr == 16000

    def test_missing_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            asr._coerce_audio(np.zeros(10, dtype=np.float32), None)

    def test_bad_tuple_length_raises(self):
        with pytest.raises(ValueError):
            asr._coerce_audio((np.zeros(10), 1, 2), None)


class TestAsNumpy:
    def test_passthrough_ndarray(self):
        a = np.arange(5)
        assert asr._as_numpy(a) is a

    def test_list(self):
        assert np.array_equal(asr._as_numpy([1, 2, 3]), np.array([1, 2, 3]))

    def test_duck_typed_tensor(self):
        class FakeTensor:
            def __init__(self, x): self._x = x
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self._x

        ft = FakeTensor(np.arange(4))
        assert np.array_equal(asr._as_numpy(ft), np.arange(4))


class TestMonoAndResample:
    def test_downmix_to_mono_float32(self):
        stereo = np.ones((2, 100), dtype=np.float64)
        mono = asr._to_mono_float32(stereo)
        assert mono.shape == (100,) and mono.dtype == np.float32

    def test_resample_noop_at_target(self):
        a = np.zeros(1600, dtype=np.float32)
        assert asr._resample(a, 16000, 16000) is a

    def test_resample_without_librosa_raises_clear_error(self):
        # When librosa is unavailable, a non-target rate must raise a clear error.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("no librosa")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="librosa"):
                asr._resample(np.zeros(800, dtype=np.float32), 8000, 16000)


class TestTranscriber:
    def test_transcribe_strips_and_uses_target_rate(self):
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        fake_pipe = mock.Mock(return_value={"text": "  hello world  "})
        t._pipeline = fake_pipe  # inject so load() is a no-op

        out = t.transcribe(np.zeros(1600, dtype=np.float32), sampling_rate=16000)
        assert out == "hello world"
        passed = fake_pipe.call_args_list[-1][0][0]
        assert passed["sampling_rate"] == 16000

    def test_load_is_idempotent_when_pipeline_set(self):
        # Once the pipeline is loaded, load() must early-return (no rebuild).
        t = asr.ASRTranscriber(model_id="x", device="cpu")
        sentinel = object()
        t._pipeline = sentinel
        t.load()
        assert t._pipeline is sentinel


class TestTranscriberCache:
    def test_same_key_returns_same_instance(self):
        a = asr.get_transcriber("m", "cpu")
        b = asr.get_transcriber("m", "cpu")
        assert a is b

    def test_default_model_id_resolution(self):
        t = asr.get_transcriber(None, "cpu")
        assert t.model_id == asr.DEFAULT_ASR_MODEL_ID

    def test_different_device_distinct_instance(self):
        a = asr.get_transcriber("m", "cpu")
        b = asr.get_transcriber("m", "cuda:0")
        assert a is not b
