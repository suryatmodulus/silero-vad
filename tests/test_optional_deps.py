"""Tests for the optional audio/ONNX dependencies.

torchaudio and torchcodec are both optional extras: the VAD model itself needs
neither, only the read_audio / save_audio helpers do. These tests pin down the
behaviour when one or both of them are missing.
"""
import subprocess
import sys

import pytest
import torch

from silero_vad import read_audio, save_audio
import silero_vad.utils_vad as utils_vad

WAV = "tests/data/test.wav"


def _hide_torchaudio(monkeypatch):
    """Make `import torchaudio` fail, as it would if it were not installed."""
    monkeypatch.setitem(sys.modules, "torchaudio", None)


def _hide_torchcodec(monkeypatch):
    for name in ("torchcodec", "torchcodec.decoders", "torchcodec.encoders"):
        monkeypatch.setitem(sys.modules, name, None)


def _has_torchcodec():
    try:
        import torchcodec.decoders  # noqa: F401
    except ImportError:
        return False
    return True


def test_import_without_onnxruntime_or_numpy():
    """`import silero_vad` must not pull in the sequence path.

    sequence_vad needs numpy and onnxruntime, which are optional extras, so it
    is imported lazily. Run in a subprocess so that other tests, which do use
    the sequence path, cannot mask the regression.
    """
    code = ("import silero_vad, sys; "
            "assert 'silero_vad.sequence_vad' not in sys.modules, 'sequence_vad imported eagerly'; "
            "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported eagerly'")
    subprocess.check_call([sys.executable, "-c", code])


def test_lazy_sequence_attributes_still_resolve():
    import silero_vad

    assert "get_speech_timestamps_sequence" in dir(silero_vad)
    from silero_vad import SileroVADSequence, get_speech_timestamps_sequence

    assert SileroVADSequence is silero_vad.SileroVADSequence
    assert callable(get_speech_timestamps_sequence)

    with pytest.raises(AttributeError):
        silero_vad.definitely_not_an_attribute


def test_no_audio_backend_raises_actionable_error(monkeypatch):
    _hide_torchaudio(monkeypatch)
    _hide_torchcodec(monkeypatch)

    with pytest.raises(ImportError) as excinfo:
        read_audio(WAV, sampling_rate=16000)
    message = str(excinfo.value)
    assert "silero-vad[audio]" in message
    assert "silero-vad[codec]" in message

    with pytest.raises(ImportError):
        save_audio("unused.wav", torch.zeros(16000), sampling_rate=16000)


@pytest.mark.skipif(not _has_torchcodec(), reason="torchcodec is not installed")
def test_torchcodec_backend_without_torchaudio(monkeypatch, tmp_path):
    """torchcodec alone must be enough for both reading and writing."""
    expected = read_audio(WAV, sampling_rate=16000)

    _hide_torchaudio(monkeypatch)
    audio = read_audio(WAV, sampling_rate=16000)

    assert audio.dim() == 1
    assert audio.dtype == torch.float32
    # test.wav is already mono 16 kHz, so no resampling is involved and the two
    # decoders must agree exactly.
    assert torch.equal(audio, expected)

    out = tmp_path / "out.wav"
    save_audio(str(out), audio[:16000], sampling_rate=16000)
    assert out.stat().st_size > 0
    assert read_audio(str(out), sampling_rate=16000).numel() == 16000
