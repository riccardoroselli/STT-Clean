"""stt.py - speech-to-text engines, all on this Mac (MLX), picked in the menu's Speech list.

available() lists the models a click away (downloaded, engine installed): the menu's Speech list.
load(engine, model) -> (transcribe, sample_rate). transcribe(audio) takes 1-D float32
mono audio at sample_rate, in [-1, 1], and returns its sentences as (text, start, end) in seconds.
Each engine imports its package only when picked, so only the engine in use has to be installed.

    parakeet  NVIDIA Parakeet TDT with parakeet-mlx (default)
    whisper   OpenAI Whisper family with mlx-whisper (install.sh adds it without PyTorch)
"""
import importlib.util
import os
import sys

import numpy as np

import hfcache

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # use the models already on this Mac, never download one

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"   # loaded at launch until you pick another one
MLX_CACHE = 1 << 30
# the live draft re-transcribes ever-longer clips; MLX would cache every buffer size (22 GB after
# a 60 s dictation with Parakeet) and starve the LLM. A 1 GB cap costs no speed.


PACKAGES = {"parakeet": "parakeet_mlx", "whisper": "mlx_whisper"}
WEIGHTS = {"parakeet": ["model.safetensors"], "whisper": ["weights.safetensors", "weights.npz"]}   # MLX formats


def engine_of(config: dict) -> str:
    """The engine that loads a model with this config.json ("" = not a speech model we can run).
    By content, not by repo name: fine-tunes and renamed copies are found too."""
    if str(config.get("target", "")).startswith("nemo.collections.asr.models"):
        return "parakeet"     # NVIDIA NeMo transducer / CTC models (Parakeet TDT, RNNT, CTC)
    if "n_audio_state" in config and "n_mels" in config:
        return "whisper"      # mlx-whisper's own format (not Transformers' openai/whisper-*)
    return ""


def available() -> list[tuple[str, str]]:
    """(engine, model) for each model in the Hugging Face cache that its installed engine can load."""
    return [(engine, repo) for repo, folder, config in hfcache.configs()
            if (engine := engine_of(config)) and importlib.util.find_spec(PACKAGES[engine])
            and any(os.path.exists(os.path.join(folder, w)) for w in WEIGHTS[engine])]


def free_cache() -> None:
    """Between dictations: give MLX's buffer cache (~1 GB after a long clip) back to the system.
    Measured: the next transcription is as fast (5-40 s clips, 27 Oct 2026)."""
    if "mlx.core" in sys.modules:           # loaded by the engine in use
        sys.modules["mlx.core"].clear_cache()


def load(engine: str, model: str):
    """`engine` and `model` as available() lists them."""
    if "mlx_whisper.transcribe" in sys.modules:   # mlx-whisper keeps its model in a class attribute: free it
        sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None   # (reloaded on its next use if still picked)
    return {"parakeet": _parakeet, "whisper": _whisper}[engine](model)


def _parakeet(model: str):
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel
    m = from_pretrained(model)
    mx.set_cache_limit(MLX_CACHE)

    def transcribe(audio: np.ndarray) -> list:
        # get_logmel needs float32 (it reinterprets the complex STFT as 4-byte floats)
        result = m.generate(get_logmel(mx.array(audio.astype(np.float32)), m.preprocessor_config))[0]
        return [(s.text.strip(), s.start, s.end) for s in result.sentences if s.text.strip()]
    return transcribe, m.preprocessor_config.sample_rate


def _whisper(model: str):
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.audio import SAMPLE_RATE
    mx.set_cache_limit(MLX_CACHE)

    def transcribe(audio: np.ndarray) -> list:   # the model is loaded (and cached) on the first call
        # temperature 0 = no random retries: a 40 s clip took 5.1 s and came out garbled with them, 1.0 s
        # without, and the live drafts stay comparable; no previous-text conditioning: no repeat loops past 30 s
        result = mlx_whisper.transcribe(audio.astype(np.float32), path_or_hf_repo=model, temperature=0.0,
                                        condition_on_previous_text=False)
        return [(s["text"].strip(), s["start"], s["end"]) for s in result["segments"] if s["text"].strip()]
    return transcribe, SAMPLE_RATE

