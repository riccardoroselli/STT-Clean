"""stt.py - speech-to-text engines, all on this Mac (MLX), picked in the menu's Speech list.

available() lists the models a click away (downloaded, engine installed): the menu's Speech list.
load(engine, model) -> (transcribe, sample_rate). transcribe(audio) takes 1-D float32
mono audio at sample_rate, in [-1, 1], and returns its sentences as (text, start, end) in seconds;
text(sentences) joins them. Each engine imports its package only when picked, so only the engine
in use has to be installed.

    parakeet  NVIDIA Parakeet TDT with parakeet-mlx (default)
    whisper   OpenAI Whisper family with mlx-whisper (install.sh adds it without PyTorch)
"""
import glob
import importlib.util
import os
import sys

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # use the models already on this Mac, never download one

DEFAULT_MODELS = {"parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
                  "whisper": "mlx-community/whisper-large-v3-turbo"}
MLX_CACHE = 1 << 30
# the live draft re-transcribes ever-longer clips; MLX would cache every buffer size (22 GB after
# a 60 s dictation with Parakeet) and starve the LLM. A 1 GB cap costs no speed.


PACKAGES = {"parakeet": "parakeet_mlx", "whisper": "mlx_whisper"}
WEIGHTS = {"parakeet": ["model.safetensors"], "whisper": ["weights.safetensors", "weights.npz"]}   # MLX formats


def available() -> list[tuple[str, str]]:
    """(engine, model) for each model in the Hugging Face cache that its installed engine can load."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    found = []
    for folder in sorted(glob.glob(os.path.join(hub, "models--*"))):
        repo = os.path.basename(folder)[len("models--"):].replace("--", "/", 1)
        for engine, package in PACKAGES.items():
            if (engine in repo.lower() and importlib.util.find_spec(package)
                    and any(glob.glob(os.path.join(folder, "snapshots", "*", w)) for w in WEIGHTS[engine])):
                found.append((engine, repo))
    return found


def text(sentences: list) -> str:
    return " ".join(s[0] for s in sentences)


def free_cache() -> None:
    """Between dictations: give MLX's buffer cache (~1 GB after a long clip) back to the system.
    Measured: the next transcription is as fast (5-40 s clips, 27 Oct 2026)."""
    if "mlx.core" in sys.modules:           # loaded by the engine in use
        sys.modules["mlx.core"].clear_cache()


def load(engine: str, model: str = ""):
    engines = {"parakeet": _parakeet, "whisper": _whisper}
    if engine not in engines:
        raise ValueError(f'unknown speech engine "{engine}": use {", ".join(engines)}')
    if "mlx_whisper.transcribe" in sys.modules:   # mlx-whisper keeps its model in a class attribute: free it
        sys.modules["mlx_whisper.transcribe"].ModelHolder.model = None   # (reloaded on its next use if still picked)
    return engines[engine](model or DEFAULT_MODELS[engine])


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
    try:
        import mlx_whisper
        from mlx_whisper.audio import SAMPLE_RATE
    except ImportError:
        raise ImportError("the whisper engine needs mlx-whisper: run ./install.sh again") from None
    mx.set_cache_limit(MLX_CACHE)

    def transcribe(audio: np.ndarray) -> list:   # the model is loaded (and cached) on the first call
        # temperature 0 = no random retries: a 40 s clip took 5.1 s and came out garbled with them, 1.0 s
        # without, and the live drafts stay comparable; no previous-text conditioning: no repeat loops past 30 s
        result = mlx_whisper.transcribe(audio.astype(np.float32), path_or_hf_repo=model, temperature=0.0,
                                        condition_on_previous_text=False)
        return [(s["text"].strip(), s["start"], s["end"]) for s in result["segments"] if s["text"].strip()]
    return transcribe, SAMPLE_RATE

