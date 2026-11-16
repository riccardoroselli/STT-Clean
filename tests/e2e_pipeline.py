"""End-to-end check without mic/keyboard: macOS `say` -> speech engine -> glossary -> LLM.

    HF_HUB_OFFLINE=1 python tests/e2e_pipeline.py    # speech model and LLM picked in the app's menu
"""
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import stt  # noqa: E402
from cleanup import DEFAULT_LLM, Cleaner, Glossary  # noqa: E402
from menubar import GLOSSARY, load_settings  # noqa: E402

PHRASES = [
    "Ci vediamo martedì alle cinque, no cancella, mercoledì alle sei.",
    "Domani mando la relazione al professore. Scrivilo in inglese.",
]


def speak(text: str, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "t.aiff"), os.path.join(d, "t.wav")
        subprocess.run(["say", "-v", "Alice", text, "-o", aiff], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sr}", "-c", "1", aiff, wav], check=True)
        with wave.open(wav) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main() -> None:
    s = load_settings()
    transcribe, sr = stt.load(s["stt"], s["stt_model"])
    glossary = Glossary(GLOSSARY)
    cleaner = Cleaner(s["llm_model"] or DEFAULT_LLM, "", glossary, provider=s["llm_provider"])
    if warning := cleaner.check():
        sys.exit(warning)
    cleaner.warm_up()
    try:
        for phrase in PHRASES:
            audio = speak(phrase, sr)
            t0 = time.perf_counter()
            raw = stt.text(transcribe(audio))
            t1 = time.perf_counter()
            out = cleaner.clean(glossary.apply(raw))
            t2 = time.perf_counter()
            print(f"spoken : {phrase}\n{s['stt']:8} [{t1 - t0:.2f}s]: {raw}\nllm      [{t2 - t1:.2f}s]: {out!r}\n")
    finally:
        cleaner.unload()


if __name__ == "__main__":
    main()
