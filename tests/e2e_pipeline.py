"""End-to-end check of the real worker loop without mic, keyboard or menu bar:
macOS `say` -> the app's worker (live draft, speech model, glossary, LLM) -> the text it would paste.

    HF_HUB_OFFLINE=1 python tests/e2e_pipeline.py    # speech model and LLM picked in the app's menu

Exits with status 1 if a check fails. Needs a speech model and a running LLM server.
"""
import atexit
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import menubar  # noqa: E402
import stt_clean  # noqa: E402

SPEED = 4                 # the audio is fed this many times faster than real time
said: list = []           # every status line of the worker


def status(msg, keep=False, log=True, detail=""):
    said.append(msg + detail)
    if not msg.startswith("●"):
        print(f"    {msg}{detail}", flush=True)


class FakeMenuBar:
    """What the worker uses of menubar.MenuBar."""
    def __init__(self):
        self.events, self.settings = queue.Queue(), menubar.load_settings()
        self.current_llm = self.current_stt = ("", "")
        self.sr, self.ready = 0, False

    def show(self, busy=False):
        pass

    def save(self, changes):
        self.settings.update(changes)


def speak(text: str, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        aiff, wav = os.path.join(d, "t.aiff"), os.path.join(d, "t.wav")
        subprocess.run(["say", "-v", "Alice", text, "-o", aiff], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{sr}", "-c", "1", aiff, wav], check=True)
        with wave.open(wav) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def wait_for(start: str, since: int, timeout: float = 120) -> str:
    """The first status line after `since` that starts with one of `start` (separated by |)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for msg in said[since:]:
            if msg.startswith(tuple(start.split("|"))):
                return msg
        time.sleep(0.05)
    sys.exit(f"✗ no status starting with {start!r} within {timeout:.0f} s")


def dictate(ui, text: str) -> tuple[str, str, int]:
    """Say `text` to the worker: (raw transcript, what the LLM made of it, live drafts shown)."""
    audio, since = speak(text, ui.sr), len(said)
    block = ui.sr // 20
    ui.events.put(("start", (ui.sr, ())))
    t0 = time.monotonic()
    for i in range(0, len(audio), block):
        ui.events.put(("audio", audio[i:i + block]))
        time.sleep(max(0, t0 + (i + block) / ui.sr / SPEED - time.monotonic()))
    ui.events.put(("stop", None))
    raw = wait_for("✓", since).split(": ", 1)[1]
    cleaned = wait_for("✨ 0|✨ 1|✨ 2|✨ 3|✨ 4|✨ 5|✨ 6|✨ 7|✨ 8|✨ 9|(LLM skipped", since)
    cleaned = cleaned.split(": ", 1)[-1].replace(" ⏎ ", "\n")
    return raw, cleaned, sum(m.startswith("● Recording ... ") and len(m) > 30 for m in said[since:])


def main() -> None:
    stt_clean.status = status
    ui = FakeMenuBar()
    ui.settings.update(llm=True, live=True, style="clean")
    threading.Thread(target=stt_clean.run, args=(types.SimpleNamespace(no_paste=True, debug=False), ui),
                     daemon=True).start()
    wait_for("Ready", 0)
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'✓' if ok else '✗'} {name}\n")

    raw, out, drafts = dictate(ui, "Ci vediamo martedì alle cinque, no cancella, mercoledì alle sei.")
    check("self-correction", "mercoledì" in out and "martedì" not in out)   # (the LLM may still be loading here)
    raw, out, _ = dictate(ui, "Domani mando la relazione al professore. Scrivilo in inglese.")
    check("spoken instruction: translated", "professor" in out and "relazione" not in out)
    ui.settings["style"] = "bullets"
    raw, out, drafts = dictate(ui, "Per la cena servono tre cose. Il pane, il vino rosso e un dolce al cioccolato.")
    check("style from the menu: bullet list, with live drafts", out.count("\n- ") >= 2 and "vino" in out and drafts >= 2)
    ui.settings["style"] = "clean"

    class Flaky(stt_clean.Tail):                      # its first live update fails (as MLX can when memory is short)
        def draft(self, audio):
            if not hasattr(self, "failed"):
                self.failed = True
                raise RuntimeError("test: this live update fails")
            return super().draft(audio)
    stt_clean.Tail, since = Flaky, len(said)
    raw, out, _ = dictate(ui, "Per la cena servono tre cose. Il pane, il vino rosso e un dolce al cioccolato.")
    check("a failed live update doesn't end the dictation",
          "vino" in out and any(m.startswith("⚠️  Live text") for m in said[since:]))
    stt_clean.Tail = Flaky.__base__

    stt_clean.IDLE_UNLOAD, since = 2, len(said)       # idle: the LLM is freed, then loads again while speaking
    ui.events.put(("llm", True))                      # (any event, so the loop picks up the short idle time)
    wait_for("Idle", since)
    raw, out, _ = dictate(ui, "Prenota il tavolo per otto persone, anzi per dieci, alle nove di sera.")
    check("LLM freed when idle, loaded again while speaking", ("dieci" in out or "10" in out) and "otto" not in out and "8" not in out)

    since = len(said)
    ui.events.put(("llm", False))
    wait_for("Smart clean-up off", since)
    atexit._run_exitfuncs()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
