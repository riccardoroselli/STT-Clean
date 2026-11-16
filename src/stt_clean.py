#!/usr/bin/env python
"""stt_clean.py - STT-Clean: local dictation for macOS, as a menu-bar app.

Click the STT-Clean icon in the menu bar, speak, click it again (Esc cancels).
While you speak, a live draft is typed where your cursor is, in any app.
On stop the audio is transcribed again, cleaned up by a local LLM (self-corrections like "no cancella",
spoken punctuation, glossary spellings, "scrivilo in inglese"), and the draft is replaced with the final text.
Right-click it for About, the last status line, the Speech and LLM lists (what is installed, one
click to switch), the switches (smart clean-up, live text), Edit Glossary and Quit.
Everything runs on this Mac: speech with MLX (stt.py), the LLM in Ollama or LM Studio (cleanup.py).

    open build/STT-Clean.app          # normal use (./install.sh builds it)
    .venv/bin/python src/stt_clean.py           # same, from a terminal (log in the terminal)
    .venv/bin/python src/stt_clean.py --no-paste  # only print the text (live draft in the terminal)

Modules: menubar (icon, menu, status line), audio (mic), live (live-draft text), output (typing,
clipboard, paste), esc_key (Esc while recording), stt (speech engines), cleanup (LLM + glossary).
This file: the options and the worker loop.
"""
import argparse
import atexit
import os
import shutil
import signal
import sys
import threading
import time

import numpy as np
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSImage, NSTimer
from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from PyObjCTools import AppHelper

import stt
from audio import SOUND_CANCEL, SOUND_START, SOUND_STOP, play, resample
import cleanup
from cleanup import DEFAULT_LLM, Cleaner, Glossary
from live import Tail, advance
from menubar import GLOSSARY, ROOT, MenuBar, status
from output import can_post, copy, paste, retype

MIN_SECONDS = 0.3         # shorter clicks are ignored


def save_debug(audio: np.ndarray, drafts: list) -> None:
    """Last dictation only (overwritten): exact float32 audio + every live draft and the text typed
    after it, to replay offline."""
    d = os.path.join(ROOT, "logs", "debug")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "last.npy"), audio)
    with open(os.path.join(d, "last_drafts.txt"), "w", encoding="utf-8") as f:
        f.writelines(f"{n}\t{draft}\t{typed}\n" for n, draft, typed in drafts)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="STT-Clean: local menu-bar dictation. Speech model and LLM "
                                 "are picked in its menu (right-click the icon).")
    ap.add_argument("--no-paste", action="store_true", help="only print the text, don't type or paste it")
    ap.add_argument("--debug", action="store_true",
                    help="overwrite logs/debug/last.npy (audio) and last_drafts.txt after each dictation")
    args = ap.parse_args()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)   # menu bar only, no Dock icon
    icon = NSImage.alloc().initWithContentsOfFile_(os.path.join(ROOT, "macos", "AppIcon.icns"))
    if icon:
        app.setApplicationIconImage_(icon)  # About panel and alerts (the process is Python, not the .app)
    ui = MenuBar.alloc().init()
    app.setDelegate_(ui)
    # Ctrl+C, closing the terminal, kill: quit through AppKit so the LLM still gets unloaded.
    # Python runs signal handlers only when the main thread runs Python code, hence the 0.5 s tick.
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.terminate_(None))
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.5, True, lambda _: None)
    threading.Thread(target=run, args=(args, ui), daemon=True).start()
    app.run()


def run(args, ui: MenuBar) -> None:
    """Worker thread: owns all MLX calls, consumes ui.events."""
    try:
        work(args, ui)
    except Exception as e:                 # e.g. the model failed to load: show it in the menu instead of dying silently
        status(f"⚠️  {type(e).__name__}: {e}", keep=True)


def work(args, ui: MenuBar) -> None:
    s = ui.settings
    warnings = []             # shown after Ready, so they stay in the menu's status line
    glossary = Glossary(GLOSSARY)
    provider, model = s["llm_provider"], s["llm_model"]
    if not model:             # nothing picked yet: the recommended model if installed, else the first one found
        found = [(p, n) for _, p, names in cleanup.available() for n in names]
        provider, model = ("ollama", DEFAULT_LLM) if not found or ("ollama", DEFAULT_LLM) in found else found[0]
    cleaner = Cleaner(model, "", glossary, provider=provider)
    ui.current_llm = (provider, model)
    atexit.register(lambda: cleaner.unload())   # frees the LLM's RAM on any exit (the current one)

    def llm_name() -> str:
        return f"{cleaner.model} ({cleaner.label})"

    def llm_error() -> str | None:
        """Check the server and load the model; why it can't be used, or None."""
        if warning := cleaner.check():
            return warning
        status(f"Loading {llm_name()} ...")
        return None if cleaner.warm_up() else f"LLM warm-up failed: {cleaner.last_error}"

    def load_speech(engine: str, model: str):
        status(f"Loading {model} ({engine}) ...")
        new, new_sr = stt.load(engine, model)
        new(np.zeros(new_sr, dtype=np.float32))   # warm-up
        return new, new_sr

    llm = "off"
    if s["llm"]:
        error = llm_error()
        llm = "unavailable, raw text" if error else llm_name()
        if error:
            warnings.append(f"⚠️  {error} Raw transcript until that's fixed.")

    # only speech models already on this Mac: the last pick first, then Parakeet v3, then any other
    found = sorted(stt.available(), key=lambda c: (c != (s["stt"], s["stt_model"]), c[1] != stt.DEFAULT_MODELS["parakeet"]))
    transcribe, sr, engine, stt_model = None, 0, "", ""
    for e, m in found:
        try:
            transcribe, sr = load_speech(e, m)
            engine, stt_model = e, m
            break
        except Exception as ex:
            warnings.append(f"⚠️  {m} didn't load: {type(ex).__name__}: {ex}")
    if not transcribe:        # the app stays up: download one, then pick it in the Speech menu
        warnings.append("⚠️  No speech model found on this Mac: download one (see the README), "
                        "then pick it in the menu → Speech.")

    events = ui.events
    ui.current_stt = (engine, stt_model)
    ui.sr = sr
    ui.ready = transcribe is not None
    AppHelper.callAfter(ui.show)
    if ui.ready:
        status(f"Ready: click the STT-Clean icon to dictate, click again to stop, Esc to cancel. LLM: {llm}.", keep=True)
    # typing the draft and ⌘V need Accessibility; this also opens the system prompt to allow it
    if not args.no_paste and not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
        warnings.append("⚠️  Allow STT-Clean in System Settings → Privacy & Security → Accessibility "
                        "(works as soon as it is on).")
    for warning in warnings:
        status(warning, keep=True)

    recording = False
    capture_sr = sr           # the mic's rate for this recording (resampled to sr)
    target = ()               # focus() at start (app, window, field): the only place we type into
    chunks: list = []
    tail = None               # Tail of this recording: transcribes only what isn't finished yet
    fresh = 0                 # samples recorded since the last live-draft update
    floor = 0.0               # level of this recording's quietest 50 ms (the room's noise)
    heard = False             # something louder than the room since the last live-draft update
    still = False             # the last two live drafts were the same
    typed = ""                # live draft currently typed in the focused app
    prev = ""                 # previous live draft
    drafts: list = []         # (samples, draft, typed) of each live update, for --debug
    can_type = False          # Accessibility allowed: without it macOS silently drops our keystrokes

    def erase_draft() -> None:
        if typed and can_type and can_post(target, wait=2):
            paste("", typed)

    def deliver(text: str) -> None:
        """Replace the live draft with `text` in the app it was typed into, or leave `text` on the clipboard."""
        if args.no_paste:
            return
        if can_type and can_post(target, wait=2):
            if not paste(text if text[-1].isspace() else text + " ", typed):
                status("⚠️  The app didn't paste: the text is on the clipboard, press ⌘V.", keep=True)
            return
        copy(text)                       # don't lose the dictation
        why = ("you moved to another app or window (or a key is held)" if can_type
               else "Accessibility is off for STT-Clean")
        status(f"⚠️  Not typed: {why}. Text copied, paste it with ⌘V.", keep=True)

    def finish(audio: np.ndarray) -> None:
        status("… transcribing")
        t0 = time.perf_counter()
        text = tail.final(audio)
        dt = time.perf_counter() - t0
        if not text:
            status("(no speech detected)", keep=True)
            erase_draft()
            return
        status(f"✓ [{len(audio) / sr:.1f}s audio, {dt:.2f}s] {text}", keep=True, log=False)   # no dictated text on disk
        glossary.maybe_reload()
        text = glossary.apply(text)
        if ui.settings["llm"]:
            status("✨ cleaning up ...")
            t0 = time.perf_counter()
            cleaned = cleaner.clean(text)
            dt = time.perf_counter() - t0
            if cleaned is None:
                status(f"(LLM skipped, raw text pasted: {cleaner.last_error})", keep=True)
            else:
                status(f"✨ [{dt:.2f}s] " + (cleaned.replace("\n", " ⏎ ") if cleaned != text else "(unchanged)"),
                       keep=True, log=False)
                text = cleaned
        deliver(text)

    while True:
        kind, data = events.get()

        if kind == "llm":
            if not data:
                cleaner.unload()          # frees its RAM (~8 GB for a 12B model)
                status("Smart clean-up off: raw transcript (glossary still applied).", keep=True)
            elif error := llm_error():
                status(f"⚠️  {error} Raw transcript until that's fixed.", keep=True)
            else:
                status(f"Smart clean-up on ({llm_name()}).", keep=True)

        elif kind == "llm_model":         # picked in the LLM menu: switch, or stay on the old one if it fails
            provider, model = data
            old = cleaner
            old.unload()                  # don't hold two models in RAM
            cleaner = Cleaner(model, "", glossary, provider=provider)
            if error := llm_error():
                cleaner.unload()          # it may have loaded before failing (e.g. it answers instead of cleaning)
                cleaner = old
                llm_error()
                status(f"⚠️  {model} didn't work, still using {old.model}: {error}", keep=True)
            else:
                ui.current_llm = data
                AppHelper.callAfter(ui.save, {"llm_provider": provider, "llm_model": model})
                status(f"LLM: {llm_name()}.", keep=True)

        elif kind == "stt":               # picked in the Speech menu (the menu set ui.ready = False)
            new_engine, new_model = data
            try:
                new, new_sr = load_speech(new_engine, new_model)
            except Exception as e:
                status(f"⚠️  {new_model} didn't load, still using {stt_model or 'none'}: {type(e).__name__}: {e}", keep=True)
            else:
                transcribe, sr, engine, stt_model = new, new_sr, new_engine, new_model
                ui.sr, ui.current_stt = sr, data
                AppHelper.callAfter(ui.save, {"stt": engine, "stt_model": stt_model})
                status(f"Speech model: {stt_model} ({engine}).", keep=True)
            stt.free_cache()              # the old model's buffers
            ui.ready = transcribe is not None
            AppHelper.callAfter(ui.show)

        elif kind == "start":
            recording, chunks, fresh, typed, prev, drafts = True, [], 0, "", "", []
            tail, floor, heard, still = Tail(transcribe, sr), float("inf"), False, False
            capture_sr, target = data
            can_type = not args.no_paste and AXIsProcessTrusted()   # re-checked: may be allowed mid-run
            play(SOUND_START)
            status("● recording ...")

        elif kind == "audio" and recording:
            chunk = resample(data, capture_sr, sr)
            chunks.append(chunk)
            fresh += len(chunk)
            level = float(np.sqrt(np.mean(chunk * chunk)))
            floor = min(floor, level)
            heard = heard or level > 4 * floor
            # live draft: every ~0.5 s, once the queue is drained, transcribe again what isn't finished
            # yet (transcribe_stream was too unstable for this, see CLAUDE.md). Not during silence:
            # once two drafts are the same, wait for a sound above the room's level (or 3 s).
            if (ui.settings["live"] and fresh >= sr // 2 and events.empty()
                    and (heard or not still or fresh >= 3 * sr)):
                fresh, heard = 0, False
                so_far = np.concatenate(chunks)
                chunks = [so_far]
                draft = tail.draft(so_far)
                cols = shutil.get_terminal_size().columns - 3   # one line, or \r can't redraw it
                status("● " + (draft[-cols:] if draft else "recording ..."), log=False)
                if can_type and can_post(target):   # else skip this update: the next one catches up
                    new = advance(typed, prev, draft)
                    retype(typed, new)
                    typed = new
                still, prev = draft == prev, draft
                drafts.append((len(so_far), draft, typed))

        elif kind == "cancel" and recording:
            recording = False
            play(SOUND_CANCEL)
            erase_draft()
            stt.free_cache()
            status("✕ Cancelled (Esc): nothing pasted.", keep=True)

        elif kind == "stop" and recording:
            recording = False
            play(SOUND_STOP)
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            if args.debug:
                save_debug(audio, drafts)
            try:
                if len(audio) < MIN_SECONDS * sr:
                    status("(too short, ignored)", keep=True)
                    erase_draft()
                else:
                    finish(audio)
            except Exception as e:        # one bad dictation must not kill the loop
                status(f"⚠️  {type(e).__name__}: {e}", keep=True)
            stt.free_cache()              # idle until the next click: don't hold ~1 GB of MLX cache
            AppHelper.callAfter(ui.show)  # back to the idle icon


if __name__ == "__main__":
    if sys.platform != "darwin":
        sys.exit("STT-Clean only runs on macOS.")
    main()
