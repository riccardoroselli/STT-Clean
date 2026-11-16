"""menubar.py - the STT-Clean icon in the menu bar, its menu and switches, and the status line."""
import atexit
import math
import os
import queue
import subprocess
import sys
import time

import objc
from AppKit import (NSApplication, NSAttributedString, NSBezierPath, NSColor, NSCompositingOperationClear,
                    NSCompositingOperationSourceOver, NSEventMaskLeftMouseUp, NSEventMaskRightMouseUp,
                    NSEventTypeRightMouseUp, NSEvenOddWindingRule, NSFont, NSFontAttributeName,
                    NSForegroundColorAttributeName, NSGraphicsContext, NSImage, NSImageLeft, NSMenu, NSMenuItem,
                    NSObject, NSStatusBar, NSTimer, NSUserDefaults, NSVariableStatusItemLength)
from PyObjCTools import AppHelper

import cleanup
import stt
from audio import open_mic
from hotkey import HotKey
from output import focus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = None                 # the MenuBar, once created


def status(msg: str, keep: bool = False, log: bool = True, detail: str = "") -> None:
    """Show `msg` in the menu's status line and print it. In a terminal the line is redrawn in place
    unless `keep`, followed by `detail`: the dictated text, which only a terminal run shows (never
    the menu, never the log file). log=False keeps a repeated message out of the log."""
    if sys.stdout.isatty():
        print(f"\r\033[K{msg}{detail}", end="\n" if keep else "", flush=True)
    elif log:
        print(msg, flush=True)
    if UI is not None:
        AppHelper.callAfter(UI.note, msg)


def problem(key: str, title: str = "") -> None:
    """Show (or with no title, remove) a menu item that fixes an open problem when clicked.
    Keys: "llm" (start its server), "access" and "mic" (open the privacy setting), "speech" (how to get a model)."""
    if UI is not None:
        AppHelper.callAfter(UI.problem, key, title)


# ---------------------------------------------------------------- settings: the menu's switches and picks
# Remembered by macOS in ~/Library/Preferences/<SUITE>.plist (same for the app and a terminal run;
# start over with:  defaults delete com.riccardoroselli.stt-clean).
SUITE = "com.riccardoroselli.stt-clean"
PREFS = NSUserDefaults.alloc().initWithSuiteName_(SUITE)
VERSION = "1.1"
DEFAULTS = {"llm": True, "live": True,                    # switches: smart clean-up, live text while speaking
            "style": "clean",                             # what the LLM does with the text (cleanup.STYLES)
            "shortcut": "",                               # start/stop hot key ("" = off)
            "stt": "parakeet", "stt_model": "",           # speech engine and model ("" = its default)
            "llm_provider": "ollama", "llm_model": ""}    # LLM server and model ("" = found at launch)


def load_settings() -> dict:
    return {k: v if (x := PREFS.objectForKey_(k)) is None else type(v)(x) for k, v in DEFAULTS.items()}


SHORTCUTS = ["", "⌥Space", "⌃⌥Space", "⌃⌥D", "F5"]          # start/stop hot keys offered in the menu
PRIVACY = "x-apple.systempreferences:com.apple.preference.security?Privacy_"
FIXES = {"access": PRIVACY + "Accessibility", "mic": PRIVACY + "Microphone",
         "speech": "https://github.com/riccardoroselli/STT-Clean#models"}   # "llm" is fixed by the worker
GLOSSARY = os.path.join(ROOT, "config", "glossary.txt")
GLOSSARY_TEMPLATE = """\
# STT-Clean glossary: how to spell names, brands and jargon the speech model gets wrong.
# One term per line:   Correct spelling: heard as 1, heard as 2
#   - the "heard as" variants (optional) are replaced exactly, before the LLM runs
#   - every term is also given to the LLM, which fixes close mis-hearings on its own
# Lines starting with # are ignored. Changes apply from the next dictation. Examples:
# Ollama: o lama, olama
# GitHub: git hub
"""


# ---------------------------------------------------------------- the logo as a menu-bar glyph
BRAND_BLUE = NSColor.colorWithSRGBRed_green_blue_alpha_(0x38 / 255, 0xB6 / 255, 0xFF / 255, 1)   # assets/LOGO.png


def _star(path, cx, cy, rx, ry):
    """Four-point sparkle (the logo's star): concave sides, tips at (cx±rx, cy) and (cx, cy±ry)."""
    k = 0.18
    path.moveToPoint_((cx, cy + ry))
    for x, y, sx, sy in ((cx + rx, cy, 1, 1), (cx, cy - ry, 1, -1), (cx - rx, cy, -1, -1), (cx, cy + ry, -1, 1)):
        c = (cx + sx * rx * k, cy + sy * ry * k)
        path.curveToPoint_controlPoint1_controlPoint2_((x, y), c, c)


def brand_glyph(center: str, color, levels=None):
    """30x20 pt menu-bar version of the logo in `color`: the swoosh ring with its two sparkles, and
    `center` inside ("wave" = voice, "sparkle" = cleaning). `levels`: the mic level (0-1) for each of
    the wave's 5 bars, oldest first, so they move with the voice while recording."""
    def draw(rect):
        color.set()
        ring = NSBezierPath.bezierPathWithOvalInRect_(((1, 4), (28, 12)))
        ring.appendBezierPath_(NSBezierPath.bezierPathWithOvalInRect_(((3.6, 5.6), (22.8, 8.8))))
        ring.setWindingRule_(NSEvenOddWindingRule)
        ring.fill()
        stars = NSBezierPath.bezierPath()
        _star(stars, 17.5, 15.6, 1.7, 3.9)        # top, right of center; bottom, left of center
        _star(stars, 12.5, 4.4, 1.7, 3.9)
        if center == "sparkle":
            _star(stars, 15, 10, 2.6, 3.4)
            _star(stars, 20.4, 11.6, 0.9, 1.3)
        stars.fill()
        # the logo's two cuts in the ring (upper right, lower left) that make it a swoosh
        NSGraphicsContext.currentContext().setCompositingOperation_(NSCompositingOperationClear)
        for a in (math.radians(35), math.radians(215)):
            cut = NSBezierPath.bezierPath()
            cut.moveToPoint_((15 + 11 * math.cos(a), 10 + 4 * math.sin(a)))
            cut.lineToPoint_((15 + 15 * math.cos(a), 10 + 7 * math.sin(a)))
            cut.setLineWidth_(0.9)
            cut.stroke()
        NSGraphicsContext.currentContext().setCompositingOperation_(NSCompositingOperationSourceOver)
        if center == "wave":
            for i, h in enumerate((2.2, 4.6, 6.6, 4.6, 2.2)):
                if levels:
                    h = max(1.4, h * (0.35 + 0.9 * levels[i]))   # silence: a row of dots
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((15 + (i - 2) * 2.6 - 0.7, 10 - h / 2), (1.4, h)), 0.7, 0.7).fill()
        return True
    img = NSImage.imageWithSize_flipped_drawingHandler_((30, 20), False, draw)
    img.setAccessibilityDescription_("STT-Clean")
    return img


# ---------------------------------------------------------------- the menu-bar item
class MenuBar(NSObject):
    """Logo icon in the menu bar: click (or the shortcut) = start / stop dictation; right-click = the menu.
    Lives on the main thread (AppKit); the worker thread talks to it via AppHelper.callAfter."""

    def init(self):
        global UI
        self = objc.super(MenuBar, self).init()
        self.events = queue.Queue()        # consumed by the worker thread
        self.recording = False
        self.ready = False                 # model loaded
        self.sr = 0                        # model sample rate, set by the worker when ready
        self.stream = None                 # mic, open only while recording
        self.settings = load_settings()    # read by the worker
        self.current_stt = ("", "")        # (engine, model) in use, set by the worker once loaded
        self.current_llm = ("", "")        # (provider, model) in use, set by the worker
        self.timer = None                  # while recording: moves the icon's wave and the clock
        self.level = 0.0                   # loudest mic level (0-1) since the last tick, set by the audio thread
        self.levels = [0.0] * 5            # the last 5 ticks: the wave's bars
        self.esc = HotKey("Esc", lambda: self.end("cancel") if self.recording else None)
        self.hotkey = None                 # the start/stop shortcut, if one is picked
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        button = self.item.button()
        button.setTarget_(self)
        button.setAction_("clicked:")
        button.sendActionOn_(NSEventMaskLeftMouseUp | NSEventMaskRightMouseUp)
        button.setImagePosition_(NSImageLeft)
        self.font = NSFont.monospacedDigitSystemFontOfSize_weight_(NSFont.menuBarFontOfSize_(0).pointSize(), 0)
        self.menu = NSMenu.alloc().init()
        about = self.menu.addItemWithTitle_action_keyEquivalent_("About STT-Clean", "about:", "")
        about.setTarget_(self)
        about.setImage_(brand_glyph("wave", BRAND_BLUE))
        self.menu.addItem_(NSMenuItem.separatorItem())
        self.info = self.menu.addItemWithTitle_action_keyEquivalent_("Starting ...", None, "")
        self.problems = {}                 # hidden until problem(key, title): click = fix it
        for key in ("speech", "llm", "access", "mic"):
            fix = self.menu.addItemWithTitle_action_keyEquivalent_("", "fix:", "")
            fix.setTarget_(self)
            fix.setRepresentedObject_(key)
            fix.setHidden_(True)
            self.problems[key] = fix
        self.menu.addItem_(NSMenuItem.separatorItem())
        self.speech_menu = self.menu.addItemWithTitle_action_keyEquivalent_("Speech Model", None, "")
        self.llm_menu = self.menu.addItemWithTitle_action_keyEquivalent_("LLM", None, "")
        self.style_menu = self.menu.addItemWithTitle_action_keyEquivalent_("Style", None, "")
        self.menu.addItem_(NSMenuItem.separatorItem())
        self.switches = {}
        for key, title in (("llm", "Smart clean-up (LLM)"), ("live", "Live text while speaking")):
            switch = self.menu.addItemWithTitle_action_keyEquivalent_(title, "toggle:", "")
            switch.setTarget_(self)
            switch.setRepresentedObject_(key)
            switch.setState_(self.settings[key])
            self.switches[key] = switch
        self.shortcut_menu = self.menu.addItemWithTitle_action_keyEquivalent_("Shortcut", None, "")
        self.menu.addItemWithTitle_action_keyEquivalent_("Edit Glossary…", "editGlossary:", "").setTarget_(self)
        self.menu.addItem_(NSMenuItem.separatorItem())
        self.menu.addItemWithTitle_action_keyEquivalent_("Quit STT-Clean", "terminate:", "q")
        self.show()
        UI = self
        self.set_shortcut(self.settings["shortcut"])
        return self

    def clicked_(self, sender):
        if NSApplication.sharedApplication().currentEvent().type() == NSEventTypeRightMouseUp:
            self.fill_models()
            self.item.setMenu_(self.menu)          # pop the menu up once, keep left-click for dictation
            self.item.button().performClick_(None)
            self.item.setMenu_(None)
        else:
            self.toggle()

    @objc.python_method
    def toggle(self):
        """Left click or the shortcut: start dictating, or stop."""
        if self.recording:
            self.end("stop")
        elif self.ready:
            self.start()

    @objc.python_method
    def heard(self, samples):
        """Audio thread, every 50 ms of mic audio: to the worker, and its level to the icon."""
        self.events.put(("audio", samples))
        db = 10 * math.log10(float(samples @ samples) / len(samples) + 1e-12)
        self.level = max(self.level, min(1.0, max(0.0, (db + 55) / 35)))   # -55 dB (a quiet room) .. -20 dB (a voice)

    @objc.python_method
    def start(self):
        try:                               # here, not in the worker: it may still be busy with the last dictation
            self.stream, rate = open_mic(self.sr, self.heard)
        except Exception as e:             # no mic, or it refused to open
            status(f"⚠️  Microphone: {type(e).__name__}: {e}", keep=True)
            return
        self.events.put(("start", (rate, focus())))   # where the text will be typed
        self.recording = True
        self.levels = [0.0] * 5
        t0 = time.monotonic()
        self.timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.1, True, lambda _: self.tick(time.monotonic() - t0))
        self.tick(0)
        if not self.esc.on():
            status("⚠️  Esc is taken by another app: cancel with Esc won't work.", keep=True)

    @objc.python_method
    def tick(self, seconds):
        """While recording, 10 times a second: the wave follows the voice, the clock the time."""
        self.levels = self.levels[1:] + [self.level]
        self.level = 0.0
        clock = f" {int(seconds) // 60}:{int(seconds) % 60:02d}"
        if self.item.button().title() != clock:
            self.item.button().setAttributedTitle_(NSAttributedString.alloc().initWithString_attributes_(
                clock, {NSForegroundColorAttributeName: NSColor.whiteColor(), NSFontAttributeName: self.font}))
        self.show()

    @objc.python_method
    def end(self, kind):
        """Stop recording: "stop" transcribes and pastes, "cancel" (Esc) throws the dictation away."""
        self.recording = False
        self.stream.stop()                 # returns after the last callback: all audio is queued before `kind`
        self.stream.close()
        self.events.put((kind, None))
        self.esc.off()
        self.timer.invalidate()
        self.item.button().setTitle_("")
        self.show(busy=kind == "stop")

    def toggle_(self, sender):
        key = sender.representedObject()
        self.save({key: not self.settings[key]})
        if key == "llm":
            self.events.put(("llm", self.settings[key]))   # the worker loads / unloads it

    @objc.python_method
    def save(self, changes):
        """Apply `changes` to the settings and remember them."""
        self.settings.update(changes)
        for key, value in changes.items():
            PREFS.setObject_forKey_(value, key)
        for key, switch in self.switches.items():
            switch.setState_(self.settings[key])

    # ---------------------------------------------------------------- model pickers
    @objc.python_method
    def fill_models(self):
        """Rebuild the Speech Model and LLM submenus with what works right now (runs as the menu opens:
        the Hugging Face cache and the local servers answer in ~40 ms). The models in use are ticked."""
        engine, model = self.current_stt
        speech = stt.available()
        if model and (engine, model) not in speech:
            speech.insert(0, (engine, model))       # in use, but no longer found in the cache
        groups = {}                                 # one section per engine
        for e, m in speech:
            groups.setdefault(e.capitalize(), []).append((m.split("/")[-1], f"{e}\t{m}", (e, m) == self.current_stt))
        self.speech_menu.setTitle_(f"Speech: {model.split('/')[-1]}" if model else "Speech Model")
        self.speech_menu.setSubmenu_(self.picker(list(groups.items()), "pickSpeech:", "No speech model found (see the README)"))

        provider, model = self.current_llm
        servers = cleanup.available()
        if model and not any(p == provider and model in names for _, p, names in servers):
            servers.insert(0, ("In use", provider, [model]))   # its server stopped or doesn't list it
        def short(name):                            # llama.cpp names a model by its file's whole path
            return os.path.basename(name) if name.startswith("/") else name.removesuffix(":latest")
        sections = [(label, [(short(n), f"{p}\t{n}", (p, n) == self.current_llm) for n in names])
                    for label, p, names in servers]
        self.llm_menu.setTitle_(f"LLM: {short(model)}" if model else "LLM")
        self.llm_menu.setSubmenu_(self.picker(sections, "pickLLM:", "No LLM found (see the README)"))

        style = self.settings["style"]
        self.style_menu.setTitle_(f"Style: {cleanup.STYLES[style][0]}")
        self.style_menu.setSubmenu_(self.picker(
            [("", [(title, key, key == style) for key, (title, _) in cleanup.STYLES.items()])], "pickStyle:", ""))
        shortcut = self.settings["shortcut"]
        self.shortcut_menu.setTitle_(f"Shortcut: {shortcut or 'off'}")
        self.shortcut_menu.setSubmenu_(self.picker(
            [("", [(combo or "Off", combo, combo == shortcut) for combo in SHORTCUTS])], "pickShortcut:", ""))

    @objc.python_method
    def picker(self, sections, action, empty):
        """A submenu: a header per section, then (title, choice, ticked) items that call `action`."""
        menu = NSMenu.alloc().init()
        for header, items in sections:
            if header:
                menu.addItem_(NSMenuItem.sectionHeaderWithTitle_(header))
            for title, choice, ticked in items:
                item = menu.addItemWithTitle_action_keyEquivalent_(title, action, "")
                item.setTarget_(self)
                item.setRepresentedObject_(choice)
                item.setState_(ticked)
        if not sections:
            menu.addItemWithTitle_action_keyEquivalent_(empty, None, "")
        return menu

    def pickSpeech_(self, sender):
        engine, model = sender.representedObject().split("\t")
        if (engine, model) == self.current_stt:
            return
        if self.recording:                 # (while loading is fine: the worker takes picks in order)
            status("⚠️  Switch the speech model after this dictation.", keep=True)
            return
        self.ready = False                 # no new dictation until it's loaded
        self.current_stt = (engine, model)  # ticked at once; the worker puts the old one back if this fails
        self.show()
        self.events.put(("stt", (engine, model)))

    def pickLLM_(self, sender):
        choice = tuple(sender.representedObject().split("\t"))
        if self.recording:
            status("⚠️  Switch the LLM after this dictation.", keep=True)
            return
        was_on = self.settings["llm"]
        if not was_on:
            self.save({"llm": True})       # picking an LLM means using it
        if choice != self.current_llm:
            self.current_llm = choice      # ticked at once; the worker puts the old one back if this fails
            self.events.put(("llm_model", choice))
        elif not was_on:
            self.events.put(("llm", True))

    def pickStyle_(self, sender):
        self.save({"style": sender.representedObject()})
        if not self.settings["llm"]:       # a style is the LLM's work: picking one means using it
            self.save({"llm": True})
            self.events.put(("llm", True))

    def pickShortcut_(self, sender):
        self.save({"shortcut": sender.representedObject()})
        self.set_shortcut(self.settings["shortcut"])

    @objc.python_method
    def set_shortcut(self, combo):
        """Make `combo` ("" = none) the global start/stop hot key."""
        if self.hotkey:
            self.hotkey.off()
        self.hotkey = HotKey(combo, self.toggle) if combo in SHORTCUTS[1:] else None
        if self.hotkey and not self.hotkey.on():
            status(f"⚠️  {combo} is taken by another app: pick another shortcut.", keep=True)

    @objc.python_method
    def problem(self, key, title):
        self.problems[key].setTitle_(title)
        self.problems[key].setHidden_(not title)

    def fix_(self, sender):
        key = sender.representedObject()
        if key == "llm":
            self.events.put(("llm_start", None))   # the worker starts the LLM's server and loads the model
        else:
            subprocess.run(["open", FIXES[key]])

    def about_(self, sender):
        app = NSApplication.sharedApplication()
        app.activate()                     # menu-bar-only app: bring the panel to the front
        app.orderFrontStandardAboutPanelWithOptions_({
            "ApplicationName": "STT-Clean", "ApplicationVersion": VERSION, "Version": "",
            "Credits": NSAttributedString.alloc().initWithString_(
                f"Live speech-to-text with LLM refinement, on this Mac.\nSpeech: {self.current_stt[1]} ({self.current_stt[0]})\n"
                f"LLM: {self.current_llm[1]} ({self.current_llm[0]})")})

    def editGlossary_(self, sender):
        if not os.path.exists(GLOSSARY):
            os.makedirs(os.path.dirname(GLOSSARY), exist_ok=True)
            with open(GLOSSARY, "w", encoding="utf-8") as f:
                f.write(GLOSSARY_TEMPLATE)
        subprocess.run(["open", "-t", GLOSSARY])

    @objc.python_method
    def show(self, busy=False):
        """Logo icon, always white: sparkle while loading / transcribing / cleaning up,
        else the voice wave (recording = wave + clock). White is drawn into the image: as a template,
        macOS turned it black on light menu bars and ignored contentTintColor."""
        busy = (busy or not self.ready) and not self.recording
        self.item.button().setImage_(brand_glyph("sparkle" if busy else "wave", NSColor.whiteColor(),
                                                 self.levels if self.recording else None))

    @objc.python_method
    def note(self, msg):
        self.info.setTitle_(msg if len(msg) <= 80 else msg[:79] + "…")

    def applicationWillTerminate_(self, notification):
        atexit._run_exitfuncs()            # Quit ends via C exit(), which skips Python's atexit (LLM unload)
