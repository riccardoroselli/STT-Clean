"""output.py - putting text into the focused app: keystrokes, the focus check, clipboard and ⌘V.
All of it needs the Accessibility permission; without it macOS silently drops our key events."""
import threading
import time

import objc
import Quartz
from AppKit import NSData, NSObject, NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSWorkspace
from ApplicationServices import (AXUIElementCopyAttributeValue, AXUIElementCreateApplication,
                                 AXUIElementSetMessagingTimeout, kAXFocusedUIElementAttribute, kAXFocusedWindowAttribute)

from live import diff

BACKSPACE_KEYCODE = 51
V_KEYCODE = 9             # the V key (same place on the Italian layout), for ⌘V
CMD = Quartz.kCGEventFlagMaskCommand
MODIFIERS = CMD | Quartz.kCGEventFlagMaskAlternate | Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskShift


# ---------------------------------------------------------------- keystrokes
def post_key(keycode: int = 0, char: str = "", flags: int = 0) -> None:
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
        Quartz.CGEventSetFlags(ev, flags)
        if char:
            Quartz.CGEventKeyboardSetUnicodeString(ev, len(char), char)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def retype(old: str, new: str) -> None:
    """The app shows `old` (typed by us): erase its changed tail and type the rest of `new`."""
    erase, add = diff(old, new)
    for _ in range(erase):
        post_key(BACKSPACE_KEYCODE)
    for ch in add:
        post_key(char=ch)


# ---------------------------------------------------------------- where the keystrokes land
def focus() -> tuple:
    """Where our keystrokes land: (front app pid, its focused window, its focused text field).
    A part the app doesn't report is None."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    pid = app.processIdentifier() if app else 0
    ax = AXUIElementCreateApplication(pid)
    AXUIElementSetMessagingTimeout(ax, 0.5)   # a hung app would block us 6 s per check
    # shortcut: Chrome and Electron apps (VS Code) report no text field, so a tab switch inside one of
    # their windows isn't seen; enabling their accessibility tree (AXManualAccessibility) would catch it
    return (pid,) + tuple(AXUIElementCopyAttributeValue(ax, attr, None)[1]
                          for attr in (kAXFocusedWindowAttribute, kAXFocusedUIElementAttribute))


def can_post(target: tuple, wait: float = 0) -> bool:
    """Our keystrokes go wherever the focus is, and macOS adds any modifier the user holds
    (a typed letter becomes ⌘A, Backspace ⌘⌫). True if the focus is still `target` (same app,
    window and text field) and no modifier is held, waiting up to `wait` s for the user to let go."""
    held = lambda: Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState) & MODIFIERS
    end = time.monotonic() + wait
    while held() and time.monotonic() < end:
        time.sleep(0.05)
    return not held() and focus() == target


# ---------------------------------------------------------------- clipboard and paste
class ClipText(NSObject, protocols=[objc.protocolNamed("NSPasteboardItemDataProvider")]):
    """Clipboard text handed out on demand (on the main thread): the first read is the app's ⌘V."""
    def pasteboard_item_provideDataForType_(self, pb, item, typ):
        item.setString_forType_(self.text, typ)
        self.read.set()


def copy(text: str) -> None:
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


def clipboard_items(pb) -> list:
    """A copy of everything on `pb`: every item in every type (text, rich text, images, files)."""
    items = []
    for old in pb.pasteboardItems() or []:
        item = NSPasteboardItem.alloc().init()
        for typ in old.types():
            data = old.dataForType_(typ)
            if data is not None:
                item.setData_forType_(data, typ)
        items.append(item)
    return items


def paste(text: str, typed: str) -> bool:
    """Replace the live draft `typed` with `text` (erase the changed tail, paste the rest with ⌘V),
    then restore the old clipboard once the app has read it. A fixed delay wasn't enough: after a
    long draft the app is still working through hundreds of backspaces, reads the clipboard late
    and pasted the already restored old clipboard. False if the app never read it: the text is
    then left on the clipboard for a manual ⌘V."""
    erase, text = diff(typed, text)
    for _ in range(erase):
        post_key(BACKSPACE_KEYCODE)
    if not text:
        return True
    pb = NSPasteboard.generalPasteboard()
    old = clipboard_items(pb)
    clip = ClipText.alloc().init()
    clip.text, clip.read = text, threading.Event()
    item = NSPasteboardItem.alloc().init()
    item.setDataProvider_forTypes_(clip, [NSPasteboardTypeString])
    item.setData_forType_(NSData.data(), "org.nspasteboard.TransientType")   # clipboard managers skip it (nspasteboard.org)
    pb.clearContents()
    pb.writeObjects_([item])
    post_key(V_KEYCODE, flags=CMD)
    # shortcut: a clipboard manager that ignores the transient mark and reads first counts as the paste
    if not clip.read.wait(10 + erase / 50):
        copy(text)                         # as plain data: the lazy item dies with its provider
        return False
    time.sleep(0.2)                        # let the data reach the app before the clipboard changes again
    pb.clearContents()
    if old:
        pb.writeObjects_(old)
    return True
