"""hotkey.py - global hot keys: Esc while recording (cancel) and the start/stop shortcut."""
import ctypes

_carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class _HotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


class _EventType(ctypes.Structure):
    _fields_ = [("cls", ctypes.c_uint32), ("kind", ctypes.c_uint32)]


def _code(four: bytes) -> int:
    return int.from_bytes(four, "big")


_carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
_carbon.InstallEventHandler.argtypes = [ctypes.c_void_p, _HANDLER, ctypes.c_ulong, ctypes.POINTER(_EventType),
                                        ctypes.c_void_p, ctypes.c_void_p]
_carbon.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _HotKeyID, ctypes.c_void_p,
                                        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
_carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
_carbon.GetEventParameter.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                      ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]

MODIFIERS = {"⌘": 0x100, "⇧": 0x200, "⌥": 0x800, "⌃": 0x1000}   # Carbon cmdKey, shiftKey, optionKey, controlKey
KEYCODES = {"Esc": 53, "Space": 49, "D": 2, "F5": 96}             # key positions: the same on every layout
_pressed: dict = {}       # hot key id -> its callback


@_HANDLER
def _dispatch(_call, event, _data):
    """One handler for all our hot keys: Carbon only calls the newest one installed on a target."""
    key = _HotKeyID()
    _carbon.GetEventParameter(event, _code(b"----"), _code(b"hkid"), None, ctypes.sizeof(key), None, ctypes.byref(key))
    if key.id in _pressed:
        _pressed[key.id]()
    return 0


class HotKey:
    """A global hot key like "Esc" or "⌥Space" (Carbon RegisterEventHotKey): reaches us in any app, is
    swallowed there and needs no permission (a keyboard event tap would need Input Monitoring).
    Main thread only. It only works between on() and off()."""

    def __init__(self, combo: str, pressed):
        self.modifiers = sum(MODIFIERS[c] for c in combo if c in MODIFIERS)
        self.keycode = KEYCODES[combo.lstrip("".join(MODIFIERS))]
        self.pressed, self.ref = pressed, None
        if not _pressed:
            kind = _EventType(_code(b"keyb"), 5)   # kEventClassKeyboard, kEventHotKeyPressed
            _carbon.InstallEventHandler(_carbon.GetApplicationEventTarget(), _dispatch, 1, ctypes.byref(kind), None, None)
        self.id = max(_pressed, default=0) + 1
        _pressed[self.id] = lambda: self.pressed()

    def on(self) -> bool:
        ref = ctypes.c_void_p()
        err = _carbon.RegisterEventHotKey(self.keycode, self.modifiers, _HotKeyID(_code(b"sttc"), self.id),
                                          _carbon.GetApplicationEventTarget(), 0, ctypes.byref(ref))
        self.ref = ref if err == 0 else None
        return err == 0                    # fails if another app already holds this key

    def off(self) -> None:
        if self.ref:
            _carbon.UnregisterEventHotKey(self.ref)
            self.ref = None
