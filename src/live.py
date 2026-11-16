"""live.py - what the live draft shows while you speak, and which audio still has to be transcribed:
pure logic, no macOS calls."""
import os
from difflib import SequenceMatcher


def diff(old: str, new: str) -> tuple[int, str]:
    """(chars to erase from the end of old, text to add after that) to turn old into new."""
    n = len(os.path.commonprefix([old, new]))
    return len(old) - n, new[n:]


MARGIN = 2.0              # a sentence is finished once it ended this many seconds ago
WINDOW = 60               # seconds taken at a time when a long recording has no draft yet


class Tail:
    """Transcribes a growing recording without redoing its finished sentences. Each draft only
    transcribes the audio after the last finished sentence, so an update costs the same after 3
    minutes as after 10 s (the whole clip again cost 0.3 s at 10 s, 1.6 s at 60 s, 6.6 s at 180 s).
    A sentence is finished when two drafts in a row agree on it, another sentence follows it and
    it ended MARGIN s ago; the cut is in the middle of the gap to the next one.
    Measured on 25-70 s clips: same words as the whole clip, a few commas differ."""

    def __init__(self, transcribe, sr: int):
        self.transcribe, self.sr = transcribe, sr   # transcribe(audio) -> [(text, start, end), ...]
        self.start = 0        # samples before this are finished
        self.done: list = []  # their sentences
        self.prev: list | None = []   # the last draft's sentences after `start` (None: take any)

    def draft(self, audio) -> str:
        """Text of `audio`, the whole recording so far."""
        rest = audio[self.start:]
        seconds = len(rest) / self.sr
        segs = self.transcribe(rest) if seconds >= 0.1 else []
        k = 0
        while (k < len(segs) - 1 and segs[k][2] <= seconds - MARGIN
               and (self.prev is None or self.prev[k:k + 1] == [segs[k][0]])):
            k += 1
        if k:
            self.start += int((segs[k - 1][2] + segs[k][1]) / 2 * self.sr)
            self.done += [s[0] for s in segs[:k]]
        self.prev = [s[0] for s in segs[k:]]
        return " ".join(self.done + self.prev)

    def final(self, audio) -> str:
        """Text of the finished recording. A long stretch without drafts (live text off) is taken
        WINDOW s at a time: the whole clip at once needs time and memory that grow much faster
        than its length."""
        step = WINDOW * self.sr
        while len(audio) - self.start > step:
            start, self.prev = self.start, None    # no earlier draft: the window's later audio is the context
            self.draft(audio[:start + step])
            if self.start == start:                # not one sentence end in the window: cut at its end
                self.done, self.start = self.done + self.prev, start + step
        return self.draft(audio)


def _words(ws: list) -> list:
    return [w.strip(".,;:!?…\"'«»()").lower() for w in ws]


def _blocks(x: list, y: list) -> list:
    return SequenceMatcher(None, x, y, autojunk=False).get_matching_blocks()


def advance(typed: str, prev: str, draft: str) -> str:
    """Live text after an update. Adds the words the last two drafts agree on (ignoring case and
    punctuation), continuing from where the typed text sits in the newest draft. Typed words stay
    frozen (the final pass fixes them) except the last one, which may still change, e.g. its comma;
    so the text in the app only grows instead of flickering."""
    a, b, t = prev.split(), draft.split(), typed.split()
    bn, tn = _words(b), _words(t)
    # agreed up to the end of the last run of ≥2 shared words (or a shared first word), so one
    # hard word that keeps changing doesn't hold back the stable words after it
    n = max((m.b + m.size for m in _blocks(_words(a), bn) if m.size >= 2 or (m.size and m.a == m.b == 0)),
            default=0)
    if not t:
        return " ".join(b[:n])
    # where the typed text sits in the newest draft: its last run of ≥2 matching words (e, j = end in
    # typed, end in draft). Typed words after that run changed and are erased: normally only the last
    # one (e.g. its comma), rarely more ("Dips Ic." -> "Dipsik."); without this it would stay stuck.
    e = j = 0
    for m in reversed(_blocks(tn, bn)):
        if m.size >= min(2, len(t)):
            e, j = m.a + m.size, m.b + m.size
            break
    if e == len(t):                       # all typed words matched: still refresh the last one
        e, j = e - 1, j - 1
    if j >= n:
        return typed                      # nothing agreed past it yet: wait for the next draft
    return " ".join(t[:e] + b[j:n])
