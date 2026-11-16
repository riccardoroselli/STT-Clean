"""Unit test for the live-draft diff (no mic/keyboard):  python -m unittest discover tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from AppKit import NSData, NSPasteboard  # noqa: E402
from live import Tail, advance, diff  # noqa: E402
from output import can_post, clipboard_items, focus  # noqa: E402


def apply(shown: str, new: str) -> str:
    erase, add = diff(shown, new)
    return shown[:len(shown) - erase] + add


class TestDiff(unittest.TestCase):
    def test_ops(self):
        self.assertEqual(diff("", "ciao"), (0, "ciao"))
        self.assertEqual(diff("aiutami a cerca", "aiutami a cercare lavoro"), (0, "re lavoro"))
        self.assertEqual(diff("lavoro a roma", "lavoro a Roma,"), (4, "Roma,"))
        self.assertEqual(diff("già scritto", "già scritto"), (0, ""))
        self.assertEqual(diff("da cancellare", ""), (13, ""))

    def test_draft_sequence_ends_as_final(self):
        shown = ""
        for new in ["aiutami", "aiutami a cerca", "aiutami a cercare lavoro a roma",
                    "aiutami a cercare lavoro a roma, no scusa ho sbagliato a milano",
                    "Aiutami a cercare lavoro a Milano. "]:
            shown = apply(shown, new)
            self.assertEqual(shown, new)


class TestAdvance(unittest.TestCase):
    def test_types_only_agreed_words(self):
        self.assertEqual(advance("", "", "aiutami a"), "")                       # nothing to compare yet
        self.assertEqual(advance("", "aiutami a cerca", "aiutami a cercare lavoro"), "aiutami a")
        self.assertEqual(advance("aiutami a", "aiutami a cercare lavoro", "aiutami a cercare lavoro a"),
                         "aiutami a cercare lavoro")

    def test_hard_word_does_not_block_the_rest(self):
        self.assertEqual(advance("API per", "API per DipsIc. Dopo aver chiamato",
                                 "API per Dipsik. Dopo aver chiamato l'API"),
                         "API per Dipsik. Dopo aver chiamato")
        # a lone common word matching further on is not enough to skip ahead
        self.assertEqual(advance("vedo", "vedo due punti e", "vedo dei box e"), "vedo")

    def test_continues_where_typed_text_sits_in_draft(self):
        # a word split mid-dictation ("4." -> "Quattro,") must not duplicate or skip words
        self.assertEqual(advance("giochi per la PlayStation 4.",
                                 "giochi per la PlayStation Quattro, e infine",
                                 "giochi per la PlayStation Quattro, e infine, io"),
                         "giochi per la PlayStation Quattro, e infine,")

    def test_never_stuck_when_typed_words_changed(self):
        self.assertEqual(advance("Provia", "Proviamo a registrare", "Proviamo a registrare qualcosa"),
                         "Proviamo a registrare")
        self.assertEqual(advance("API per Dips Ic.", "API per Dipsik. Dopo aver chiamato",
                                 "API per Dipsik. Dopo aver chiamato l"),
                         "API per Dipsik. Dopo aver chiamato")

    def test_silence_types_everything(self):
        self.assertEqual(advance("a Roma", "a Roma, ciao.", "a Roma, ciao."), "a Roma, ciao.")

    def test_last_word_may_change(self):
        self.assertEqual(advance("lavoro a Roma.", "lavoro a Roma, no", "lavoro a Roma, no scusa"),
                         "lavoro a Roma, no")

    def test_older_words_stay_frozen(self):
        self.assertEqual(advance("ci sei. alcune", "ci sei? Alcune proteine x", "ci sei? Alcune proteine y"),
                         "ci sei. Alcune proteine")     # "sei." frozen, last word "alcune" fixed
        self.assertEqual(advance("ci sei. alcune", "ci sei?", "ci sei? Alcune"), "ci sei. alcune")


class TestCanPost(unittest.TestCase):
    def test_other_app_in_front_blocks_typing(self):
        self.assertFalse(can_post((-1, None, None)))        # no app has pid -1: never "still in front"
        self.assertTrue(can_post(focus()))                  # nothing moved (and no key held while testing)


class TestClipboard(unittest.TestCase):
    def test_restore_keeps_every_type(self):
        pb = NSPasteboard.pasteboardWithUniqueName()        # private, not the user's clipboard
        self.addCleanup(pb.releaseGlobally)
        png = NSData.dataWithBytes_length_(b"\x89PNG fake", 9)
        pb.clearContents()
        pb.declareTypes_owner_(["public.png", "public.utf8-plain-text"], None)
        pb.setData_forType_(png, "public.png")
        pb.setString_forType_("città", "public.utf8-plain-text")
        saved = clipboard_items(pb)
        pb.clearContents()
        pb.setString_forType_("dictated", "public.utf8-plain-text")
        pb.clearContents()
        pb.writeObjects_(saved)
        self.assertEqual(bytes(pb.dataForType_("public.png")), b"\x89PNG fake")
        self.assertEqual(pb.stringForType_("public.utf8-plain-text"), "città")


class TestTail(unittest.TestCase):
    """Fake speech model: one "sentence" per second of audio at 10 samples/s, named after the second it
    starts in; the newest one is still changing (ends with "?") until audio follows it."""

    def setUp(self):
        self.calls = []       # seconds of audio in each transcription

        def transcribe(audio):
            self.calls.append(len(audio) / 10)
            n = len(audio) // 10
            return [(f"s{int(audio[i * 10])}" + ("?" if i == n - 1 else "."), i, i + 1.0) for i in range(n)]
        self.tail = Tail(transcribe, 10)
        self.audio = [t // 10 for t in range(3000)]      # 300 s; each sample holds its second

    def test_live_drafts_only_redo_the_unfinished_tail(self):
        for n in range(5, 1205, 5):                      # a draft every 0.5 s for 2 minutes
            draft = self.tail.draft(self.audio[:n])
        self.assertEqual(draft, " ".join(f"s{i}." for i in range(119)) + " s119?")
        self.assertLessEqual(max(self.calls), 5)         # never more than a few seconds, however long the clip
        self.assertEqual(self.tail.final(self.audio[:1200]), draft)

    def test_sentence_needs_two_drafts_a_follower_and_the_margin(self):
        self.assertEqual(self.tail.draft(self.audio[:40]), "s0. s1. s2. s3?")
        self.assertEqual(self.tail.start, 0)             # first draft: nothing to agree with
        self.tail.draft(self.audio[:40])
        self.assertEqual((self.tail.start, self.tail.done), (20, ["s0.", "s1."]))   # s2 ended <2 s ago, s3 is last

    def test_final_without_drafts_goes_in_windows(self):
        text = self.tail.final(self.audio)               # live text off: 300 s at once
        self.assertEqual(text, " ".join(f"s{i}." for i in range(299)) + " s299?")
        self.assertLessEqual(max(self.calls), 60)

    def test_window_without_sentence_end_is_cut(self):
        tail = Tail(lambda audio: [("word", 0, len(audio) / 10)] if len(audio) else [], 10)
        self.assertEqual(tail.final(self.audio[:1500]), "word word word")   # 60 + 60 + 30 s

    def test_short_and_empty(self):
        self.assertEqual(self.tail.final(self.audio[:0]), "")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
