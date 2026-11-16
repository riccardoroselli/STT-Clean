"""Unit tests for cleanup.py (no Ollama needed):  python -m unittest discover tests"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cleanup import Cleaner, Glossary, parse_glossary, plausible, strip_wrapping  # noqa: E402

SAMPLE = """# comment
Ollama: o lama, olama
MLX: emme elle ics
Parakeet

Claude Code: cloud code
"""


def glossary_from(content: str) -> tuple[Glossary, str]:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Glossary(path), path


class TestGlossary(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_glossary(SAMPLE), [
            ("Ollama", ["o lama", "olama"]), ("MLX", ["emme elle ics"]),
            ("Parakeet", []), ("Claude Code", ["cloud code"])])

    def test_apply_whole_words_case_insensitive(self):
        g, path = glossary_from(SAMPLE)
        self.addCleanup(os.remove, path)
        self.assertEqual(g.apply("Uso O lama e emme elle ics."), "Uso Ollama e MLX.")
        self.assertEqual(g.apply("Apro cloud code, poi olama"), "Apro Claude Code, poi Ollama")
        self.assertEqual(g.apply("la lama del coltello"), "la lama del coltello")   # no partial match
        self.assertEqual(g.apply("pomodoro lama"), "pomodoro lama")

    def test_prompt_block(self):
        g, path = glossary_from(SAMPLE)
        self.addCleanup(os.remove, path)
        self.assertIn("Ollama (heard as: o lama, olama)", g.prompt_block())
        self.assertIn("\nParakeet\n", g.prompt_block())

    def test_missing_file(self):
        g = Glossary("/nonexistent/glossary.txt")
        self.assertEqual(g.apply("o lama"), "o lama")
        self.assertEqual(g.prompt_block(), "(empty)")

    def test_reload_on_change(self):
        g, path = glossary_from("Ollama: o lama\n")
        self.addCleanup(os.remove, path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("Gemma: jemma\n")
        os.utime(path, (time.time() + 5, time.time() + 5))
        self.assertTrue(g.maybe_reload())
        self.assertEqual(g.apply("o lama jemma"), "o lama Gemma")
        self.assertFalse(g.maybe_reload())


class TestSafety(unittest.TestCase):
    def test_plausible(self):
        self.assertTrue(plausible("ciao come stai", "Ciao, come stai?"))
        self.assertFalse(plausible("ciao", ""))
        self.assertFalse(plausible("Cos'è un transformer?", "Un transformer è " + "x" * 300))

    def test_strip_wrapping(self):
        self.assertEqual(strip_wrapping("ciao", '"Ciao."'), "Ciao.")
        self.assertEqual(strip_wrapping('"ciao" disse', '"Ciao" disse'), '"Ciao" disse')
        self.assertEqual(strip_wrapping("ciao", "```\nCiao.\n```"), "Ciao.")
        self.assertEqual(strip_wrapping("ciao", "«Ciao.»"), "Ciao.")

    def test_unreachable_ollama_falls_back(self):
        c = Cleaner("x", Glossary(None), timeout=0.5)
        c.url = "http://127.0.0.1:9"       # nothing listens there
        self.assertIsNone(c.clean("ciao"))
        self.assertTrue(c.last_error)
        self.assertIn("isn't running", c.check())

    def test_cut_off_output_falls_back(self):
        c = Cleaner("x", Glossary(None))
        c._post = lambda *_, **__: {"done_reason": "length", "message": {"content": "Ciao, come"}}
        self.assertIsNone(c.clean("ciao come stai"))
        self.assertIn("cut off", c.last_error)


if __name__ == "__main__":
    unittest.main()
