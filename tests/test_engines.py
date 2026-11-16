"""Unit tests for the pluggable parts: speech engines (stt.py), LLM servers (cleanup.py).

    python -m unittest discover tests
"""
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import stt  # noqa: E402
from cleanup import Cleaner, Glossary, strip_wrapping  # noqa: E402


class TestSpeechEngines(unittest.TestCase):
    def test_unknown_engine(self):
        with self.assertRaisesRegex(ValueError, "parakeet, whisper"):
            stt.load("piper")

    def test_available_lists_loadable_models_only(self):
        parakeet = {"target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"}
        whisper = {"n_mels": 80, "n_audio_state": 384}
        with tempfile.TemporaryDirectory() as hub:
            for repo, config, weights in [
                    ("mlx-community--parakeet-tdt-0.6b-v3", parakeet, "model.safetensors"),
                    ("me--my-italian-asr", parakeet, "model.safetensors"),               # found by its config, not its name
                    ("mlx-community--whisper-tiny", whisper, "weights.npz"),
                    ("openai--whisper-large-v3", {"model_type": "whisper"}, "model.safetensors"),   # Transformers format: no
                    ("mlx-community--whisper-small-mlx", whisper, "weights.npz.incomplete"),    # downloading
                    ("mlx-community--whisper-broken", None, "weights.npz"),                    # no config.json
                    ("Comfy-Org--z_image_turbo", {"model_type": "z"}, "model.safetensors")]:    # not speech
                snap = os.path.join(hub, f"models--{repo}", "snapshots", "abc")
                os.makedirs(snap)
                open(os.path.join(snap, weights), "w").close()
                if config:
                    with open(os.path.join(snap, "config.json"), "w") as f:
                        json.dump(config, f)
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": hub}), \
                    mock.patch("importlib.util.find_spec", return_value=True):
                self.assertEqual(stt.available(), [("parakeet", "me/my-italian-asr"),
                                                   ("parakeet", "mlx-community/parakeet-tdt-0.6b-v3"),
                                                   ("whisper", "mlx-community/whisper-tiny")])
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": hub}), \
                    mock.patch("importlib.util.find_spec", side_effect=lambda p: p == "parakeet_mlx"):
                self.assertEqual({e for e, _ in stt.available()}, {"parakeet"})   # mlx-whisper not installed


class TestLLMProviders(unittest.TestCase):
    def test_unknown_provider(self):
        for provider in ("openai", "port:", "port:80a"):
            with self.assertRaisesRegex(ValueError, "ollama, lmstudio or port:<number>"):
                Cleaner("x", "", Glossary(None), provider=provider)

    def test_default_urls(self):
        self.assertEqual(Cleaner("x", "", Glossary(None)).url, "http://127.0.0.1:11434")
        self.assertEqual(Cleaner("x", "", Glossary(None), provider="lmstudio").url, "http://127.0.0.1:1234/v1")

    def test_lmstudio_request_and_reply(self):
        c = Cleaner("qwen3", "http://h/v1/", Glossary(None), provider="lmstudio")
        sent = {}

        def post(path, body, timeout):
            sent.update(path=path, body=body)
            return {"choices": [{"message": {"content": "<think>\nhmm\n</think>\n\nCiao, come stai?"},
                                 "finish_reason": "stop"}]}
        c._post = post
        self.assertEqual(c.clean("ciao come stai"), "Ciao, come stai?")
        self.assertEqual(sent["path"], "/chat/completions")
        self.assertEqual(sent["body"]["temperature"], 0)
        self.assertNotIn("think", sent["body"])        # Ollama-only fields stay out

    def test_lmstudio_strict_server_drops_thinking_fields(self):
        c = Cleaner("x", "", Glossary(None), provider="lmstudio")
        bodies = []

        def post(path, body, timeout):
            bodies.append(body)
            if "reasoning_effort" in body:
                raise urllib.error.HTTPError(path, 400, "Unsupported parameter", {}, None)
            return {"choices": [{"message": {"content": "Ciao."}, "finish_reason": "stop"}]}
        c._post = post
        self.assertEqual(c.clean("ciao"), "Ciao.")
        self.assertEqual(c.clean("ciao"), "Ciao.")
        self.assertEqual(len(bodies), 3)               # one rejected request, then plain ones only

    def test_other_400_keeps_thinking_off(self):
        c = Cleaner("x", "", Glossary(None), provider="lmstudio")

        def post(path, body, timeout):               # e.g. a dictation too long for the server's context
            raise urllib.error.HTTPError(path, 400, "context length exceeded", {}, None)
        c._post = post
        self.assertIsNone(c.clean("ciao"))
        self.assertTrue(c.no_think)

    def test_lmstudio_cut_off(self):
        c = Cleaner("x", "", Glossary(None), provider="lmstudio")
        c._post = lambda *_, **__: {"choices": [{"message": {"content": "Ciao"}, "finish_reason": "length"}]}
        self.assertIsNone(c.clean("ciao come stai"))
        self.assertIn("cut off", c.last_error)

    def test_lmstudio_check(self):
        c = Cleaner("missing", "", Glossary(None), provider="lmstudio")
        c._post = lambda *_, **__: {"data": [{"id": "qwen3-8b"}, {"id": "text-embedding-nomic"}]}
        self.assertEqual(c.models(1), ["qwen3-8b"])     # embedding models can't chat
        self.assertIn("isn't in LM Studio", c.check())
        c.model = "qwen3-8b"
        self.assertIsNone(c.check())

    def test_any_local_server_speaks_the_openai_api(self):
        c = Cleaner("gemma", "", Glossary(None), provider="port:9931")
        self.assertEqual((c.url, c.label), ("http://127.0.0.1:9931/v1", "the server on port 9931"))
        sent = []
        c._post = lambda path, body, timeout, base="": sent.append(path) or {
            "choices": [{"message": {"content": "Ciao."}, "finish_reason": "stop"}]}
        self.assertEqual(c.clean("ciao"), "Ciao.")
        c.unload()                                   # nothing to free there: no request
        self.assertEqual(sent, ["/chat/completions"])

    def test_local_ports(self):
        import cleanup
        lsof = "p1\ncollama\nn127.0.0.1:11434\np2\ncllama-server\nn127.0.0.1:50291\nn*:9931\nn[::1]:9931\n" \
               "p3\ncpostgres\nn192.168.1.5:5432\np4\nclaunchd\nn*:22\n"
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=lsof)):
            # not: temporary ports (Ollama's own runner), other addresses, system ports
            self.assertEqual(cleanup.local_ports(), [(9931, "llama-server"), (11434, "ollama")])
        with mock.patch("subprocess.run", side_effect=OSError("no lsof")):
            self.assertEqual(cleanup.local_ports(), [])

    def test_available_finds_servers_on_any_port(self):
        import cleanup
        asked = []

        def models(self, timeout):
            asked.append(self.provider)
            if self.provider not in ("ollama", "port:9931"):
                raise OSError("connection refused, or not an LLM server")
            return ["m1"] if self.provider == "port:9931" else []     # Ollama runs, but has no model
        ports = [(1234, "LM Studio"), (5000, "ControlCenter"), (9931, "llama-server"), (11434, "ollama")]
        with mock.patch.object(Cleaner, "models", models), mock.patch.object(cleanup, "local_ports", return_value=ports), \
                mock.patch.object(cleanup, "_not_llm", set()):
            self.assertEqual(cleanup.available(), [("llama-server (port 9931)", "port:9931", ["m1"])])
            self.assertEqual(sorted(asked), ["lmstudio", "ollama", "port:5000", "port:9931"])
            cleanup.available()
            self.assertEqual(asked.count("port:5000"), 1)            # not an LLM server: not asked again
            self.assertEqual(asked.count("port:9931"), 2)

    def test_start_needs_a_known_server(self):
        self.assertFalse(Cleaner("x", "", Glossary(None), provider="port:9931").start(wait=0))

    def test_style_adds_one_rule(self):
        c = Cleaner("x", "", Glossary(None))
        self.assertNotIn("\n8.", c.system_prompt())
        self.assertIn("\n8. Then format the result as a bullet list", c.system_prompt("bullets"))
        self.assertEqual(c.system_prompt("gone-in-an-update"), c.system_prompt())   # a saved style that no longer exists
        sent = {}
        c._post = lambda path, body, timeout: sent.update(body) or {"message": {"content": "- a\n- b"}}
        self.assertEqual(c.clean("a e poi b", style="bullets"), "- a\n- b")
        self.assertIn("bullet list", sent["messages"][0]["content"])

    def test_think_block_only_at_start(self):
        self.assertEqual(strip_wrapping("x", "Use <think></think> tags"), "Use <think></think> tags")


class TestHotKey(unittest.TestCase):
    def test_combo_and_dispatch(self):
        import ctypes
        import hotkey
        pressed = []
        esc = hotkey.HotKey("Esc", lambda: pressed.append("esc"))
        key = hotkey.HotKey("⌃⌥D", lambda: pressed.append("shortcut"))
        self.assertEqual((esc.keycode, esc.modifiers, key.keycode, key.modifiers), (53, 0, 2, 0x1800))
        # a hot-key event as Carbon sends it, for each key: the one handler calls the right callback
        carbon = hotkey._carbon
        carbon.CreateEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double,
                                       ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        carbon.SetEventParameter.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_ulong, ctypes.c_void_p]
        for k in (key, esc):
            event, hk = ctypes.c_void_p(), hotkey._HotKeyID(hotkey._code(b"sttc"), k.id)
            carbon.CreateEvent(None, hotkey._code(b"keyb"), 5, 0, 0, ctypes.byref(event))
            carbon.SetEventParameter(event, hotkey._code(b"----"), hotkey._code(b"hkid"), ctypes.sizeof(hk), ctypes.byref(hk))
            hotkey._dispatch(None, event, None)
        self.assertEqual(pressed, ["shortcut", "esc"])


class TestSettings(unittest.TestCase):
    def test_saved_values_keep_their_types(self):
        import menubar
        from AppKit import NSUserDefaults
        prefs = NSUserDefaults.alloc().initWithSuiteName_("stt-clean-unittest")
        self.addCleanup(prefs.removePersistentDomainForName_, "stt-clean-unittest")
        with mock.patch.object(menubar, "PREFS", prefs):
            self.assertEqual(menubar.load_settings(), menubar.DEFAULTS)          # nothing saved yet
            prefs.setObject_forKey_(False, "live")
            prefs.setObject_forKey_("qwen3:8b", "llm_model")
            s = menubar.load_settings()
        self.assertIs(s["live"], False)
        self.assertEqual((s["llm_model"], type(s["llm_model"]), s["llm"]), ("qwen3:8b", str, True))


if __name__ == "__main__":
    unittest.main()
