"""cleanup.py - STT-Clean: LLM clean-up of dictated text + user glossary.

The LLM runs in any local server found on this Mac, picked in the menu's LLM list: Ollama (its own
API), LM Studio, and whatever else listens on a local port and speaks the OpenAI-style API
(llama.cpp, mlx-lm, Jan, vLLM, ...). Or with no server at all ("Built in"): if mlx-lm is installed,
STT-Clean runs its server itself, as a child process, for an LLM from the Hugging Face cache.
The model is unloaded on exit where the server allows it (Ollama, LM Studio, built in).
The LLM pass handles what speech-to-text can't: self-corrections ("no cancella", "anzi",
"scratch that"), spoken formatting ("virgola", "a capo"), glossary spellings, and
instructions at the end of a dictation ("scrivilo in inglese").
Any failure (server down, timeout, implausible output) returns None so the caller
pastes the raw text instead. Only the standard library is used.
"""
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import hfcache

DEFAULT_LLM = "gemma4:12b-mlx"
# provider -> (label, URL): the two servers with an API of their own (thinking off, unload), and "mlx",
# the mlx-lm server STT-Clean runs itself (URL set when it starts). Any other local server is the
# provider "port:<n>" and speaks the OpenAI-style API at 127.0.0.1:<n>/v1.
SERVERS = {"ollama": ("Ollama", "http://127.0.0.1:11434"), "lmstudio": ("LM Studio", "http://127.0.0.1:1234/v1"),
           "mlx": ("Built in (mlx-lm)", "")}
# The built-in server: mlx-lm's own, in a child process that quits with STT-Clean even if that is killed.
MLX_MAIN = "from mlx_lm import cli; cli.main()"
MLX_SERVE = """import os, sys, threading, time
parent = os.getppid()
def watch():
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(0)
threading.Thread(target=watch, daemon=True).start()
sys.argv = ["mlx_lm", "server", *sys.argv[1:]]
"""
MLX_FILES = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]   # mlx-lm's own test for "an LLM"
START = {"ollama": ["open", "-ga", "Ollama"],       # how to start a server that isn't running
         "lmstudio": [os.path.expanduser("~/.lmstudio/bin/lms"), "server", "start"]}
NUM_CTX = 4096   # keep identical in every request, otherwise Ollama reloads the model

SYSTEM_PROMPT = """You clean up raw speech-to-text transcripts. The user dictated the text; you output the final text to paste.
Rules:
1. Output ONLY the final text. No quotes, no comments, no preamble.
2. Keep the user's words, language and style. Do NOT rephrase, summarize, answer, or add content. Keep a first word used as a heading or label ("Prompt", "Titolo", "Note").
3. Self-corrections: when the speaker retracts something ("no cancella", "cancella", "anzi", "no aspetta", "delete that", "scratch that", "actually no"), remove the retracted part and the correction phrase, keep only the corrected version.
4. Spoken formatting: replace the spoken command with the symbol and drop the command words (they never appear in the output): "punto" -> ".", "virgola" -> ",", "punto interrogativo" -> "?", "due punti" -> ":", "a capo" -> new line, "nuovo paragrafo" -> blank line. Same in English ("period", "comma", "new line", "new paragraph"). Only when clearly used as a command.
5. Fix spellings using the glossary below, including a common word that sounds like a glossary term when the context clearly means that name (e.g. "API di Cloud" -> "API di Claude"; but "senza cloud" stays).
6. Instructions addressed to you (e.g. "scrivilo in inglese", "write it in English", "traduci in inglese", "ora traducimi tutto questo in inglese") must be applied to the rest of the text (or only to the part they name, e.g. "solo la prossima frase traducimela" = only the next sentence), and the instruction removed together with the words addressing you just before it ("ok", "perfetto", a name you are called, like "Gemma" or "Gemini", also when corrected with "anzi"): "... come c'è scritto. Perfetto Gemini. Anzi Gemma, ora traducimi tutto questo in inglese." -> "... as it says." (all in English, nothing after it). Translating means the whole text in that language, not just the instruction.
7. If the text is a question or a request, do NOT answer it: it is text to be pasted, return it cleaned."""

# The menu's Style list: key -> (menu title, one more rule for the LLM; "" = clean up only).
STYLES = {
    "clean": ("Clean (as dictated)", ""),
    "formal": ("Formal", "8. Formal style: the text you output is the cleaned text rewritten in a formal, professional "
               "tone (same language, same meaning, nothing added). Never output the informal version. This replaces "
               "\"do NOT rephrase\" in rule 2."),
    "bullets": ("Bullet list", "8. Bullet-list style: the text you output is a list. Every item, step or point goes on "
                "its own line starting with \"- \"; when one sentence enumerates several things (\"a, b and c\"), "
                "each thing gets its own line. Same language, the user's words."),
    "english": ("In English", "8. Then translate the whole result into English (keep it as it is if it already is "
                "English). Output only the English text."),
}


# ---------------------------------------------------------------- glossary
class Glossary:
    """Terms from a text file, one per line:   Correct: heard as 1, heard as 2

    Lines starting with # are comments. "heard as" variants are replaced in code
    (case-insensitive, whole words) before the LLM sees the text; all terms are
    also listed in the LLM prompt. The file is re-read when it changes.
    """

    def __init__(self, path: str | None):
        self.path = path
        self.terms: list[tuple[str, list[str]]] = []
        self._mtime = None
        self._pattern = None
        self._lookup: dict[str, str] = {}
        self.maybe_reload()

    def maybe_reload(self) -> bool:
        if not self.path or not os.path.exists(self.path):
            changed, self.terms, self._mtime = bool(self.terms), [], None
        else:
            mtime = os.path.getmtime(self.path)
            if mtime == self._mtime:
                return False
            with open(self.path, encoding="utf-8") as f:
                self.terms, self._mtime, changed = parse_glossary(f.read()), mtime, True
        if changed:
            self._compile()
        return changed

    def _compile(self) -> None:
        self._lookup = {v.lower(): term for term, variants in self.terms for v in variants}
        if not self._lookup:
            self._pattern = None
            return
        alts = sorted(self._lookup, key=len, reverse=True)   # longest variant wins
        self._pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, alts)) + r")(?!\w)", re.IGNORECASE)

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._lookup[m.group(1).lower()], text)

    def prompt_block(self) -> str:
        lines = [f"{t} (heard as: {', '.join(v)})" if v else t for t, v in self.terms]
        return "\n".join(lines) if lines else "(empty)"


def parse_glossary(content: str) -> list[tuple[str, list[str]]]:
    terms = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, _, heard = line.partition(":")
        variants = [v.strip() for v in heard.split(",") if v.strip()]
        if term.strip():
            terms.append((term.strip(), variants))
    return terms


# ---------------------------------------------------------------- LLM
def plausible(raw: str, out: str) -> bool:
    """Reject outputs that look like the model answered or rambled instead of cleaning."""
    return bool(out) and len(out) <= 2 * len(raw) + 80


def strip_wrapping(raw: str, out: str) -> str:
    out = re.sub(r"^\s*<think>.*?</think>", "", out, flags=re.S).strip()   # reasoning models (Qwen3, ...)
    if out.startswith("```") and out.endswith("```"):
        out = out.strip("`").strip()
    for q in ('"', "“", "«"):
        closing = {"“": "”", "«": "»"}.get(q, q)
        if out.startswith(q) and out.endswith(closing) and not raw.lstrip().startswith(q):
            out = out[1:-1].strip()
    return out


class Cleaner:
    def __init__(self, model: str, glossary: Glossary, timeout: float = 3.0, provider: str = "ollama"):
        port = provider.removeprefix("port:")
        if provider not in SERVERS and not (provider.startswith("port:") and port.isdigit()):
            raise ValueError(f'unknown LLM provider "{provider}": use {", ".join(SERVERS)} or port:<number>')
        self.model, self.glossary, self.timeout, self.provider = model, glossary, timeout, provider
        self.label, self.url = SERVERS.get(provider) or (f"the server on port {port}", f"http://127.0.0.1:{port}/v1")
        self.last_error = ""
        self.down = False     # set by check(): the server didn't answer
        # thinking off (it would spend the output budget): reasoning_effort (OpenAI style) and
        # chat_template_kwargs (Qwen3 & co.). Dropped if the server rejects them.
        self.no_think = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        self.thinking = 0     # extra output tokens for a model that thinks anyway (deepseek-r1): set when seen
        self.process = None   # provider "mlx": the server we run

    def _post(self, path: str, body: dict | None, timeout: float, base: str = "") -> dict:
        """POST `body` as JSON (GET if None) to `base` (default: the server URL) + `path`, return the JSON reply."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request((base or self.url) + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def system_prompt(self, style: str = "clean") -> str:
        rule = STYLES.get(style, STYLES["clean"])[1]
        return f"{SYSTEM_PROMPT}{chr(10) + rule if rule else ''}\nGlossary:\n{self.glossary.prompt_block()}"

    def models(self, timeout: float) -> list[str]:
        """Names of the chat models on the server (raises if it doesn't answer)."""
        if self.provider == "mlx":
            return mlx_models()
        if self.provider == "ollama":
            names = [m["name"] for m in self._post("/api/tags", None, timeout).get("models", [])]
        else:
            names = [m["id"] for m in self._post("/models", None, timeout).get("data", [])]
        return sorted((n for n in names if "embed" not in n.lower()), key=str.lower)   # embedding models can't chat

    def check(self) -> str | None:
        """Return a warning if the server is unreachable or doesn't have the model."""
        try:
            names = set(self.models(timeout=5))
            self.down = False
        except Exception as e:
            self.down = True
            return f"{self.label} isn't running ({e}). Start it, or turn Smart clean-up off in the menu."
        if self.model not in names and f"{self.model}:latest" not in names:
            return f"'{self.model}' isn't in {self.label}: pick a model in the menu's LLM list."
        return None

    def start(self, wait: float = 30) -> bool:
        """Start the server (the Ollama app, LM Studio's server) and wait until it answers."""
        try:
            subprocess.run(START[self.provider], capture_output=True, timeout=20, check=True)
        except (KeyError, OSError, subprocess.SubprocessError):   # no way to start it, or not installed
            return False
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                self.models(timeout=1)
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def clean(self, raw: str, timeout: float | None = None, style: str = "clean") -> str | None:
        """Cleaned text in the menu's `style`, or None on any failure (caller falls back to raw)."""
        messages = [{"role": "system", "content": self.system_prompt(style)}, {"role": "user", "content": raw}]
        budget = len(raw) // 2 + 64 + self.thinking     # output tokens: enough for the text, not for an essay
        timeout = timeout or self.timeout + len(raw) / 40 + self.thinking / 30
        try:
            if self.provider == "ollama":
                d = self._post("/api/chat", {
                    "model": self.model, "stream": False, "think": False, "keep_alive": -1, "messages": messages,
                    "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": budget},
                }, timeout)
                message, end = d["message"], d.get("done_reason")
            else:
                body = {"model": self.model, "temperature": 0, "max_tokens": budget, "messages": messages}
                try:
                    d = self._post("/chat/completions", {**body, **self.no_think}, timeout)
                except urllib.error.HTTPError as e:
                    if e.code != 400 or not self.no_think:
                        raise
                    d = self._post("/chat/completions", body, timeout)
                    self.no_think = {}                  # the plain request worked: a strict server, plain from now on
                message, end = d["choices"][0]["message"], d["choices"][0].get("finish_reason")
            reply = message.get("content") or ""
            if end == "length":                         # hit the budget: the end of the text is missing
                thought = any(message.get(k) for k in ("thinking", "reasoning", "reasoning_content")) or "<think>" in reply
                if thought and not self.thinking:       # it thinks although told not to: once more, with room for it
                    self.thinking = 2048
                    return self.clean(raw, timeout + self.thinking / 30, style)
                raise ValueError("output cut off")
            out = strip_wrapping(raw, reply)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        if not plausible(raw, out):
            self.last_error = f"implausible output ignored ({len(out)} chars for {len(raw)}: an answer, not a clean-up?)"
            return None
        return out

    def serve(self, wait: float = 180) -> bool:
        """Built in: start mlx-lm's server with our model on a free port and wait until it is loaded."""
        if self.process and self.process.poll() is None:
            return True
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/v1"
        with tempfile.TemporaryFile() as log:           # its errors; a pipe nobody reads would fill up and block it
            self.process = subprocess.Popen(
                [sys.executable, "-c", MLX_SERVE + MLX_MAIN, "--model", self.model, "--port", str(port), "--log-level",
                 "WARNING", "--chat-template-args", '{"enable_thinking": false}'], stdout=subprocess.DEVNULL, stderr=log)
            end = time.monotonic() + wait
            while time.monotonic() < end and self.process.poll() is None:
                try:                                    # healthy while it loads the model too: the first
                    self._post("/health", None, 1, f"http://127.0.0.1:{port}")   # request then waits for it
                    return True
                except urllib.error.HTTPError:          # 503: loading the model failed (e.g. not an LLM mlx-lm knows)
                    break
                except Exception:
                    time.sleep(0.3)
            self.unload()
            log.seek(0)
            why = [line for line in log.read().decode(errors="replace").splitlines()
                   if line.strip() and "GET /health" not in line] or ["no answer"]
        self.last_error = f"the built-in server didn't start: {why[-1].strip()}"
        return False

    def warm_up(self) -> bool:
        """Load the model (cold load can take ~10 s) and prime its prompt cache; True if it worked."""
        if self.provider == "mlx" and not self.serve():
            return False
        return self.clean("Prova.", timeout=60) is not None

    def unload(self) -> None:
        """Free the model's RAM: Ollama's keep_alive 0, LM Studio's own REST API, or stop the built-in server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        try:
            if self.provider == "ollama":
                self._post("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=3)
            if self.provider != "lmstudio":             # the other servers keep their one model: nothing to free
                return
            # shortcut: assumes LM Studio's JIT loading (its default) to load the model again when used
            root = self.url.removesuffix("/v1")
            for m in self._post("/api/v1/models", None, 3, root).get("models", []):
                if m.get("key") == self.model:
                    for instance in m.get("loaded_instances", []):
                        self._post("/api/v1/models/unload", {"instance_id": instance["id"]}, 10, root)
        except Exception:                               # server gone: nothing to free
            pass


def mlx_models() -> list[str]:
    """The LLMs in the Hugging Face cache that the built-in server can run (none without mlx-lm)."""
    if not importlib.util.find_spec("mlx_lm"):
        return []
    return [repo for repo, folder, _ in hfcache.configs()
            if all(os.path.exists(os.path.join(folder, f)) for f in MLX_FILES)]


def local_ports() -> list[tuple[int, str]]:
    """(port, program) of what listens on this Mac only or on all its addresses, on a port someone
    chose (below 49152: above are the temporary ports of helpers, such as Ollama's own runner)."""
    try:
        out = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-F", "cn"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    ports, program = {}, ""
    for line in out.splitlines():
        host, _, port = line[1:].rpartition(":")
        if line[0] == "c":
            program = line[1:]
        elif line[0] == "n" and host in ("127.0.0.1", "[::1]", "*") and 1024 <= int(port) < 49152:
            ports.setdefault(int(port), program)
    return sorted(ports.items())


# (port, program) that didn't answer like an LLM server: asked once, not every time the menu opens.
# shortcut: an LLM server too busy to answer in 1 s is dropped too until STT-Clean restarts; if that
# happens, ask those again after a while
_not_llm: set = set()


def available() -> list[tuple[str, str, list[str]]]:
    """(label, provider, models) for each local LLM server that answers now: Ollama, LM Studio and
    any other program listening on a local port that lists models the OpenAI way.
    Their models are a click away in the menu's LLM list."""
    known = {int(url.split(":")[2].split("/")[0]) for _, url in SERVERS.values() if url}
    others = [(port, program) for port, program in local_ports() if port not in known and (port, program) not in _not_llm]
    servers = [(label, provider) for provider, (label, _) in SERVERS.items()]
    servers += [(f"{program} (port {port})", f"port:{port}") for port, program in others]

    def models(server):
        try:
            return Cleaner("", Glossary(None), provider=server[1]).models(timeout=1)
        except Exception:                          # not running, or not an LLM server: nothing to offer
            return None
    with ThreadPoolExecutor(len(servers)) as pool:  # together: a port that doesn't answer costs its 1 s once
        found = list(pool.map(models, servers))
    _not_llm.update(other for other, names in zip(others, found[len(SERVERS):]) if names is None)
    return [(label, provider, names) for (label, provider), names in zip(servers, found) if names]
