<img src="assets/LOGO.png" width="320" alt="STT-Clean logo">

# STT-Clean: Live Speech-to-Text with LLM Refinement

![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![macOS 14+, Apple silicon](https://img.shields.io/badge/macOS-14%2B%2C%20Apple%20silicon-lightgrey)
![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)

STT-Clean is a macOS menu-bar app that types what you say into any app, at the cursor. While you
speak, a live draft appears as you talk, transcribed on your Mac by Parakeet or Whisper (MLX on
Apple silicon). When you stop, a local LLM refines it: it applies spoken corrections ("scratch
that"), punctuation ("comma", "new line"), your glossary's spellings and translation ("write it
in English"), and the clean text replaces the draft. The LLM runs in Ollama, LM Studio or any
other local server (llama.cpp, mlx-lm, Jan, ...). STT-Clean is a wrapper around models you
already have: it finds them, lists them in its menu, and never downloads one. Nothing leaves your
Mac. On an M4 with Parakeet TDT 0.6B v3, the live draft updates every 0.5 s however long you
speak, the final transcript of a 70 s dictation is ready 0.2 s after you stop, and
`gemma4:12b-mlx` refines a dictation in 0.6–3.3 s.

| You say | Pasted |
|---|---|
| "Send the report to Luca by Friday, actually no, by Thursday." | Send the report to Luca by Thursday. |
| "Caro Marco virgola a capo ti scrivo per confermare la riunione di domani punto" | Caro Marco,<br>ti scrivo per confermare la riunione di domani. |
| "Ho installato o lama e uso il modello emme elle ics per la dettatura." | Ho installato Ollama e uso il modello MLX per la dettatura. *(glossary)* |
| "Domani mando la relazione al professore. Scrivilo in inglese." | I will send the report to the professor tomorrow. |

*Real outputs of `gemma4:12b-mlx` from `tests/bench_llm.py` and `tests/e2e_pipeline.py`.*

## Install

| You need | Check / get it |
|---|---|
| A Mac with Apple silicon (M1 or newer), macOS 14 or newer | Apple menu → About This Mac |
| Apple's command line tools (for `git` and to build the app) | `xcode-select --install` |
| Python 3.12, 3.13 or 3.14 (3.15 is too new for some dependencies yet) | [python.org](https://www.python.org/downloads/macos/) or `brew install python@3.14` |

```bash
git clone https://github.com/riccardoroselli/STT-Clean.git
cd STT-Clean
./install.sh
```

`install.sh` creates a Python environment in `.venv` inside the folder (under 1 GB, no
models), builds `build/STT-Clean.app` and opens it. When it finishes it tells you if a speech model
or an LLM is missing, with the command to get one. To use a Python environment you already have
instead of `.venv`: `STT_CLEAN_PYTHON=/path/to/python ./install.sh`.

The app runs from this folder, so keep it where you cloned it. To update: `git pull`, then
`./install.sh` again.

### First launch

1. **Accessibility:** allow it when macOS asks. STT-Clean needs it to type the text.
2. **Microphone:** allow it on your first dictation.
3. **Speech model:** right-click the icon → **Speech**. If the list is empty, download one (next
   section). The new model shows up the next time you open the menu, no restart needed.
4. **Start at login** (optional): System Settings → General → Login Items → + →
   `build/STT-Clean.app`.

## Models

STT-Clean uses the models already on your Mac and checks for new ones every time you open its menu.

| Kind | Found in | Formats | Picked at launch |
|---|---|---|---|
| Speech | The Hugging Face cache: `~/.cache/huggingface/hub`, or `$HF_HOME/hub` if it was set when you ran `install.sh` | Parakeet and Whisper in MLX format (`mlx-community/…`) | Your last pick, else Parakeet TDT 0.6B v3, else the first one found |
| LLM in a server | Every chat model in [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai) and any other server running on a local port with the OpenAI-style API (`llama-server`, `mlx_lm.server`, Jan, vLLM, ...) | Any instruct model | Your last pick, else `gemma4:12b-mlx`, else the first one found |
| LLM built in | The Hugging Face cache (menu → LLM → **Built in**). STT-Clean runs it itself with [mlx-lm](https://github.com/ml-explore/mlx-lm), no other app needed | Any LLM that mlx-lm supports | Only when you pick it |

Download a speech model with the Hugging Face CLI that `install.sh` put in `.venv`:

```bash
.venv/bin/hf download mlx-community/parakeet-tdt-0.6b-v3      # 2.3 GB, 25 European languages, fastest
.venv/bin/hf download mlx-community/whisper-large-v3-turbo    # 1.6 GB, ~100 languages
```

Whisper models in other formats are not found: whisper.cpp or MacWhisper `.bin` files,
faster-whisper, OpenAI `.pt` files, Transformers `openai/whisper-*`. Whisper pads every clip to
30 s, so with large Whisper models the live draft updates less often than with Parakeet.

The LLM is optional; without one you get the raw transcript. For the refinement, either install
Ollama and run `ollama pull gemma4:12b-mlx` (7.7 GB, ~8 GB of RAM while loaded), download a model in
LM Studio, start any other local server, or download an LLM for the built-in server:

```bash
.venv/bin/hf download mlx-community/gemma-4-12B-it-4bit       # 6.8 GB
```

A built-in model is loaded by a child process of STT-Clean (mlx-lm's server on a private local
port) and leaves memory when that process is stopped: when you switch model, turn clean-up off,
quit, or after 10 idle minutes.

The prompt was tuned on Gemma 4. Results of `tests/bench_llm.py` (21 dictations) on an M4:

| Model (in Ollama) | Passed | Time per dictation |
|---|---|---|
| `gemma4:12b-mlx` | 21/21 | 0.6–4.2 s |
| `gemma4:26b-mlx` | 15/16 ¹ | 0.6–3.5 s |
| `qwen3.8` (27B) | 15/16 ¹ | 12–23 s |
| `SparkLLM/Spark-X2.5-4B` | 14/21 | 1.1–10 s |

¹ Measured on the 16 cases without the styles.

Run it to check another model. A model that answers the text instead of refining it fails the
check run when you pick it, and the previous model stays. A reasoning model that cannot switch
its thinking off (deepseek-r1) works, but each dictation waits for its thoughts.

## Usage

| Action | How |
|---|---|
| Dictate | Click the menu-bar icon (or press your shortcut), speak, click again. The icon's bars move with your voice and a clock shows the recording time. |
| Cancel | Esc while recording: the live text is erased, nothing is pasted. |
| Menu | Right-click the icon. |
| Keyboard shortcut | Menu → **Shortcut** → ⌥Space, ⌃⌥Space, ⌃⌥D or F5 starts and stops a dictation from any app. Off by default. |
| Switch speech model or LLM | Menu → **Speech** or **LLM** → click a model. The one in use is ticked. |
| Style | Menu → **Style**: *Clean* keeps your words, *Formal* rewrites them in a professional tone, *Bullet list* formats them as a list, *In English* translates every dictation. |
| Smart clean-up (LLM) | Menu switch. Off: raw transcript, and the LLM's memory is freed. |
| Live text while speaking | Menu switch. Off: only the final text is typed. |
| Custom spellings | Menu → **Edit Glossary…** (see [Glossary](#glossary)) |
| Fix a problem | When something is missing, the menu shows a ⚠️ item that fixes it: it starts Ollama or LM Studio's server, opens the Accessibility or Microphone setting, or shows how to get a speech model. |

Your choices are remembered. Only one LLM is in memory at a time; it is unloaded when you switch,
turn clean-up off or quit, and after 10 minutes without a dictation. It then loads again while you
record the next one, so only a dictation shorter than the load (about 7–12 s for a 12B model)
waits for it.

Keystrokes only go to the app, window and text field where the dictation started. Switch away and
the typing pauses; come back and it catches up. If you're still away when the dictation ends, the
text goes to the clipboard and the menu says so.

### Voice commands

The LLM follows these in Italian and English:

| Kind | Say | Result |
|---|---|---|
| Self-correction | "scratch that" · "actually no" · "no cancella" · "anzi" | Only the corrected version is kept |
| Punctuation | "comma", "period" · "virgola", "punto", "due punti", "punto interrogativo" | `,` `.` `:` `?` |
| Layout | "new line" / "a capo" · "new paragraph" / "nuovo paragrafo" | Line break / blank line |
| Translation | "… write it in English" · "scrivilo in inglese" · "traduci in inglese" | Whole text translated, instruction removed |
| Partial translation | "solo la prossima frase traducimela in inglese" | Only the next sentence translated |

Questions and requests are pasted as text, not answered.

### Glossary

For names, brands and jargon the speech model gets wrong, open Menu → **Edit Glossary…** and add one
term per line:

```
Ollama: o lama, olama
GitHub: git hub
Kubernetes
```

The variants after the colon are replaced exactly, before the LLM runs. Every term is also given to
the LLM, which fixes close mis-hearings from context. Changes apply from the next dictation.

## How it works

| Step | What happens | Code |
|---|---|---|
| Record | The mic opens on click and closes on stop. Esc is a global hot key only while recording. | `src/menubar.py`, `src/audio.py`, `src/esc_key.py` |
| Live draft | Every ≥0.5 s the audio after the last finished sentence is transcribed again; finished sentences are not redone, and nothing runs while you are silent. Only words that two consecutive drafts agree on are typed, so the text grows instead of flickering. | `src/stt.py`, `src/live.py` |
| Final pass | The unfinished tail is transcribed once more and the glossary's variants are replaced. | `src/stt_clean.py`, `src/cleanup.py` |
| Refinement | The LLM (temperature 0, thinking off) applies the voice commands. On any failure, or an output that looks like an answer, the raw text is used. | `src/cleanup.py` |
| Replace | Only the changed tail of the draft is erased; the rest is pasted with ⌘V. The previous clipboard (text, images, files) is restored once the app has read the paste. | `src/output.py` |

A sentence is finished when two drafts in a row agree on it, another sentence follows it and it
ended 2 s ago. With Parakeet a live update takes 0.2–0.5 s after 10 s of speech and after 3.5
minutes alike (transcribing the whole clip each time took 0.3 s at 10 s, 1.6 s at 60 s and 6.6 s
at 3 minutes). With live text off, a long recording is transcribed 60 s at a time. Parakeet uses
~1.3 GB of RAM; its working cache is freed after each dictation.

`build/STT-Clean.app` is a small compiled launcher (`macos/launcher.c`) that starts the Python app
as a child process, so macOS gives the Microphone and Accessibility permissions to STT-Clean
rather than to Python.

## Privacy

| | |
|---|---|
| **Network** | Only `127.0.0.1`: Ollama (port 11434), LM Studio (port 1234), and one `GET /v1/models` to each other program listening on a local port (1024–49151) to see whether it is an LLM server; one that isn't is not asked again. The Hugging Face library runs offline (`HF_HUB_OFFLINE=1`), so no model is ever downloaded by the app. |
| **Your voice** | Kept in memory during the dictation, never written to disk (unless you run it from a terminal with `--debug`, see [Development](#development)). |
| **Dictated text** | Typed or pasted into your app, shown in the menu's status line, never written to disk. The clipboard is used for the paste, marked as transient so clipboard managers skip it, and then restored. |
| **Files it writes** | Menu choices: `~/Library/Preferences/com.riccardoroselli.stt-clean.plist`. Glossary: `config/glossary.txt`. Log of the current run (status and errors, no dictated text): `logs/stt-clean.log`, overwritten at each launch. |

## Troubleshooting

| Problem | Fix |
|---|---|
| `install.sh`: "Needs Python 3.12, 3.13 or 3.14" | Install Python 3.14 (see [Install](#install)) and run `./install.sh` again. |
| The live text or the final text never appears | Turn STT-Clean on in System Settings → Privacy & Security → Accessibility. The menu's status line warns about this. |
| The app does nothing when opened, and the project is on an external drive | Allow STT-Clean in Privacy & Security → Files & Folders → Removable Volumes. |
| "No speech model found" | Download one ([Models](#models)), then pick it in the menu → Speech. |
| "Ollama isn't running" | Click the ⚠️ **Start Ollama** item in the menu, or turn Smart clean-up off. |
| The microphone gives only silence | Allow STT-Clean in Privacy & Security → Microphone (the menu's ⚠️ item opens it). |
| A shortcut does nothing | Another app holds it (the menu says so): pick another one in Menu → Shortcut. |
| A model "didn't work" | It answered instead of refining, or thought too long. Pick another; `gemma4:12b-mlx` passes all checks. |
| Anything else | Read `logs/stt-clean.log`. |

Every `./install.sh` rebuilds the app, and macOS identifies an unsigned app by its exact contents:
allow Accessibility and the Microphone again after an update.

**Reset or uninstall.** `defaults delete com.riccardoroselli.stt-clean` clears the menu choices.
To uninstall, run it, remove STT-Clean from Login Items if you added it, and delete the folder. Your
models stay where they are.

## Limitations

- The start/stop shortcut is one of four fixed combinations; there is no hold-to-talk.
- Secure input fields (passwords, terminals with Secure Keyboard Entry) block typing and pasting.
- Chrome and Electron apps (VS Code) don't report their focused text field, so switching tabs inside
  one of their windows during a dictation isn't detected.
- A clipboard manager that ignores the transient mark and reads the clipboard before the target app
  counts as the paste, so the old clipboard can come back too early.
- A sentence the live draft has finished is not transcribed again, so its punctuation can differ
  slightly from a whole-clip transcript (measured on 25–70 s clips: same words, a few commas).
- The LLM rewrites the whole text: long dictations take ~1 s per 20–25 words. Turn Smart clean-up off
  for raw speed.
- Voice commands work as phrased in the prompt (`SYSTEM_PROMPT` in `src/cleanup.py`); there is no
  general voice-command engine.

## Project structure

```
install.sh             One-step install: Python packages in .venv, then the app
src/                   The app (Python, ~1600 lines)
  stt_clean.py         Entry point and worker loop: record → live draft → final → refine → paste
  stt.py               Speech engines (Parakeet, Whisper on MLX); finds the models on this Mac
  menubar.py           Menu-bar icon, menu, Speech/LLM lists, switches, saved choices, status line
  audio.py             Mic stream opened per recording, resampling, sounds
  hfcache.py           Lists the models in the Hugging Face cache
  live.py              Live-draft logic: finished sentences, which words to type, minimal erase/append diff
  output.py            Keystrokes, focus check (app/window/field), clipboard save/restore, ⌘V
  hotkey.py            Global hot keys without extra permission: Esc while recording, the start/stop shortcut
  cleanup.py           LLM client (Ollama, LM Studio, any local server), system prompt, styles, glossary, fallbacks
  requirements.txt     Pinned dependencies

macos/                 App bundle: build_app.sh, launcher.c, Info.plist, AppIcon.icns, make_icon.py
tests/                 46 unit tests, LLM benchmark, end-to-end and typing checks
assets/                Logo and 1024 px app icon
```

`.venv/`, `build/`, `logs/` and your `config/glossary.txt` are created locally and not tracked.

## Development

```bash
.venv/bin/python src/stt_clean.py --no-paste    # run from a terminal: log there, text printed, not typed
.venv/bin/python -m unittest discover tests     # 46 unit tests: live-draft logic, glossary, LLM servers, built-in server, engines, hot keys, settings
.venv/bin/python tests/bench_llm.py [model ...] # 21 dictations through the LLM picked in the menu, exit 1 on a failure
.venv/bin/python tests/e2e_pipeline.py          # macOS `say` → the real worker loop (live draft, LLM, style, idle unload), no mic
.venv/bin/python tests/typing_check.py          # replays heavy live-draft churn into TextEdit (takes the keyboard ~15 s)
```

`--debug` keeps the last dictation's audio and live drafts in `logs/debug/` to replay it offline.
After changing `SYSTEM_PROMPT` in `src/cleanup.py`, rerun `tests/bench_llm.py`.

## Licence

[MIT](LICENSE). The models are not part of this repo and keep their own licences, e.g.
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NVIDIA, CC-BY-4.0),
[Whisper](https://github.com/openai/whisper) (OpenAI, MIT) and
[Gemma](https://ai.google.dev/gemma/terms) (Google).
