#!/bin/zsh
# Installs STT-Clean in this folder:  ./install.sh
#   1. a Python environment in .venv with the app's Python packages (~0.5 GB; no models)
#   2. the app, build/STT-Clean.app, and opens it
# It downloads no model: STT-Clean uses the speech models and LLMs already on this Mac.
# Run it again after updating (git pull): it only adds what is missing and rebuilds the app.
# Use your own Python environment instead of .venv:  STT_CLEAN_PYTHON=/path/to/python ./install.sh
set -e
cd "${0:A:h}"
step() { print -P "\n%B==> $1%b" }
fail() { print -r -- "✗ $1"; exit 1 }

[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || fail "STT-Clean needs a Mac with Apple silicon (M1 or newer)."
xcode-select -p >/dev/null 2>&1 || fail "Install Apple's command line tools first:  xcode-select --install  (then run ./install.sh again)"

PY=$STT_CLEAN_PYTHON
if [[ -z $PY ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    # Python 3.12-3.14: every dependency has a ready-made package for them; for 3.15 some don't yet (PyYAML)
    # shortcut: 3.15 left out; add it once  pip install --dry-run --only-binary=:all: --python-version 3.15 --platform macosx_14_0_arm64 --target /tmp/x -r src/requirements.txt  passes
    for p in python3.14 python3.13 python3.12 python3; do
      if command -v $p >/dev/null && $p -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null; then
        BASE=$p; break
      fi
    done
    [[ -n $BASE ]] || fail "Needs Python 3.12, 3.13 or 3.14 (3.15 is too new for some packages yet): install 3.14 from https://www.python.org/downloads/macos/ (or: brew install python@3.14), then run ./install.sh again."
    step "Creating the Python environment (.venv) with $($BASE --version)"
    $BASE -m venv .venv
  fi
  PY=$PWD/.venv/bin/python
fi

step "Installing the dependencies"
$PY -m pip install --disable-pip-version-check -r src/requirements.txt
$PY -m pip install --disable-pip-version-check --no-deps mlx-whisper==0.4.3   # Whisper engine, without PyTorch

step "Building the app"
STT_CLEAN_PYTHON=$PY zsh macos/build_app.sh
open build/STT-Clean.app

found() { $PY -c "import sys; sys.path.insert(0, 'src'); import $1; sys.exit(not $1.available())" 2>/dev/null }
print -r -- "
✓ STT-Clean is running: its icon is in the menu bar (top right).
  - Allow Accessibility when macOS asks (to type the text), and the Microphone on your first dictation.
  - Right-click the icon: Speech and LLM list the models already on this Mac.
  - Click the icon, speak, click again.
  - Start it at login: System Settings → General → Login Items → + → $PWD/build/STT-Clean.app"
found stt || print -r -- "
! No speech model on this Mac yet. Download one (2.3 GB), then pick it in the menu → Speech:
    ${PY:h}/hf download mlx-community/parakeet-tdt-0.6b-v3
  Other models: see \"Models\" in README.md."
found cleanup || print -r -- "
! No LLM found, so you get the raw transcript. For the clean-up, either
  - install Ollama (https://ollama.com), then:  ollama pull gemma4:12b-mlx   (or use any model in LM Studio), or
  - download an LLM that STT-Clean runs itself (7 GB, no other app needed), then pick it in the menu → LLM:
      ${PY:h}/hf download mlx-community/gemma-4-12B-it-4bit"
