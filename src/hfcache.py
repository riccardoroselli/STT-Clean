"""hfcache.py - the models already downloaded to this Mac's Hugging Face cache (never downloads)."""
import glob
import json
import os


def configs():
    """(repo, folder, config.json as a dict) for each downloaded model, e.g.
    ("mlx-community/parakeet-tdt-0.6b-v3", ".../snapshots/<hash>", {...}). ~40 ms for a dozen models."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    seen = set()
    for path in sorted(glob.glob(os.path.join(hub, "models--*", "snapshots", "*", "config.json"))):
        repo = path.split(os.sep)[-4][len("models--"):].replace("--", "/", 1)
        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, ValueError):
            continue
        if repo not in seen and isinstance(config, dict):   # one snapshot per model
            seen.add(repo)
            yield repo, os.path.dirname(path), config
