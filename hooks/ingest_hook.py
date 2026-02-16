"""UserPromptSubmit hook: auto-ingest session logs on first message.

Place this in your Claude Code hooks chain (settings.json) to automatically
run memory_ingest in the background when a new session starts.

Time-based dedup: skips if last ingest was < 5 minutes ago.
Launches ingest.py in background (non-blocking).
No output — the agent doesn't need to know.

Paths are resolved relative to this file's location in the repo.
Venv is auto-detected (.venv/ or venv-win/).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INGEST_SCRIPT = REPO_ROOT / "ingest.py"
LAST_INGEST = Path.home() / ".claude" / ".last-ingest"
MIN_INTERVAL = 300  # seconds


def find_venv_python() -> Path | None:
    """Auto-detect venv python executable."""
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python3",
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / "venv-win" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def should_ingest() -> bool:
    if not LAST_INGEST.exists():
        return True
    try:
        timestamp = float(LAST_INGEST.read_text().strip())
        return (time.time() - timestamp) > MIN_INTERVAL
    except (ValueError, OSError):
        return True


def main():
    raw = sys.stdin.read()
    try:
        json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return

    if not should_ingest():
        return

    if not INGEST_SCRIPT.exists():
        return

    venv_python = find_venv_python()
    if venv_python is None:
        return

    LAST_INGEST.write_text(str(time.time()))

    subprocess.Popen(
        [str(venv_python), "-X", "utf8", str(INGEST_SCRIPT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


if __name__ == "__main__":
    main()
