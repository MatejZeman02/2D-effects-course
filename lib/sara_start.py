"""How `sara.connect` starts a Sara that outlives the notebook kernel which started it.

A kernel restart and VS Code closing end the kernel by its process tree, walking children by their
parent's id as `taskkill /T` does, or by a Windows job that kills on close, and a Sara that is still the
kernel's child dies with it, unsaved work and all. So Sara is started by a launcher that exits at once,
and its parent is gone before any kernel ends. It rides beside `sara.py` wherever that is copied.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# The process ids of the Saras `launch` started, none of them a child of this Python.
STARTED: list[int] = []
# What starts Sara for `launch` and exits at once, printing Sara's id, so Sara's parent is gone before a tool
# that ends a kernel by its process tree reaches it. A Windows job that forbids breaking away refuses the flag.
LAUNCHER = """import json, subprocess, sys
keys = json.loads(sys.argv[1])
quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    child = subprocess.Popen(sys.argv[2:], **quiet, **keys)
except PermissionError:
    keys["creationflags"] &= ~0x01000000
    child = subprocess.Popen(sys.argv[2:], **quiet, **keys)
print(child.pid)
"""


def launch(command: list[str]) -> int:
    """Starts *command* through `LAUNCHER` and answers its process id, kept in `STARTED`."""
    windows = sys.platform.startswith("win")
    # No console window for the launcher on Windows, `CREATE_NO_WINDOW`.
    ran = subprocess.run([sys.executable, "-I", "-c", LAUNCHER, json.dumps(detached()), *command], capture_output=True, encoding="utf-8", timeout=30, creationflags=0x08000000 if windows else 0)
    if ran.returncode != 0:
        raise RuntimeError((ran.stderr.strip().splitlines() or [f"the launcher ended with {ran.returncode}"])[-1])
    STARTED.append(int(ran.stdout.split()[-1]))
    return STARTED[-1]


def detached(platform: str | None = None) -> dict:
    """What `Popen` takes for a child that outlives this Python: a session of its own, or on Windows `DETACHED_PROCESS`,
    `CREATE_NEW_PROCESS_GROUP` and `CREATE_BREAKAWAY_FROM_JOB`, by value as `subprocess` names them on Windows only."""
    if (platform or sys.platform).startswith("win"):
        return {"creationflags": 0x00000008 | 0x00000200 | 0x01000000}
    return {"start_new_session": True}


def alive(pid: int) -> bool:
    """Whether process *pid* still runs, a Linux zombie counting as ended. Windows waits on it for no time."""
    if sys.platform.startswith("win"):
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE, and 0x102 below is WAIT_TIMEOUT
        running = bool(handle) and kernel.WaitForSingleObject(handle, 0) == 0x102
        if handle:
            kernel.CloseHandle(handle)
        return running
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] != "Z" if sys.platform.startswith("linux") else os.kill(pid, 0) is None
    except (OSError, IndexError):
        return False
