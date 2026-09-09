"""Guard against running the Windows-only parts of this project from WSL.

The arms appear as Windows COM ports and the cameras as DirectShow devices, so
everything that touches hardware has to run in a Windows Python. That is easy to
get wrong from a WSL shell: `uv run` there is Linux uv, which quietly deletes the
Windows .venv and rebuilds a Linux one that cannot see any of the hardware.
"""

import os
import sys
from pathlib import Path

WINDOWS_PYTHON = Path(".venv/Scripts/python.exe")


def add_venv_scripts_to_path():
    """Put this interpreter's Scripts directory on PATH.

    Running `.venv/Scripts/python.exe` directly, rather than through an activated
    environment, leaves the venv's Scripts directory off PATH. Console entry points
    installed alongside the interpreter are then invisible - Rerun's viewer binary
    among them, so `rr.spawn()` fails with "Failed to find Rerun Viewer executable
    in PATH" even though rerun-sdk is installed.
    """
    scripts = Path(sys.executable).parent
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if str(scripts) not in entries:
        os.environ["PATH"] = os.pathsep.join([str(scripts), *entries])
    return scripts


def require_windows(what="This script"):
    """Exit with instructions if we are not on a Windows Python."""
    if sys.platform == "win32":
        return
    raise SystemExit(
        f"{what} talks to Windows COM ports and DirectShow cameras, but this is "
        f"{sys.platform!r}.\n"
        "\n"
        "You are most likely in a WSL shell, where `uv run` is Linux uv: it\n"
        "replaces the Windows .venv with a Linux one that cannot see the arms or\n"
        "the cameras. Run it one of these ways instead:\n"
        "\n"
        "  PowerShell:  uv run scripts/<name>.py ...\n"
        f"  WSL bash:    ./{WINDOWS_PYTHON.as_posix()} scripts/<name>.py ...\n"
        "\n"
        "If the Windows .venv is already gone, rebuild it from PowerShell with\n"
        "`uv sync`."
    )
