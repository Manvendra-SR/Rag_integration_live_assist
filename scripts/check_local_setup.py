from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def status(label: str, ok: bool, detail: str = "") -> None:
    state = "ok" if ok else "missing"
    suffix = f" - {detail}" if detail else ""
    print(f"{label}: {state}{suffix}")


def env_value(key: str) -> str:
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return os.environ.get(key, "")

    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        current_key, value = line.split("=", 1)
        if current_key.strip() == key:
            return value.strip()
    return os.environ.get(key, "")


def key_is_set(key: str) -> bool:
    value = env_value(key)
    return bool(value and value.lower() not in {"replace_me", "none", "null"})


def main() -> None:
    print("Live Assist local setup check")
    print(f"project={PROJECT_DIR}")

    status("python", bool(shutil.which("python3") or shutil.which("python")))
    status("node", bool(shutil.which("node")))
    status("npm", bool(shutil.which("npm")))
    status("swift", bool(shutil.which("swift")))

    status(".env", (PROJECT_DIR / ".env").exists())
    status("backend venv", (PROJECT_DIR / "backend" / ".venv311").exists() or (PROJECT_DIR / "backend" / ".venv").exists())
    status("electron node_modules", (PROJECT_DIR / "frontends" / "desktop-electron" / "node_modules").exists())
    status("native swift build", (PROJECT_DIR / "native-audio-capture" / "macos" / ".build").exists())
    status("chroma index directory", (PROJECT_DIR / "backend" / "chroma_db").exists() or (PROJECT_DIR / "chroma_db").exists())

    for key in ("PYTHON_WS_SARVAM_API_KEY", "GROQ_API_KEY"):
        status(key, key_is_set(key), "set" if key_is_set(key) else "missing_or_placeholder")

    input_source = env_value("INPUT_SOURCE") or "not_set"
    audio_mode = env_value("DESKTOP_AUDIO_CAPTURE_MODE") or "not_set"
    print(f"INPUT_SOURCE={input_source}")
    print(f"DESKTOP_AUDIO_CAPTURE_MODE={audio_mode}")
    print("macOS permissions: manually confirm Microphone and Screen & System Audio Recording for Terminal/Electron host")


if __name__ == "__main__":
    main()
