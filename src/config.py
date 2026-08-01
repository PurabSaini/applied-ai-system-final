"""Loads configuration from the project .env file.

No external dependency: we parse .env ourselves (KEY="value" per line).
Exposes the two API keys and the Gemini model name to the rest of the app.
"""
from pathlib import Path

# Project root is the parent of this src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env(path: Path) -> dict:
    """Minimal .env parser: KEY="value" or KEY=value per line."""
    env: dict = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


_ENV = _load_env(ENV_PATH)

# Accept either LAST_FM_API_KEY (preferred) or LAST_FM_API (older name).
LASTFM_API_KEY = _ENV.get("LAST_FM_API_KEY") or _ENV.get("LAST_FM_API", "")
GEMINI_API_KEY = _ENV.get("GEMINI_API_KEY", "")
GEMINI_MODEL = _ENV.get("GEMINI_MODEL", "gemini-flash-latest")


def require(name: str) -> str:
    """Return a required config value or raise a clear error."""
    value = {
        "LAST_FM_API_KEY": LASTFM_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }.get(name, "")
    if not value:
        raise RuntimeError(
            f"Missing {name} in {ENV_PATH}. Copy .env.example to .env and fill it in."
        )
    return value
