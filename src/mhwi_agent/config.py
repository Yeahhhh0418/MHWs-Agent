import yaml
from pathlib import Path
from functools import lru_cache

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)