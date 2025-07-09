import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("user_config.json")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def update_config(section: str, **kwargs) -> None:
    cfg = load_config()
    sec = cfg.get(section, {})
    sec.update({k: v for k, v in kwargs.items() if v is not None})
    cfg[section] = sec
    save_config(cfg)
