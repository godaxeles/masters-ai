import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class Settings:
    openai_api_key: str
    openai_model: str


def _clean(val: Optional[str], default="") -> str:
    return val.strip() if val else default


def load_settings() -> Settings:
    api = _clean(os.getenv("OPENAI_API_KEY"), "")
    model = _clean(os.getenv("OPENAI_MODEL"), "gpt-4o-mini")

    api_file = os.getenv("OPENAI_API_KEY_FILE")
    if not api and api_file and os.path.exists(api_file):
        try:
            with open(api_file, "r", encoding="utf-8") as f:
                api = _clean(f.read())
        except (OSError, IOError):
            pass

    return Settings(openai_api_key=api, openai_model=model)
