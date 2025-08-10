
import os
from typing import Optional, Tuple
from config.settings import load_settings

def get_openai_client() -> Tuple[Optional[object], str]:
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_PROXY", "OPENAI_HTTP_PROXY"):
        os.environ.pop(v, None)

    settings = load_settings()
    if not settings.openai_api_key:
        return None, settings.openai_model
    try:
        import httpx
        from openai import OpenAI
        try:
            http_client = httpx.Client(transport=httpx.HTTPTransport(proxy=None))
        except TypeError:
            http_client = httpx.Client(proxies=None)
        client = OpenAI(api_key=settings.openai_api_key, http_client=http_client)
        return client, settings.openai_model
    except Exception:
        return None, settings.openai_model
