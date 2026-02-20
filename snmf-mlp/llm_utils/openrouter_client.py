import os
from typing import Optional

from openai import AsyncOpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_openrouter_client(
    api_key: str,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> AsyncOpenAI:
    """
    Build an async OpenRouter client using OpenAI-compatible SDK surface.

    Optional headers improve OpenRouter analytics/routing:
    - HTTP-Referer: your app/site URL
    - X-Title: your app name
    """
    referer = site_url or os.getenv("OPENROUTER_SITE_URL")
    title = app_name or os.getenv("OPENROUTER_APP_NAME")

    default_headers = {}
    if referer:
        default_headers["HTTP-Referer"] = referer
    if title:
        default_headers["X-Title"] = title

    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers if default_headers else None,
    )
