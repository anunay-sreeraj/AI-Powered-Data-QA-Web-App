"""OpenAI-compatible chat client, defaulting to the OpenAI API.

Swappable to Groq / Ollama / OpenRouter / any other OpenAI-compatible endpoint
via .env alone (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL) - no code changes needed.
See .env.example for the open-weight-model alternatives (Groq, Ollama) - the
assignment brief asks for an open-source model as the core engine; OpenAI is
used here as a deliberate, documented tradeoff (see WRITEUP.md) after hitting
rate limits on Groq's free tier during development.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5-mini"


class LLMNotConfiguredError(RuntimeError):
    pass


def _get_config(key: str, default: str | None = None) -> str | None:
    """Reads from st.secrets first, then falls back to a plain env var (.env
    via python-dotenv). Accessing st.secrets raises when no secrets.toml
    exists at all (e.g. every local dev run that only uses .env), so this
    must stay defensive - the two config sources are not interchangeable:
    .env populates os.environ automatically, but Streamlit Community Cloud's
    Secrets UI populates st.secrets, not os.environ, unless read this way.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def is_configured() -> bool:
    return bool(_get_config("LLM_API_KEY"))


def get_model_name() -> str:
    return _get_config("LLM_MODEL", DEFAULT_MODEL)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = _get_config("LLM_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add an API key "
                "(see .env.example for OpenAI / Groq / Ollama options)."
            )
        base_url = _get_config("LLM_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()
