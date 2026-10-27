"""The LLM behind every agent: local Ollama or the remote Gemma 4 endpoint, through one code path."""

from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from bench.config import OUTPUT_TOKENS, settings

# Ollama's /v1 rejects forced tool calls and never shows the JSON schema to the model, so every Agent
# is built with NativeOutput (native JSON-schema output) and this profile writes the schema into the instructions.
PROFILE = OpenAIModelProfile(
    native_output_requires_schema_in_instructions=True,
    openai_chat_supports_multiple_system_messages=False,
    openai_chat_supports_max_completion_tokens=False,
)
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep CLI output to our own lines


def model() -> OpenAIChatModel:
    s = settings()
    model_settings = OpenAIChatModelSettings(temperature=0.0, seed=7, max_tokens=OUTPUT_TOKENS)
    if s.reasoning:
        model_settings["openai_reasoning_effort"] = s.reasoning
    provider = OpenAIProvider(base_url=s.llm_url, api_key=s.llm_key)
    return OpenAIChatModel(s.llm_model, provider=provider, profile=PROFILE, settings=model_settings)


def check_usage(input_tokens: int, sent_text: str, label: str) -> None:
    """Warn when the model read far less than we sent: a sign the prompt was truncated."""
    if input_tokens and input_tokens < 0.5 * len(sent_text) / 4.5:
        print(f"  ! {label}: model read {input_tokens} tokens for {len(sent_text)} chars; was the prompt truncated?")
