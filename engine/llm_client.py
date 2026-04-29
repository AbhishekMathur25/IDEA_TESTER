"""
Idea Tester — Unified LLM Client (Gemini + Groq)
Supports Google Gemini and Groq (Qwen, LLaMA, etc.) with a switchable backend.
"""

import json
import re
import time
import logging
from typing import Optional

from config import settings

logger = logging.getLogger("idea_tester.llm")


class LLMClient:
    """Multi-provider LLM client supporting Gemini, Groq, and Ollama."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            provider: "gemini", "groq", or "ollama". Auto-detected from available keys if None.
            model: Override model name. If None, uses config defaults.
        """
        # Auto-detect provider
        if provider:
            self.provider = provider.lower()
        elif settings.GEMINI_API_KEY:
            self.provider = "gemini"
        elif settings.GROQ_API_KEY:
            self.provider = "groq"
        else:
            self.provider = "ollama"

        # Set model
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = settings.GEMINI_MODEL_NAME
        elif self.provider == "ollama":
            self.model = settings.OLLAMA_MODEL_NAME
        else:
            self.model = settings.GROQ_MODEL_NAME

        # Initialize client
        if self.provider == "gemini":
            from google import genai
            if settings.GEMINI_API_KEY:
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
            else:
                raise ValueError("GEMINI_API_KEY is not set")
            self.thinking_level = settings.GEMINI_THINKING_LEVEL
        elif self.provider == "groq":
            from groq import Groq
            if settings.GROQ_API_KEY:
                self._client = Groq(api_key=settings.GROQ_API_KEY)
            else:
                raise ValueError("GROQ_API_KEY is not set")
        elif self.provider == "ollama":
            from ollama import Client as OllamaClient
            self._client = OllamaClient(host=settings.OLLAMA_HOST)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        logger.info(f"LLMClient initialized: provider={self.provider}, model={self.model}")

    # ── Core call ────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        retries: int = 5,
        use_search: bool = False,
        use_thinking: bool = False,
    ) -> str:
        """Send a prompt and return the text response."""
        model = model or self.model

        if self.provider == "gemini":
            return self._chat_gemini(prompt, system, model, temperature, max_tokens, retries, use_search, use_thinking)
        elif self.provider == "groq":
            return self._chat_groq(prompt, system, model, temperature, max_tokens, retries)
        else:
            return self._chat_ollama(prompt, system, model, temperature, retries)

    # ── Gemini Backend ───────────────────────────────────

    def _chat_gemini(self, prompt, system, model, temperature, max_tokens, retries, use_search, use_thinking):
        from google import genai
        from google.genai import types
        last_err = None

        for attempt in range(retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]

                config_kwargs = {}
                if use_thinking:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.thinking_level,
                    )

                tools = []
                if use_search:
                    tools.append(types.Tool(google_search=types.GoogleSearch()))

                config_kwargs["system_instruction"] = system

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    tools=tools if tools else None,
                    **config_kwargs,
                )

                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                return self._extract_text_gemini(response)

            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    wait = 35 + (attempt * 10)
                else:
                    wait = 2 ** (attempt + 1)
                logger.warning(f"Gemini failed (attempt {attempt+1}/{retries}): {err_str[:150]} — retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_err}")

    @staticmethod
    def _extract_text_gemini(response) -> str:
        """Extract text from Gemini response, skipping thinking parts."""
        parts = []
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                    continue
                if hasattr(part, 'text') and part.text:
                    parts.append(part.text)
        if parts:
            return "\n".join(parts).strip()

        if hasattr(response, 'text') and response.text:
            return response.text.strip()

        return ""

    # ── Groq Backend ─────────────────────────────────────

    def _chat_groq(self, prompt, system, model, temperature, max_tokens, retries, use_tools=False):
        last_err = None

        for attempt in range(retries):
            try:
                # Build request kwargs
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "top_p": 0.95,
                    "stream": False,
                    "stop": None,
                }

                # Enable compound tools for groq/compound model
                is_compound = "compound" in model.lower()
                if is_compound or use_tools:
                    kwargs["compound_custom"] = {
                        "tools": {
                            "enabled_tools": ["web_search", "code_interpreter", "visit_website"]
                        }
                    }

                completion = self._client.chat.completions.create(**kwargs)

                text = completion.choices[0].message.content or ""

                # Strip <think>...</think> tags from reasoning models
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

                return text

            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'rate_limit' in err_str.lower():
                    # Use exponential backoff with random jitter to prevent thread collisions
                    import random
                    wait = (attempt + 1) * 15 + random.uniform(1, 10)
                else:
                    wait = 2 ** (attempt + 1)
                logger.warning(f"Groq failed (attempt {attempt+1}/{retries}): {err_str[:150]} — retrying in {wait:.1f}s")
                time.sleep(wait)

        raise RuntimeError(f"Groq call failed after {retries} attempts: {last_err}")

    # ── JSON parsing ─────────────────────────────────────

    def chat_json(self, prompt: str, **kwargs) -> dict:
        """Send a prompt and parse the response as JSON."""
        full_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Return ONLY valid JSON. No markdown code fences, no explanation outside JSON."
        )
        raw = self.chat(full_prompt, **kwargs)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse JSON from LLM response text."""
        text = text.strip()

        # Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON response: {text[:200]}")
        raise ValueError(f"LLM did not return valid JSON: {text[:200]}")

    # ── Convenience methods ──────────────────────────────

    def strong_chat(self, prompt: str, system: str = "You are a helpful assistant.", **kwargs) -> str:
        """Use thinking mode (Gemini) or compound tools (Groq) for complex reasoning."""
        if self.provider == "gemini":
            return self.chat(prompt=prompt, system=system, use_thinking=True, **kwargs)
        else:
            return self.chat(prompt=prompt, system=system, **kwargs)

    def strong_chat_json(self, prompt: str, **kwargs) -> dict:
        """Strong reasoning with JSON output."""
        if self.provider == "gemini":
            return self.chat_json(prompt=prompt, use_thinking=True, **kwargs)
        else:
            return self.chat_json(prompt=prompt, **kwargs)

    def grounded_chat(self, prompt: str, **kwargs) -> str:
        """Chat with web search grounding (Gemini Search / Groq compound tools)."""
        if self.provider == "gemini":
            return self.chat(prompt=prompt, use_search=True, **kwargs)
        else:
            # Groq compound already has web_search enabled
            return self.chat(prompt=prompt, **kwargs)

    # ── Ollama Backend ───────────────────────────────────

    def _chat_ollama(self, prompt, system, model, temperature, retries):
        last_err = None

        for attempt in range(retries):
            try:
                response = self._client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "temperature": temperature,
                    }
                )

                text = response.get("message", {}).get("content", "")

                # Strip <think>...</think> tags which Qwen reasoning models use
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

                return text

            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'too many' in err_str.lower():
                    wait = 20 + (attempt * 15)  # 429: longer backoff (20s, 35s, 50s...)
                else:
                    wait = 2 ** (attempt + 1)
                logger.warning(f"Ollama failed (attempt {attempt+1}/{retries}): {err_str[:150]} — retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Ollama call failed after {retries} attempts: {last_err}")

