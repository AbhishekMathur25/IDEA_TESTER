"""
Idea Tester — Base Research Agent
Common functionality for all 5 specialized research agents.
Supports Tavily web search + Gemini's Google Search grounding.
"""

import logging
from typing import Optional

from engine.llm_client import LLMClient

logger = logging.getLogger("idea_tester.agents")


class BaseResearchAgent:
    """Base class for research agents with web search + LLM summarization."""

    AGENT_NAME = "base"

    def __init__(self, llm: Optional[LLMClient] = None, tavily_api_key: str = ""):
        self.llm = llm or LLMClient()
        self.tavily_api_key = tavily_api_key
        self._tavily = None

    @property
    def tavily(self):
        """Lazy-init Tavily client."""
        if self._tavily is None and self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self._tavily = TavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                logger.warning("tavily-python not installed — web search disabled")
        return self._tavily

    def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web using Tavily. Returns list of {title, url, content}."""
        if not self.tavily:
            logger.warning(f"[{self.AGENT_NAME}] No Tavily key — using LLM knowledge only")
            return []

        try:
            response = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
            )
            results = response.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Web search failed: {e}")
            return []

    def summarize_with_llm(self, search_results: list[dict], context_prompt: str) -> str:
        """Use LLM to summarize search results into structured insights."""
        if search_results:
            results_text = "\n\n".join(
                f"**{r['title']}** ({r['url']})\n{r['content']}"
                for r in search_results
            )
            prompt = f"""{context_prompt}

Based on these web search results, provide a structured analysis:

{results_text}

Provide specific, factual insights. Cite sources where possible. 
If the search results don't contain relevant information, say so and provide analysis based on your knowledge."""
        else:
            # No Tavily results — use Gemini's Google Search grounding
            prompt = f"""{context_prompt}

Search the web for the latest information and provide a thorough, data-driven analysis.
Include specific facts, figures, and recent developments.
Be clear about what is established fact vs. reasonable inference.
Provide specific, actionable insights."""

            # Try grounded search first (Gemini + Google Search)
            try:
                return self.llm.grounded_chat(prompt, temperature=0.4, max_tokens=2500)
            except Exception as e:
                logger.warning(f"[{self.AGENT_NAME}] Grounded search fallback: {e}")

        return self.llm.chat(prompt, temperature=0.4, max_tokens=2500)

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        """Override in subclasses."""
        raise NotImplementedError
