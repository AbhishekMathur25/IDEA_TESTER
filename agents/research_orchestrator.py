"""
Idea Tester — Research Orchestrator
Runs 5 specialized research agents in parallel and merges their outputs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable

from engine.llm_client import LLMClient
from agents.news_agent import NewsAgent
from agents.market_agent import MarketAgent
from agents.legal_agent import LegalAgent
from agents.competitor_agent import CompetitorAgent
from agents.macro_agent import MacroAgent

logger = logging.getLogger("idea_tester.research")


import threading
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = None
    get_script_run_ctx = None


@dataclass
class ResearchOutput:
    """Combined output from all research agents."""
    news: str = ""
    market: str = ""
    legal: str = ""
    competitors: str = ""
    macro: str = ""
    errors: list[str] = field(default_factory=list)

    def to_full_text(self) -> str:
        """Combine all research into a single document."""
        sections = []
        if self.news:
            sections.append(f"## NEWS & PUBLIC SENTIMENT\n{self.news}")
        if self.market:
            sections.append(f"## MARKET & FINANCIAL SIGNALS\n{self.market}")
        if self.legal:
            sections.append(f"## LEGAL & REGULATORY LANDSCAPE\n{self.legal}")
        if self.competitors:
            sections.append(f"## COMPETITOR INTELLIGENCE\n{self.competitors}")
        if self.macro:
            sections.append(f"## MACROECONOMIC ENVIRONMENT\n{self.macro}")
        return "\n\n---\n\n".join(sections)


class ResearchOrchestrator:
    """Orchestrates parallel research across 5 specialized agents."""

    def __init__(self, llm: Optional[LLMClient] = None, tavily_api_key: str = ""):
        self.llm = llm or LLMClient()
        self.tavily_api_key = tavily_api_key

        self.agents = {
            "news": NewsAgent(llm=self.llm, tavily_api_key=self.tavily_api_key),
            "market": MarketAgent(llm=self.llm, tavily_api_key=self.tavily_api_key),
            "legal": LegalAgent(llm=self.llm, tavily_api_key=self.tavily_api_key),
            "competitors": CompetitorAgent(llm=self.llm, tavily_api_key=self.tavily_api_key),
            "macro": MacroAgent(llm=self.llm, tavily_api_key=self.tavily_api_key),
        }

    def research(
        self,
        idea: str,
        company: str = "",
        sector: str = "",
        geography: str = "",
        progress_callback: Optional[Callable] = None,
    ) -> ResearchOutput:
        """
        Run all 5 research agents in parallel.

        Args:
            idea: The CEO's idea
            company: Company name
            sector: Industry sector
            geography: Target geography
            progress_callback: Optional (agent_name, status) callback
        """
        logger.info(f"Starting parallel research for: {idea[:80]}...")
        output = ResearchOutput()
        total = len(self.agents)
        completed = 0

        # Ollama can only handle 1-2 concurrent requests
        max_workers = 2 if self.llm.provider == "ollama" else 5
        
        ctx = get_script_run_ctx() if get_script_run_ctx else None

        def _run_agent(name: str):
            if add_script_run_ctx and ctx:
                add_script_run_ctx(threading.current_thread(), ctx)
                
            agent = self.agents[name]
            if progress_callback:
                progress_callback(name, "running")
            return name, agent.research(
                idea=idea, company=company, sector=sector, geography=geography
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_agent, name): name for name in self.agents}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    agent_name, result = future.result()
                    setattr(output, agent_name, result)
                    completed += 1
                    if progress_callback:
                        progress_callback(agent_name, "done")
                    logger.info(f"Research agent '{agent_name}' complete ({completed}/{total})")
                except Exception as e:
                    output.errors.append(f"{name}: {str(e)}")
                    completed += 1
                    if progress_callback:
                        progress_callback(name, f"error: {e}")
                    logger.error(f"Research agent '{name}' failed: {e}")

        logger.info(f"Research complete: {completed}/{total} agents succeeded")
        return output
