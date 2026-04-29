"""
Idea Tester — Seed Document Packager
Transforms raw research from 5 agents into a structured seed document
that the simulation engine can consume.
"""

import logging
from typing import Optional

from engine.llm_client import LLMClient

logger = logging.getLogger("idea_tester.seed_packager")


class SeedPackager:
    """Converts raw research into a structured seed document for the simulation engine."""

    # Mode configurations
    MODES = {
        "quick": {"entities": "10-15", "relationships": "15-25", "max_tokens": 3000},
        "deep":  {"entities": "30-40", "relationships": "50-60", "max_tokens": 5000},
    }

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def package(
        self,
        idea: str,
        company: str,
        sector: str,
        geography: str,
        research_text: str,
        research_mode: str = "quick",
    ) -> dict:
        """
        Transform research into a structured seed document.

        Args:
            research_mode: 'quick' (10-15 entities) or 'deep' (30-40 entities)

        Returns:
            {"entities": [...], "relationships": [...], "world_state": "...", "event_description": "..."}
        """
        mode = self.MODES.get(research_mode, self.MODES["quick"])
        entity_range = mode["entities"]
        rel_range = mode["relationships"]
        max_tokens = mode["max_tokens"]

        logger.info(f"Packaging seed [{research_mode}]: {entity_range} entities, {rel_range} relationships")

        prompt = f"""You are building a simulation world from real-world research data.
Convert the following research into a structured seed document that describes a market ecosystem.

CEO'S IDEA (the "event" to simulate): "{idea}"
COMPANY: {company or 'Unnamed Company'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

RESEARCH DATA:
{research_text[:8000]}

Create a JSON seed document with:
1. "entities" — The key actors in this market (generate between {entity_range} distinct entities). For EACH entity include:
   - "name": A specific realistic name (not generic like "Consumer A")
   - "type": One of: Company, Consumer, Regulator, Competitor, Investor, Employee, Media, Technology, Market, Organization
   - "description": What they do, their market position, and their relevance to the idea
   - "attributes": {{"influence": "high/medium/low", "financial_position": "strong/moderate/weak"}}

2. "relationships" — How entities are connected (generate between {rel_range} relationships):
   - "source": entity name
   - "target": entity name
   - "type": relationship type (e.g., "competes_with", "regulates", "supplies_to", "invests_in", "adversely_affected_by")
   - "fact": A specific fact about this relationship

3. "world_state" — A 3-4 paragraph narrative describing the current state of this market

4. "event_description" — How to frame the CEO's idea as a market event/announcement

Return a valid JSON object with these 4 keys. Be specific and realistic.
Use real company names, real market data, and real regulatory bodies where possible.
Make each entity distinct with unique motivations."""

        try:
            seed = self.llm.strong_chat_json(prompt, temperature=0.6, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Seed packaging failed: {e}")
            seed = self._fallback_seed(idea, company, sector)

        # Validate structure
        seed.setdefault("entities", [])
        seed.setdefault("relationships", [])
        seed.setdefault("world_state", f"Market state for {sector} in {geography}")
        seed.setdefault("event_description", idea)

        logger.info(
            f"Seed packaged [{research_mode}]: {len(seed['entities'])} entities, "
            f"{len(seed['relationships'])} relationships"
        )
        return seed

    @staticmethod
    def _fallback_seed(idea: str, company: str, sector: str) -> dict:
        """Minimal fallback seed if LLM fails."""
        return {
            "entities": [
                {"name": company or "The Company", "type": "Company", "description": f"Company testing the idea in {sector}"},
                {"name": "Market Regulator", "type": "Regulator", "description": f"Regulatory body overseeing {sector}"},
                {"name": "Key Competitor", "type": "Competitor", "description": f"Major competitor in {sector}"},
                {"name": "Target Customer", "type": "Consumer", "description": "Primary target customer segment"},
                {"name": "Industry Investor", "type": "Investor", "description": "Institutional investor in the space"},
            ],
            "relationships": [
                {"source": company or "The Company", "target": "Key Competitor", "type": "competes_with", "fact": "Direct competitors in the market"},
                {"source": "Market Regulator", "target": company or "The Company", "type": "regulates", "fact": "Regulatory oversight"},
                {"source": "Target Customer", "target": company or "The Company", "type": "buys_from", "fact": "Primary customer relationship"},
            ],
            "world_state": f"The {sector} market is in a state of flux.",
            "event_description": idea,
        }
