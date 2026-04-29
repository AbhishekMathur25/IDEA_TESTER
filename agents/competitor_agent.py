"""Competitor Intelligence Research Agent."""
from agents.base_agent import BaseResearchAgent

class CompetitorAgent(BaseResearchAgent):
    AGENT_NAME = "competitors"

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        queries = [
            f"{sector} competitors top companies {geography}",
            f"{sector} recent launches pivots acquisitions",
            f"{sector} competitive landscape analysis",
        ]
        all_results = []
        for q in queries:
            all_results.extend(self.web_search(q, max_results=3))

        context = f"""You are a competitive intelligence analyst researching competitors for this business idea:
IDEA: {idea}
COMPANY: {company or 'Not specified'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

Analyze:
1. Key competitors and their current market positions
2. Recent competitor moves: launches, pivots, acquisitions, partnerships
3. Competitor job postings that signal strategic direction
4. Patent filings and R&D focus areas
5. Competitive advantages and vulnerabilities of major players
6. How competitors might react to this specific idea"""

        return self.summarize_with_llm(all_results, context)
