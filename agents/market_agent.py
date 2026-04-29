"""Market & Financial Signals Research Agent."""
from agents.base_agent import BaseResearchAgent

class MarketAgent(BaseResearchAgent):
    AGENT_NAME = "market"

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        queries = [
            f"{sector} market size growth forecast {geography}",
            f"{sector} startup funding venture capital recent",
            f"{company} competitors financial performance stock",
        ]
        all_results = []
        for q in queries:
            all_results.extend(self.web_search(q, max_results=3))

        context = f"""You are a financial analyst researching market signals for this business idea:
IDEA: {idea}
COMPANY: {company or 'Not specified'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

Analyze:
1. Market size and growth trajectory for this sector
2. Recent funding rounds and investor sentiment in this space
3. Competitor financial performance and valuations
4. Revenue models that work in this space
5. Financial risks and opportunities specific to this idea"""

        return self.summarize_with_llm(all_results, context)
