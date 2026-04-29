"""Macroeconomic Environment Research Agent."""
from agents.base_agent import BaseResearchAgent

class MacroAgent(BaseResearchAgent):
    AGENT_NAME = "macro"

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        queries = [
            f"{geography or 'global'} economic outlook GDP inflation consumer confidence",
            f"{sector} economic impact recession growth forecast",
            f"geopolitical risks supply chain {geography or 'global'} 2024 2025",
        ]
        all_results = []
        for q in queries:
            all_results.extend(self.web_search(q, max_results=3))

        context = f"""You are a macroeconomic analyst researching the economic environment for this business idea:
IDEA: {idea}
COMPANY: {company or 'Not specified'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

Analyze:
1. Current GDP growth, inflation, and interest rate trends in target markets
2. Consumer confidence and spending patterns
3. Supply chain conditions and commodity price trends
4. Geopolitical risks that could affect this idea
5. Labor market conditions relevant to this sector
6. Currency and trade policy considerations"""

        return self.summarize_with_llm(all_results, context)
