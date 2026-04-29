"""News & Public Sentiment Research Agent."""
from agents.base_agent import BaseResearchAgent

class NewsAgent(BaseResearchAgent):
    AGENT_NAME = "news"

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        queries = [
            f"{company} {sector} latest news {geography}".strip(),
            f"{sector} industry trends market sentiment 2024 2025",
            f"{idea[:60]} news analysis",
        ]
        all_results = []
        for q in queries:
            all_results.extend(self.web_search(q, max_results=3))

        context = f"""You are a news analyst researching public sentiment around this business idea:
IDEA: {idea}
COMPANY: {company or 'Not specified'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

Analyze:
1. Current public narrative and media coverage of this sector
2. Market sentiment (bullish/bearish/mixed)
3. Recent controversies or positive developments
4. Public perception trends that could affect this idea
5. Key opinion leaders and their stances"""

        return self.summarize_with_llm(all_results, context)
