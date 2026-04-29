"""Legal & Regulatory Research Agent."""
from agents.base_agent import BaseResearchAgent

class LegalAgent(BaseResearchAgent):
    AGENT_NAME = "legal"

    def research(self, idea: str, company: str = "", sector: str = "", geography: str = "") -> str:
        queries = [
            f"{sector} regulation compliance requirements {geography}",
            f"{sector} new laws policy changes 2024 2025",
            f"{sector} regulatory risks antitrust {geography}",
        ]
        all_results = []
        for q in queries:
            all_results.extend(self.web_search(q, max_results=3))

        context = f"""You are a regulatory compliance analyst researching the legal landscape for this business idea:
IDEA: {idea}
COMPANY: {company or 'Not specified'}
SECTOR: {sector or 'Not specified'}
GEOGRAPHY: {geography or 'Global'}

Analyze:
1. Current regulatory framework applicable to this idea
2. Recent or pending regulation changes that could impact execution
3. Compliance requirements and licensing needs
4. Antitrust, monopoly, or competition law risks
5. Data privacy, consumer protection, or industry-specific regulations
6. Regulatory differences across target geographies"""

        return self.summarize_with_llm(all_results, context)
