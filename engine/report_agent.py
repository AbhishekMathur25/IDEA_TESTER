"""
Idea Tester — Report Agent
Uses ReACT-style loop with Zep tools to analyze simulation results.
Port of MiroFish's report_agent.py pattern.
"""

import json
import logging
import re
from typing import Optional

from engine.llm_client import LLMClient
from engine.knowledge_graph import KnowledgeGraph
from engine.simulation_runner import RoundSummary
from engine.persona_generator import AgentPersona

logger = logging.getLogger("idea_tester.report_agent")


class ReportAgent:
    """
    Analyzes simulation results and generates a multi-section prediction report.
    Uses Zep tools (InsightForge, PanoramaSearch) for evidence retrieval.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def generate_report(
        self,
        graph: KnowledgeGraph,
        personas: list[AgentPersona],
        rounds: list[RoundSummary],
        idea_description: str,
        progress_callback=None,
    ) -> dict:
        """Generate a comprehensive prediction report from simulation data."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Generating prediction report...")

        # Step 1: Compile simulation data
        if progress_callback:
            progress_callback("compiling", 10, "Compiling simulation data")
        sim_data = self._compile_simulation_data(graph, personas, rounds)

        # Step 2: Zep insights
        if progress_callback:
            progress_callback("analyzing", 30, "Analyzing patterns with graph retrieval")
        zep_insights = self._gather_zep_insights(graph, idea_description)

        # Step 3+4: Generate report + risk matrix IN PARALLEL (Optimization #5)
        if progress_callback:
            progress_callback("generating", 50, "Generating report & risk matrix in parallel")

        with ThreadPoolExecutor(max_workers=2) as executor:
            report_future = executor.submit(
                self._generate_full_report, sim_data, idea_description, zep_insights
            )
            risk_future = executor.submit(
                self._generate_risk_matrix, sim_data, idea_description, zep_insights
            )
            report = report_future.result()
            risk_matrix = risk_future.result()

        report["risk_matrix"] = risk_matrix

        # Step 5: Recommendation (depends on report + risk)
        if progress_callback:
            progress_callback("recommendation", 85, "Forming recommendation")

        recommendation = self._generate_recommendation(report, idea_description)
        report["recommendation"] = recommendation

        if progress_callback:
            progress_callback("complete", 100, "Report complete")

        logger.info("Report generation complete")
        return report

    def _gather_zep_insights(self, graph: KnowledgeGraph, idea: str) -> str:
        """Use Zep tools for deep graph retrieval (if available)."""
        if not graph._use_zep or not graph._graph_id:
            return ""

        try:
            from engine.zep_tools import ZepToolsService
            tools = ZepToolsService(zep_client=graph._zep, llm_client=self.llm)

            insights_parts = []

            # InsightForge — deep multi-dimensional search
            insight = tools.insight_forge(
                graph_id=graph._graph_id,
                query=f"What are the key outcomes, reactions, and risks of: {idea[:200]}",
                simulation_requirement=idea,
            )
            if insight.semantic_facts:
                insights_parts.append(insight.to_text())

            # PanoramaSearch — broad view with evolution
            panorama = tools.panorama_search(
                graph_id=graph._graph_id,
                query=f"Complete overview of simulation results for: {idea[:100]}",
            )
            if panorama.active_facts:
                insights_parts.append(panorama.to_text())

            return "\n\n---\n\n".join(insights_parts)

        except Exception as e:
            logger.warning(f"Zep insights retrieval failed: {e}")
            return ""

    def _compile_simulation_data(
        self, graph: KnowledgeGraph, personas: list[AgentPersona],
        rounds: list[RoundSummary],
    ) -> str:
        """Compile all simulation data into a text summary."""
        parts = [
            f"## Simulation Summary",
            f"- Total agents: {len(personas)}",
            f"- Total rounds: {len(rounds)}",
            f"- Agent types: {', '.join(set(p.entity_type for p in personas))}",
            "",
            "## Agent Roster:",
        ]
        for p in personas:
            parts.append(f"- **{p.name}** ({p.role}): {p.initial_stance}")

        parts.append("\n## Round-by-Round Summary:")
        for r in rounds:
            parts.append(f"\n### Round {r.round_number} (Market Sentiment: {r.market_sentiment})")
            for a in r.actions[:8]:
                parts.append(f"- [{a.action_type.upper()}] {a.agent_name}: {a.content}")
            if r.key_events:
                parts.append(f"Key events: {'; '.join(r.key_events[:3])}")

        # Key facts from graph
        facts = graph.get_all_facts()
        if facts:
            parts.append(f"\n## Key Facts from Knowledge Graph ({len(facts)} total):")
            for f in facts[:30]:
                parts.append(f"- {f}")

        # Sentiment arc
        parts.append("\n## Sentiment Arc:")
        for r in rounds:
            parts.append(f"Round {r.round_number}: {r.market_sentiment}")

        return "\n".join(parts)

    def _generate_full_report(self, sim_data: str, idea: str, zep_insights: str) -> dict:
        """Generate the main report sections using Gemini."""
        insights_section = ""
        if zep_insights:
            insights_section = f"""

DEEP INSIGHTS FROM KNOWLEDGE GRAPH (use these as primary evidence):
{zep_insights[:8000]}"""

        prompt = f"""You are an expert strategy analyst writing a prediction report for a CEO.
This report is based on a multi-agent market simulation.

THE IDEA TESTED: "{idea}"

SIMULATION DATA:
{sim_data[:10000]}
{insights_section}

Generate a detailed report. Return JSON:
{{
    "executive_summary": "3-4 paragraph executive summary of findings",
    "market_reaction_analysis": "Detailed analysis of how market participants reacted",
    "opportunity_analysis": "Key opportunities identified",
    "threat_analysis": "Key threats and risks identified",
    "competitor_reaction_forecast": "How competitors are likely to respond",
    "regulatory_flags": "Any regulatory or compliance concerns",
    "consumer_sentiment": "How consumers/customers are likely to react",
    "timeline_recommendation": "Recommended implementation timeline with phases",
    "key_metrics": ["metric 1 to track", "metric 2", "metric 3"],
    "confidence_level": "high/medium/low",
    "confidence_reasoning": "Why this confidence level"
}}

Be specific and data-driven. Reference specific agent behaviors and events.
Quote simulation facts where possible using > quote format."""

        try:
            return self.llm.strong_chat_json(prompt, temperature=0.5, max_tokens=8000)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._fallback_report(e)

    def _generate_risk_matrix(self, sim_data: str, idea: str, zep_insights: str) -> list[dict]:
        """Generate a risk matrix from simulation data."""
        prompt = f"""Based on simulation of the idea: "{idea}"

{sim_data[:5000]}
{zep_insights[:3000] if zep_insights else ''}

Create a risk matrix. Return JSON array:
[
    {{
        "risk": "Risk description",
        "likelihood": "high/medium/low",
        "impact": "high/medium/low",
        "mitigation": "Suggested mitigation strategy",
        "source": "Which agent/event raised this risk"
    }}
]

Include 4-6 specific risks from the simulation."""

        try:
            result = self.llm.chat_json(prompt, temperature=0.4)
            if isinstance(result, list):
                return result
            return result.get("risks", result.get("risk_matrix", []))
        except Exception as e:
            logger.error(f"Risk matrix generation failed: {e}")
            return [{"risk": "Unable to generate", "likelihood": "unknown",
                      "impact": "unknown", "mitigation": "N/A", "source": "system"}]

    def _generate_recommendation(self, report: dict, idea: str) -> dict:
        """Generate a final go/no-go recommendation."""
        prompt = f"""Based on analysis of the idea: "{idea}"

Executive Summary: {report.get('executive_summary', '')[:1000]}
Threats: {report.get('threat_analysis', '')[:500]}
Opportunities: {report.get('opportunity_analysis', '')[:500]}
Confidence: {report.get('confidence_level', 'unknown')}

Provide a final recommendation. Return JSON:
{{
    "decision": "GO / NO-GO / CONDITIONAL GO",
    "confidence_percentage": 75,
    "reasoning": "2-3 sentence justification",
    "conditions": ["condition 1 if conditional", "condition 2"],
    "next_steps": ["immediate action 1", "action 2", "action 3"]
}}"""

        try:
            return self.llm.chat_json(prompt, temperature=0.3)
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                "decision": "INSUFFICIENT DATA",
                "confidence_percentage": 0,
                "reasoning": f"Could not generate recommendation: {e}",
                "conditions": [],
                "next_steps": ["Retry simulation with adjusted parameters"],
            }

    @staticmethod
    def _fallback_report(error) -> dict:
        return {
            "executive_summary": f"Report generation encountered an error: {error}",
            "market_reaction_analysis": "Unable to generate.",
            "opportunity_analysis": "Unable to generate.",
            "threat_analysis": "Unable to generate.",
            "competitor_reaction_forecast": "Unable to generate.",
            "regulatory_flags": "Unable to generate.",
            "consumer_sentiment": "Unable to generate.",
            "timeline_recommendation": "Unable to generate.",
            "key_metrics": [],
            "confidence_level": "low",
            "confidence_reasoning": "Report generation failed.",
        }
