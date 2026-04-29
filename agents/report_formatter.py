"""
Idea Tester — CEO Report Formatter
Transforms the raw simulation report into a polished, CEO-friendly output.
"""

import logging
from typing import Optional

from engine.llm_client import LLMClient

logger = logging.getLogger("idea_tester.report_formatter")


class ReportFormatter:
    """Formats simulation report into CEO-ready presentation."""

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def format(self, report: dict, idea: str) -> dict:
        """
        Re-format the raw report into a CEO-friendly structure.

        Returns a dict with formatted sections ready for display.
        """
        recommendation = report.get("recommendation", {})
        risk_matrix = report.get("risk_matrix", [])

        return {
            "title": f"Strategy Simulation Report",
            "idea": idea,
            "decision": recommendation.get("decision", "PENDING"),
            "confidence": recommendation.get("confidence_percentage", 0),
            "executive_summary": report.get("executive_summary", ""),
            "sections": [
                {
                    "title": "📊 Market Reaction Analysis",
                    "content": report.get("market_reaction_analysis", ""),
                    "icon": "📊",
                },
                {
                    "title": "🎯 Opportunities",
                    "content": report.get("opportunity_analysis", ""),
                    "icon": "🎯",
                },
                {
                    "title": "⚠️ Threats & Risks",
                    "content": report.get("threat_analysis", ""),
                    "icon": "⚠️",
                },
                {
                    "title": "🏢 Competitor Response Forecast",
                    "content": report.get("competitor_reaction_forecast", ""),
                    "icon": "🏢",
                },
                {
                    "title": "⚖️ Regulatory Flags",
                    "content": report.get("regulatory_flags", ""),
                    "icon": "⚖️",
                },
                {
                    "title": "👥 Consumer Sentiment",
                    "content": report.get("consumer_sentiment", ""),
                    "icon": "👥",
                },
                {
                    "title": "📅 Timeline Recommendation",
                    "content": report.get("timeline_recommendation", ""),
                    "icon": "📅",
                },
            ],
            "risk_matrix": risk_matrix,
            "key_metrics": report.get("key_metrics", []),
            "recommendation": recommendation,
            "confidence_reasoning": report.get("confidence_reasoning", ""),
        }
