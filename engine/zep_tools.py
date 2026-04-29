"""
Idea Tester — Zep Tools Service
Port of MiroFish's zep_tools.py.
Search, retrieval, and analysis tools for Report Agent's ReACT loop.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger("idea_tester.zep_tools")


@dataclass
class NodeInfo:
    """Graph node information."""
    uuid: str
    name: str
    labels: list
    summary: str
    attributes: dict = field(default_factory=dict)

    def to_text(self) -> str:
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Entity")
        return f"Entity: {self.name} (Type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Graph edge information."""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    def to_text(self, include_temporal: bool = False) -> str:
        text = f"Fact: {self.fact}"
        if include_temporal and self.valid_at:
            text += f"\nValid: {self.valid_at} - {self.invalid_at or 'present'}"
        return text


@dataclass
class SearchResult:
    """Search result from Zep."""
    facts: list[str]
    edges: list[dict]
    nodes: list[dict]
    query: str
    total_count: int

    def to_text(self) -> str:
        parts = [f"Search query: {self.query}", f"Found {self.total_count} results"]
        if self.facts:
            parts.append("\n### Relevant Facts:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(f'{i}. "{fact}"')
        return "\n".join(parts)


@dataclass
class InsightForgeResult:
    """Deep insight retrieval result (multi-dimensional)."""
    query: str
    sub_queries: list[str]
    semantic_facts: list[str] = field(default_factory=list)
    entity_insights: list[dict] = field(default_factory=list)
    relationship_chains: list[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0

    def to_text(self) -> str:
        parts = [
            f"## Deep Insight Analysis",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Facts: {self.total_facts}",
            f"- Entities: {self.total_entities}",
        ]
        if self.semantic_facts:
            parts.append(f"\n### Key Facts (cite these in report)")
            for i, f in enumerate(self.semantic_facts, 1):
                parts.append(f'{i}. "{f}"')
        if self.entity_insights:
            parts.append(f"\n### Core Entities")
            for e in self.entity_insights:
                parts.append(f"- **{e.get('name', 'Unknown')}** ({e.get('type', 'Entity')})")
                if e.get('summary'):
                    parts.append(f'  Summary: "{e["summary"]}"')
        if self.relationship_chains:
            parts.append(f"\n### Relationship Chains")
            for c in self.relationship_chains:
                parts.append(f"- {c}")
        return "\n".join(parts)


@dataclass
class PanoramaResult:
    """Broad search result with active + historical facts."""
    query: str
    all_nodes: list[NodeInfo] = field(default_factory=list)
    all_edges: list[EdgeInfo] = field(default_factory=list)
    active_facts: list[str] = field(default_factory=list)
    historical_facts: list[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0

    def to_text(self) -> str:
        parts = [
            f"## Panorama View",
            f"Query: {self.query}",
            f"Nodes: {self.total_nodes}, Edges: {self.total_edges}",
            f"Active facts: {len(self.active_facts)}, Historical: {len(self.historical_facts)}",
        ]
        if self.active_facts:
            parts.append(f"\n### Active Facts (current state)")
            for i, f in enumerate(self.active_facts, 1):
                parts.append(f'{i}. "{f}"')
        if self.historical_facts:
            parts.append(f"\n### Historical Facts (evolution)")
            for i, f in enumerate(self.historical_facts, 1):
                parts.append(f'{i}. "{f}"')
        if self.all_nodes:
            parts.append(f"\n### Entities")
            for n in self.all_nodes:
                t = next((l for l in n.labels if l not in ["Entity", "Node"]), "Entity")
                parts.append(f"- **{n.name}** ({t})")
        return "\n".join(parts)


class ZepToolsService:
    """
    Zep retrieval tools for Report Agent.
    
    Tools:
    1. insight_forge — Deep multi-dimensional retrieval (strongest)
    2. panorama_search — Broad view including historical/expired facts
    3. quick_search — Fast simple search
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, zep_client=None, llm_client=None):
        self._zep = zep_client
        self._llm = llm_client

        if not self._zep and settings.ZEP_API_KEY:
            try:
                from zep_cloud.client import Zep
                self._zep = Zep(api_key=settings.ZEP_API_KEY)
            except Exception as e:
                logger.warning(f"Zep client init failed: {e}")

    @property
    def llm(self):
        if self._llm is None:
            from engine.llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    def _call_with_retry(self, func, operation_name: str):
        """API call with exponential backoff."""
        last_exc = None
        delay = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                return func()
            except Exception as e:
                last_exc = e
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Zep {operation_name} attempt {attempt+1} failed: {str(e)[:100]}, retrying in {delay:.0f}s...")
                    time.sleep(delay)
                    delay *= 2
        raise last_exc

    # ── Core Tools ───────────────────────────────────────

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """Fast semantic search on graph edges."""
        if not self._zep:
            return SearchResult(facts=[], edges=[], nodes=[], query=query, total_count=0)

        try:
            result = self._call_with_retry(
                func=lambda: self._zep.graph.search(
                    graph_id=graph_id, query=query,
                    limit=limit, scope="edges", reranker="rrf",
                ),
                operation_name="quick_search",
            )
            facts = []
            if hasattr(result, 'edges') and result.edges:
                facts = [e.fact for e in result.edges if hasattr(e, 'fact') and e.fact]
            return SearchResult(facts=facts, edges=[], nodes=[], query=query, total_count=len(facts))
        except Exception as e:
            logger.error(f"Quick search failed: {e}")
            return SearchResult(facts=[], edges=[], nodes=[], query=query, total_count=0)

    def insight_forge(self, graph_id: str, query: str, simulation_requirement: str = "") -> InsightForgeResult:
        """
        Deep multi-dimensional retrieval (MiroFish InsightForge port).
        Auto-generates sub-queries and searches edges + nodes.
        """
        if not self._zep:
            return InsightForgeResult(query=query, sub_queries=[])

        # Generate sub-queries using LLM
        sub_queries = self._generate_sub_queries(query, simulation_requirement)

        all_facts = set()
        all_entities = {}

        for sq in sub_queries:
            try:
                # Edge search
                edge_result = self._call_with_retry(
                    func=lambda q=sq: self._zep.graph.search(
                        graph_id=graph_id, query=q, limit=15, scope="edges", reranker="rrf",
                    ),
                    operation_name=f"insight_edges({sq[:30]})",
                )
                if hasattr(edge_result, 'edges') and edge_result.edges:
                    for e in edge_result.edges:
                        if hasattr(e, 'fact') and e.fact:
                            all_facts.add(e.fact)

                # Node search
                node_result = self._call_with_retry(
                    func=lambda q=sq: self._zep.graph.search(
                        graph_id=graph_id, query=q, limit=10, scope="nodes", reranker="rrf",
                    ),
                    operation_name=f"insight_nodes({sq[:30]})",
                )
                if hasattr(node_result, 'nodes') and node_result.nodes:
                    for n in node_result.nodes:
                        name = getattr(n, 'name', '')
                        if name and name not in all_entities:
                            all_entities[name] = {
                                "name": name,
                                "type": next((l for l in (getattr(n, 'labels', []) or []) if l not in ["Entity", "Node"]), "Entity"),
                                "summary": getattr(n, 'summary', ''),
                            }
            except Exception as e:
                logger.warning(f"InsightForge sub-query failed ({sq[:30]}): {e}")

        return InsightForgeResult(
            query=query,
            sub_queries=sub_queries,
            semantic_facts=list(all_facts),
            entity_insights=list(all_entities.values()),
            total_facts=len(all_facts),
            total_entities=len(all_entities),
        )

    def panorama_search(self, graph_id: str, query: str) -> PanoramaResult:
        """Broad search that includes all nodes and edges (active + historical)."""
        if not self._zep:
            return PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        all_edges = self.get_all_edges(graph_id)

        active_facts = [e.fact for e in all_edges if e.fact and not e.is_expired]
        historical_facts = [e.fact for e in all_edges if e.fact and e.is_expired]

        return PanoramaResult(
            query=query,
            all_nodes=all_nodes,
            all_edges=all_edges,
            active_facts=active_facts,
            historical_facts=historical_facts,
            total_nodes=len(all_nodes),
            total_edges=len(all_edges),
        )

    # ── Basic Getters ────────────────────────────────────

    def get_all_nodes(self, graph_id: str) -> list[NodeInfo]:
        """Get all nodes from the graph."""
        if not self._zep:
            return []
        try:
            nodes = self._call_with_retry(
                func=lambda: self._zep.graph.node.get_by_graph_id(graph_id=graph_id),
                operation_name="get_all_nodes",
            )
            return [
                NodeInfo(
                    uuid=getattr(n, 'uuid_', None) or getattr(n, 'uuid', ''),
                    name=getattr(n, 'name', '') or '',
                    labels=getattr(n, 'labels', []) or [],
                    summary=getattr(n, 'summary', '') or '',
                    attributes=getattr(n, 'attributes', {}) or {},
                )
                for n in (nodes or [])
            ]
        except Exception as e:
            logger.error(f"Get all nodes failed: {e}")
            return []

    def get_all_edges(self, graph_id: str) -> list[EdgeInfo]:
        """Get all edges from the graph."""
        if not self._zep:
            return []
        try:
            edges = self._call_with_retry(
                func=lambda: self._zep.graph.edge.get_by_graph_id(graph_id=graph_id),
                operation_name="get_all_edges",
            )
            return [
                EdgeInfo(
                    uuid=getattr(e, 'uuid_', None) or getattr(e, 'uuid', ''),
                    name=getattr(e, 'name', '') or '',
                    fact=getattr(e, 'fact', '') or '',
                    source_node_uuid=getattr(e, 'source_node_uuid', '') or '',
                    target_node_uuid=getattr(e, 'target_node_uuid', '') or '',
                    created_at=getattr(e, 'created_at', None),
                    valid_at=getattr(e, 'valid_at', None),
                    invalid_at=getattr(e, 'invalid_at', None),
                    expired_at=getattr(e, 'expired_at', None),
                )
                for e in (edges or [])
            ]
        except Exception as e:
            logger.error(f"Get all edges failed: {e}")
            return []

    # ── Internals ────────────────────────────────────────

    def _generate_sub_queries(self, query: str, simulation_requirement: str) -> list[str]:
        """Use LLM to decompose a query into sub-queries."""
        try:
            prompt = f"""Decompose this analysis question into 3-4 focused sub-questions for graph search.

Main question: {query}
Context: {simulation_requirement}

Return a JSON array of strings, each being a focused search query.
Example: ["What did consumers do?", "How did competitors react?", "What regulatory concerns emerged?"]"""

            result = self.llm.chat_json(prompt, temperature=0.3, max_tokens=500)
            if isinstance(result, list):
                return result[:5]
            return result.get("queries", result.get("sub_queries", [query]))[:5]
        except Exception as e:
            logger.warning(f"Sub-query generation failed: {e}")
            return [query]
