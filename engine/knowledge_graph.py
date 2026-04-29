"""
Idea Tester — Knowledge Graph (Zep Cloud + NetworkX fallback)
Replaces local-only NetworkX with Zep Cloud as primary graph store.
Falls back to local NetworkX if Zep is not configured.
"""

import json
import time
import uuid
import logging
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

from config import settings

logger = logging.getLogger("idea_tester.knowledge_graph")


# ── Data Structures ──────────────────────────────────────

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    description: str = ""
    attributes: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "labels": self.labels,
        }


@dataclass
class Relationship:
    id: str
    source_id: str
    target_id: str
    relation_type: str
    fact: str = ""
    round_created: int = -1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "fact": self.fact, "round_created": self.round_created,
        }


@dataclass
class Memory:
    agent_id: str
    content: str
    round_number: int = -1


# ── Zep Cloud Knowledge Graph ───────────────────────────

class KnowledgeGraph:
    """
    Knowledge graph backed by Zep Cloud (primary) + local NetworkX (fallback).
    
    If ZEP_API_KEY is set: uses Zep Cloud for all graph ops (search, entities, edges).
    If not: falls back to local NetworkX in-memory graph.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self._memories: dict[str, list[Memory]] = {}

        # Zep Cloud client
        self._zep = None
        self._graph_id = None
        self._use_zep = False

        # Local fallback
        self._local_graph = nx.DiGraph()

        # Try Zep Cloud
        if settings.ZEP_API_KEY:
            try:
                from zep_cloud.client import Zep
                self._zep = Zep(api_key=settings.ZEP_API_KEY)
                self._use_zep = True
                logger.info("Zep Cloud connected — using cloud graph memory")
            except ImportError:
                logger.warning("zep-cloud not installed — using local NetworkX fallback")
            except Exception as e:
                logger.warning(f"Zep Cloud init failed ({e}) — using local NetworkX fallback")
        else:
            logger.info("No ZEP_API_KEY — using local NetworkX graph")

    # ── Graph Lifecycle ──────────────────────────────────

    def create_graph(self, name: str = "idea_tester_graph") -> str:
        """Create a new Zep graph (or local graph ID)."""
        if self._use_zep:
            try:
                # Use standard uuid for graph_id or just the name
                graph_uuid = f"graph_{uuid.uuid4().hex[:8]}"
                result = self._zep.graph.create(
                    graph_id=graph_uuid,
                    name=name,
                    description="Idea Tester simulation knowledge graph",
                )
                self._graph_id = graph_uuid
                logger.info(f"Zep graph created: {self._graph_id}")
                return self._graph_id
            except Exception as e:
                logger.error(f"Zep graph creation failed: {e}")
                self._use_zep = False

        # Fallback
        self._graph_id = f"local_{uuid.uuid4().hex[:8]}"
        logger.info(f"Local graph created: {self._graph_id}")
        return self._graph_id

    # ── Build from Seed Document ─────────────────────────

    def build_from_seed(self, seed: dict):
        """
        Build knowledge graph from a seed document.
        
        For Zep: feeds text episodes that Zep auto-extracts into entities/edges.
        For local: directly adds entities + relationships from the seed structure.
        """
        logger.info("Building knowledge graph from seed document...")

        if not self._graph_id:
            self.create_graph()

        # Extract entities from seed
        for ent_data in seed.get("entities", []):
            ent_id = str(uuid.uuid4())[:8]
            entity = Entity(
                id=ent_id,
                name=ent_data.get("name", f"Entity_{ent_id}"),
                entity_type=ent_data.get("type", "Entity"),
                description=ent_data.get("description", ""),
                attributes=ent_data.get("attributes", {}),
                labels=[ent_data.get("type", "Entity")],
            )
            self.add_entity(entity)

        # Extract relationships from seed
        for rel_data in seed.get("relationships", []):
            source_name = rel_data.get("source", "")
            target_name = rel_data.get("target", "")

            # Find entity IDs by name
            source_id = self._find_entity_by_name(source_name)
            target_id = self._find_entity_by_name(target_name)

            if source_id and target_id:
                rel = Relationship(
                    id=str(uuid.uuid4())[:8],
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel_data.get("type", "related_to"),
                    fact=rel_data.get("fact", ""),
                )
                self.add_relationship(rel)

        # Feed to Zep as text episodes for deeper extraction
        if self._use_zep:
            self._feed_seed_to_zep(seed)

        logger.info(
            f"Graph built: {len(self.entities)} entities, "
            f"{len(self.relationships)} relationships"
        )

    def _feed_seed_to_zep(self, seed: dict):
        """Feed seed document text to Zep for entity/edge extraction."""
        if not self._zep or not self._graph_id:
            return

        # Build text from seed for Zep episode ingestion
        world_state = seed.get("world_state", "")
        event_desc = seed.get("event_description", "")

        entity_text = "\n".join(
            f"{e.get('name', '')} ({e.get('type', '')}): {e.get('description', '')}"
            for e in seed.get("entities", [])
        )
        rel_text = "\n".join(
            f"{r.get('source', '')} {r.get('type', '')} {r.get('target', '')}: {r.get('fact', '')}"
            for r in seed.get("relationships", [])
        )

        full_text = f"""Market World State:
{world_state}

Event Being Simulated:
{event_desc}

Key Entities:
{entity_text}

Key Relationships:
{rel_text}"""

        # Split into chunks and add as episodes in parallel (Optimization #3)
        chunks = self._chunk_text(full_text, chunk_size=500)

        def _add_chunk(args):
            i, chunk = args
            try:
                self._zep.graph.add(
                    graph_id=self._graph_id,
                    data=chunk,
                    type="text",
                    source_description="Seed document for simulation",
                )
            except Exception as e:
                logger.warning(f"Zep chunk add failed for chunk {i}: {e}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.map(_add_chunk, enumerate(chunks))

        logger.info(f"Fed {len(chunks)} text chunks to Zep in parallel")

    # ── Entity Operations ────────────────────────────────

    def add_entity(self, entity: Entity):
        """Add entity to local + Zep graph."""
        self.entities[entity.id] = entity
        self._local_graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_all_entities(self) -> list[Entity]:
        return list(self.entities.values())

    def _find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by name (case-insensitive)."""
        name_lower = name.lower()
        for eid, entity in self.entities.items():
            if entity.name.lower() == name_lower:
                return eid
        return None

    # ── Relationship Operations ──────────────────────────

    def add_relationship(self, rel: Relationship):
        """Add relationship to local + graph."""
        self.relationships.append(rel)
        self._local_graph.add_edge(
            rel.source_id, rel.target_id,
            relation_type=rel.relation_type,
            fact=rel.fact,
        )

    def get_entity_relationships(self, entity_id: str) -> list[Relationship]:
        """Get all relationships involving an entity."""
        return [
            r for r in self.relationships
            if r.source_id == entity_id or r.target_id == entity_id
        ]

    # ── Memory Operations ────────────────────────────────

    def add_memory(self, agent_id: str, content: str, round_number: int = -1):
        """Store agent memory (and optionally push to Zep)."""
        if agent_id not in self._memories:
            self._memories[agent_id] = []
        self._memories[agent_id].append(
            Memory(agent_id=agent_id, content=content, round_number=round_number)
        )

        # Push to Zep as episode
        if self._use_zep and self._graph_id:
            try:
                agent = self.entities.get(agent_id)
                agent_name = agent.name if agent else agent_id
                self._zep.graph.add(
                    graph_id=self._graph_id,
                    data=f"[Agent: {agent_name} | Round: {round_number}] {content}",
                    type="text",
                    source_description=f"Round {round_number} agent memory",
                )
            except Exception as e:
                logger.debug(f"Zep memory push failed: {e}")

    def get_memories(self, agent_id: str) -> list[Memory]:
        return self._memories.get(agent_id, [])

    # ── Search / Retrieval ───────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[str]:
        """Search the graph for facts related to a query."""
        if self._use_zep and self._graph_id:
            return self._zep_search(query, limit)
        return self._local_search(query, limit)

    def _zep_search(self, query: str, limit: int = 10) -> list[str]:
        """Search Zep Cloud graph."""
        facts = []
        try:
            result = self._zep.graph.search(
                graph_id=self._graph_id,
                query=query,
                limit=limit,
                scope="edges",
                reranker="rrf",
            )
            if hasattr(result, 'edges') and result.edges:
                for edge in result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
        except Exception as e:
            logger.warning(f"Zep search failed ({e}), falling back to local")
            return self._local_search(query, limit)

        return facts

    def _local_search(self, query: str, limit: int = 10) -> list[str]:
        """Keyword search on local graph facts."""
        query_lower = query.lower()
        scored = []
        for r in self.relationships:
            score = 0
            if query_lower in r.fact.lower():
                score += 10
            for word in query_lower.split():
                if len(word) > 2 and word in r.fact.lower():
                    score += 1
            if score > 0:
                scored.append((score, r.fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def get_all_facts(self) -> list[str]:
        """Get all facts from the graph."""
        if self._use_zep and self._graph_id:
            try:
                from engine.zep_tools import ZepToolsService
                tools = ZepToolsService(zep_client=self._zep)
                edges = tools.get_all_edges(self._graph_id)
                return [e.fact for e in edges if e.fact]
            except Exception:
                pass

        return [r.fact for r in self.relationships if r.fact]

    # ── Zep Entity Reader (MiroFish port) ────────────────

    def get_zep_nodes(self) -> list[dict]:
        """Get all nodes from Zep (for persona generation)."""
        if not self._use_zep or not self._graph_id:
            return []

        try:
            from engine.zep_tools import ZepToolsService
            tools = ZepToolsService(zep_client=self._zep)
            nodes = tools.get_all_nodes(self._graph_id)
            return [
                {
                    "uuid": n.uuid, "name": n.name,
                    "labels": n.labels, "summary": n.summary,
                    "attributes": n.attributes,
                }
                for n in nodes
            ]
        except Exception as e:
            logger.warning(f"Zep node fetch failed: {e}")
            return []

    def zep_search_for_entity(self, entity_name: str) -> dict:
        """Search Zep for rich context about an entity (MiroFish pattern)."""
        if not self._use_zep or not self._graph_id:
            return {"facts": [], "node_summaries": [], "context": ""}

        result = {"facts": [], "node_summaries": [], "context": ""}
        query = f"Tell me everything about {entity_name} and their role, actions, and relationships"

        try:
            # Search edges
            edge_result = self._zep.graph.search(
                query=query,
                graph_id=self._graph_id,
                limit=30,
                scope="edges",
                reranker="rrf",
            )
            if hasattr(edge_result, 'edges') and edge_result.edges:
                result["facts"] = list(set(
                    e.fact for e in edge_result.edges if hasattr(e, 'fact') and e.fact
                ))

            # Search nodes
            node_result = self._zep.graph.search(
                query=query,
                graph_id=self._graph_id,
                limit=20,
                scope="nodes",
                reranker="rrf",
            )
            if hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        result["node_summaries"].append(node.summary)

            # Build context
            parts = []
            if result["facts"]:
                parts.append("Facts:\n" + "\n".join(f"- {f}" for f in result["facts"][:20]))
            if result["node_summaries"]:
                parts.append("Related entities:\n" + "\n".join(f"- {s}" for s in result["node_summaries"][:10]))
            result["context"] = "\n\n".join(parts)

        except Exception as e:
            logger.warning(f"Zep entity search failed ({entity_name}): {e}")

        return result

    # ── Utilities ────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500) -> list[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        return chunks
