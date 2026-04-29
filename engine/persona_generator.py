"""
Idea Tester — Persona Generator
Replaces OASIS profile generator.
Generates rich AI agent personas from knowledge graph entities using Gemini.
Uses Zep search enrichment for deeper persona context (MiroFish pattern).
"""

import json
import logging
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from engine.llm_client import LLMClient
from engine.knowledge_graph import KnowledgeGraph, Entity

logger = logging.getLogger("idea_tester.persona_generator")

# MiroFish constants
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP",
]

INDIVIDUAL_ENTITY_TYPES = [
    "student", "alumni", "professor", "person", "publicfigure",
    "expert", "faculty", "official", "journalist", "activist",
    "consumer", "employee", "investor",
]

GROUP_ENTITY_TYPES = [
    "university", "governmentagency", "organization", "ngo",
    "mediaoutlet", "company", "institution", "group", "community",
    "regulator", "competitor", "market", "technology",
]


@dataclass
class AgentPersona:
    """A fully-specified AI agent persona for simulation."""
    id: str
    name: str
    role: str
    entity_type: str
    personality: str
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    behavioral_traits: list[str] = field(default_factory=list)
    knowledge: str = ""
    initial_stance: str = ""
    backstory: str = ""
    communication_style: str = ""
    # MiroFish profile fields
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "entity_type": self.entity_type, "personality": self.personality,
            "goals": self.goals, "constraints": self.constraints,
            "behavioral_traits": self.behavioral_traits,
            "knowledge": self.knowledge, "initial_stance": self.initial_stance,
            "backstory": self.backstory,
            "communication_style": self.communication_style,
            "age": self.age, "gender": self.gender, "mbti": self.mbti,
            "country": self.country, "profession": self.profession,
            "interested_topics": self.interested_topics,
        }

    def to_system_prompt(self) -> str:
        """Convert persona to a system prompt for the LLM during simulation."""
        parts = [
            f"You are {self.name}, a {self.role}.",
            f"\nPersonality: {self.personality}",
        ]
        if self.backstory:
            parts.append(f"\nBackstory: {self.backstory}")
        if self.goals:
            parts.append(f"\nYour goals: {', '.join(self.goals)}")
        if self.constraints:
            parts.append(f"\nYour constraints: {', '.join(self.constraints)}")
        if self.behavioral_traits:
            parts.append(f"\nBehavioral traits: {', '.join(self.behavioral_traits)}")
        if self.knowledge:
            parts.append(f"\nWhat you know: {self.knowledge}")
        if self.initial_stance:
            parts.append(f"\nYour initial stance: {self.initial_stance}")
        if self.communication_style:
            parts.append(f"\nCommunication style: {self.communication_style}")
        if self.mbti:
            parts.append(f"\nMBTI type: {self.mbti}")
        parts.append(
            "\n\nStay in character at all times. Respond based on your personality, "
            "goals, constraints, and knowledge. Be realistic and specific."
        )
        return "\n".join(parts)


import threading
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    add_script_run_ctx = None
    get_script_run_ctx = None

class PersonaGenerator:
    """Generates agent personas from knowledge graph entities using LLM + Zep.
    
    Uses BATCHED generation (6 personas per LLM call) for speed.
    """

    BATCH_SIZE = 6  # Personas per LLM call

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()

    def generate_personas(
        self,
        graph: KnowledgeGraph,
        idea_description: str,
        parallel: int = 3,
        progress_callback=None,
    ) -> list[AgentPersona]:
        """Generate agent personas for ALL entities via batched LLM calls."""
        entities = graph.get_all_entities()
        total = len(entities)

        # Ollama can only handle 1-2 concurrent requests — go sequential
        if self.llm.provider == "ollama":
            parallel = 1

        logger.info(f"Generating personas for {total} entities (batch_size={self.BATCH_SIZE}, workers={parallel})...")

        # Split entities into batches
        batches = [entities[i:i + self.BATCH_SIZE] for i in range(0, total, self.BATCH_SIZE)]
        logger.info(f"Split into {len(batches)} batches — ~{len(batches)} LLM calls instead of {total}")

        personas: list[AgentPersona] = []
        completed = 0

        is_ollama = self.llm.provider == "ollama"
        ctx = get_script_run_ctx() if get_script_run_ctx else None

        if is_ollama:
            # Ollama: strictly sequential, one batch at a time
            import time as _time
            for i, batch in enumerate(batches):
                try:
                    batch_entries = self._build_batch_entries(graph, batch)
                    batch_personas = self._generate_batch_prompt(batch_entries, batch, idea_description)
                    personas.extend(batch_personas)
                    completed += len(batch_personas)
                    if progress_callback:
                        last_name = batch_personas[-1].name if batch_personas else "..."
                        progress_callback(completed, total, last_name)
                    logger.info(f"Batch {i+1}/{len(batches)} complete — {len(batch_personas)} personas")
                except Exception as e:
                    completed += len(batch)
                    logger.error(f"Batch {i+1} failed: {e}")
                    for entity in batch:
                        fb = self._fallback_persona(entity)
                        personas.append(AgentPersona(
                            id=entity.id, name=entity.name,
                            role=fb.get("role", entity.entity_type),
                            entity_type=entity.entity_type,
                            personality=fb.get("personality", "Rational market participant"),
                            goals=fb.get("goals", []),
                            constraints=fb.get("constraints", []),
                            behavioral_traits=fb.get("behavioral_traits", ["rational"]),
                            knowledge=entity.description,
                            initial_stance=fb.get("initial_stance", "Neutral"),
                            backstory=fb.get("backstory", ""),
                            communication_style=fb.get("communication_style", "Professional"),
                            mbti=random.choice(MBTI_TYPES),
                        ))
                    if progress_callback:
                        progress_callback(completed, total, "fallback")
                # Brief pause between batches for Ollama
                if i < len(batches) - 1:
                    _time.sleep(3)
        else:
            # Gemini/Groq: parallel execution
            def _generate_batch(batch: list[Entity], batch_idx: int) -> list[AgentPersona]:
                if add_script_run_ctx and ctx:
                    add_script_run_ctx(threading.current_thread(), ctx)
                batch_entries = self._build_batch_entries(graph, batch)
                return self._generate_batch_prompt(batch_entries, batch, idea_description)

            with ThreadPoolExecutor(max_workers=parallel) as executor:
                future_map = {
                    executor.submit(_generate_batch, batch, i): i
                    for i, batch in enumerate(batches)
                }
                for future in as_completed(future_map):
                    batch_idx = future_map[future]
                    try:
                        batch_personas = future.result()
                        personas.extend(batch_personas)
                        completed += len(batch_personas)
                        if progress_callback:
                            last_name = batch_personas[-1].name if batch_personas else "..."
                            progress_callback(completed, total, last_name)
                        logger.info(f"Batch {batch_idx+1}/{len(batches)} complete — {len(batch_personas)} personas")
                    except Exception as e:
                        completed += len(batches[batch_idx])
                        logger.error(f"Batch {batch_idx+1} failed: {e}")
                        for entity in batches[batch_idx]:
                            fb = self._fallback_persona(entity)
                            personas.append(AgentPersona(
                                id=entity.id, name=entity.name,
                                role=fb.get("role", entity.entity_type),
                                entity_type=entity.entity_type,
                                personality=fb.get("personality", "Rational market participant"),
                                goals=fb.get("goals", []),
                                constraints=fb.get("constraints", []),
                                behavioral_traits=fb.get("behavioral_traits", ["rational"]),
                                knowledge=entity.description,
                                initial_stance=fb.get("initial_stance", "Neutral"),
                                backstory=fb.get("backstory", ""),
                                communication_style=fb.get("communication_style", "Professional"),
                                mbti=random.choice(MBTI_TYPES),
                            ))

        # Sort by original order
        entity_ids = [e.id for e in entities]
        personas.sort(key=lambda p: entity_ids.index(p.id) if p.id in entity_ids else 999)

        logger.info(f"Generated {len(personas)} agent personas in {len(batches)} batches")
        return personas

    def _build_batch_entries(self, graph: KnowledgeGraph, batch: list[Entity]) -> list[dict]:
        """Build batch entry dicts for a list of entities."""
        entries = []
        for entity in batch:
            rels = graph.get_entity_relationships(entity.id)
            rel_facts = [r.fact for r in rels if r.fact][:5]
            is_individual = entity.entity_type.lower() in INDIVIDUAL_ENTITY_TYPES
            entries.append({
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
                "is_individual": is_individual,
                "relationships": rel_facts,
            })
        return entries

    def _generate_batch_prompt(
        self, batch_entries: list[dict], entities: list[Entity], idea: str
    ) -> list[AgentPersona]:
        """Generate multiple personas in a single LLM call."""
        entities_text = ""
        for entry in batch_entries:
            rels = "\n".join(f"  - {f}" for f in entry["relationships"]) if entry["relationships"] else "  None"
            entities_text += f"""
--- Entity #{entry['id']} ---
Name: {entry['name']}
Type: {entry['type']}
Description: {entry['description']}
Category: {"Individual" if entry['is_individual'] else "Organization/Group"}
Relationships:
{rels}
"""

        prompt = f"""Generate detailed simulation agent personas for ALL of the following entities.
Each persona will participate in a market simulation.

THE IDEA BEING TESTED: {idea}

ENTITIES TO GENERATE PERSONAS FOR:
{entities_text}

For EACH entity, generate a persona. Return a JSON array with one object per entity, in order.
Each object must have:
{{
    "entity_id": "the entity id from above",
    "role": "Specific role title",
    "personality": "2-3 sentence personality description",
    "goals": ["goal 1", "goal 2"],
    "constraints": ["constraint 1"],
    "behavioral_traits": ["trait 1", "trait 2"],
    "knowledge": "What they know about the market",
    "initial_stance": "Their stance toward the idea (supportive/skeptical/neutral and why)",
    "backstory": "2-sentence backstory",
    "communication_style": "How they communicate",
    "age": 35,
    "gender": "male/female/other",
    "mbti": "INTJ",
    "country": "Country",
    "profession": "Their job",
    "interested_topics": ["topic 1", "topic 2"]
}}

Return ONLY a JSON array of {len(batch_entries)} objects. No explanations."""

        try:
            data = self.llm.chat_json(prompt, temperature=0.8, max_tokens=6000)
            # Handle both list and dict with array inside
            if isinstance(data, dict):
                data = data.get("personas", data.get("entities", data.get("results", [])))
            if not isinstance(data, list):
                data = [data]
        except Exception as e:
            logger.warning(f"Batch persona generation failed: {e}, using fallbacks")
            data = [self._fallback_persona(ent) for ent in entities]

        # Map results back to entities
        personas = []
        for i, entity in enumerate(entities):
            d = data[i] if i < len(data) else self._fallback_persona(entity)

            personas.append(AgentPersona(
                id=entity.id,
                name=entity.name,
                role=d.get("role", entity.entity_type),
                entity_type=entity.entity_type,
                personality=d.get("personality", "A rational market participant."),
                goals=d.get("goals", ["Maximize own interests"]),
                constraints=d.get("constraints", []),
                behavioral_traits=d.get("behavioral_traits", ["rational"]),
                knowledge=d.get("knowledge", entity.description),
                initial_stance=d.get("initial_stance", "Neutral"),
                backstory=d.get("backstory", ""),
                communication_style=d.get("communication_style", "Professional"),
                age=d.get("age"),
                gender=d.get("gender"),
                mbti=d.get("mbti", random.choice(MBTI_TYPES)),
                country=d.get("country"),
                profession=d.get("profession"),
                interested_topics=d.get("interested_topics", []),
            ))

        return personas

    @staticmethod
    def _fallback_persona(entity: Entity) -> dict:
        """Fallback persona when LLM fails."""
        return {
            "role": entity.entity_type,
            "personality": f"A typical {entity.entity_type.lower()} with standard market behavior.",
            "goals": ["Act in self-interest", "Respond rationally to market changes"],
            "constraints": ["Limited information", "Budget constraints"],
            "behavioral_traits": ["rational", "reactive"],
            "knowledge": entity.description,
            "initial_stance": "Neutral — waiting for more information",
            "backstory": f"{entity.name} is an active participant in this market.",
            "communication_style": "Direct and factual",
            "mbti": random.choice(MBTI_TYPES),
        }

