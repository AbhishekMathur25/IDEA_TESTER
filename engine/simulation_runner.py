"""
Idea Tester — OASIS Simulation Runner
Replaces the custom LLM-based simulation with CAMEL-OASIS social media simulation.
Runs Twitter + Reddit simulations in parallel using the OASIS environment.

Requires: pip install camel-ai[all] oasis-ai
"""

import asyncio
import csv
import json
import os
import random
import sqlite3
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List, Tuple

from config import settings
from engine.knowledge_graph import KnowledgeGraph, Relationship
from engine.persona_generator import AgentPersona
from engine.action_logger import PlatformActionLogger, SimulationLogManager

logger = logging.getLogger("idea_tester.simulation_runner")


# ── Data Classes ─────────────────────────────────────────

@dataclass
class AgentAction:
    """A single action taken by an agent in a round."""
    agent_id: int
    agent_name: str
    round_number: int
    action_type: str
    content: str = ""
    target_agent: str = ""
    sentiment: str = ""
    platform: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id, "agent_name": self.agent_name,
            "round_number": self.round_number, "action_type": self.action_type,
            "content": self.content, "target_agent": self.target_agent,
            "sentiment": self.sentiment, "platform": self.platform,
            "timestamp": self.timestamp,
        }


@dataclass
class RoundSummary:
    """Summary of what happened in a simulation round."""
    round_number: int
    actions: list[AgentAction] = field(default_factory=list)
    key_events: list[str] = field(default_factory=list)
    market_sentiment: str = "neutral"
    new_relationships: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "actions": [a.to_dict() for a in self.actions],
            "key_events": self.key_events,
            "market_sentiment": self.market_sentiment,
            "new_relationships": self.new_relationships,
            "total_actions": len(self.actions),
        }


# ── OASIS Action Types & Constants ───────────────────────

FILTERED_ACTIONS = {'refresh', 'sign_up'}

ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}


# ── Profile Writers ──────────────────────────────────────

def write_twitter_profiles(personas: list[AgentPersona], file_path: str):
    """Write agent personas to OASIS Twitter CSV format."""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'name', 'username', 'user_char', 'description'])

        for idx, p in enumerate(personas):
            user_char = p.to_system_prompt().replace('\n', ' ').replace('\r', ' ')
            description = f"{p.name} — {p.role}".replace('\n', ' ')
            username = p.name.lower().replace(' ', '_').replace('.', '')[:20]

            writer.writerow([idx, p.name, username, user_char, description])

    logger.info(f"Wrote {len(personas)} Twitter profiles to {file_path}")


def write_reddit_profiles(personas: list[AgentPersona], file_path: str):
    """Write agent personas to OASIS Reddit JSON format."""
    profiles = []
    for idx, p in enumerate(personas):
        profiles.append({
            "agent_id": idx,
            "name": p.name,
            "username": p.name.lower().replace(' ', '_').replace('.', '')[:20],
            "bio": p.to_system_prompt(),
            "persona": f"{p.role}: {p.personality}",
            "age": p.age or 30,
            "mbti": p.mbti or "INTJ",
            "gender": p.gender or "other",
            "country": p.country or "US",
            "profession": p.profession or p.role,
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(personas)} Reddit profiles to {file_path}")


# ── DB Action Fetching ───────────────────────────────────

def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetch new actions from OASIS SQLite database."""
    actions = []
    new_last_rowid = last_rowid

    if not os.path.exists(db_path):
        return actions, new_last_rowid

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))

        for rowid, user_id, action, info_json in cursor.fetchall():
            new_last_rowid = rowid

            if action in FILTERED_ACTIONS:
                continue

            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}

            # Simplify args
            simplified = {}
            for key in ('content', 'post_id', 'comment_id', 'quoted_id',
                        'new_post_id', 'follow_id', 'query', 'like_id', 'dislike_id'):
                if key in action_args:
                    simplified[key] = action_args[key]

            action_type = ACTION_TYPE_MAP.get(action, action.upper())

            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified,
            })

        conn.close()
    except Exception as e:
        logger.error(f"Failed to read actions from DB: {e}")

    return actions, new_last_rowid


# ── Active Agent Selection ───────────────────────────────

def get_active_agents_for_round(
    env, config: Dict[str, Any], current_hour: int, round_num: int
) -> list:
    """Determine which agents are active this round based on time/config."""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])

    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)

    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])

    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0

    target_count = int(random.uniform(base_min, base_max) * multiplier)

    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)

        if current_hour not in active_hours:
            continue

        if random.random() < activity_level:
            candidates.append(agent_id)

    selected_ids = random.sample(
        candidates, min(target_count, len(candidates))
    ) if candidates else []

    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass

    return active_agents


# ── Simulation Config Generator ──────────────────────────

def generate_simulation_config(
    personas: list[AgentPersona],
    idea_description: str,
    total_rounds: int = 15,
) -> dict:
    """Generate OASIS simulation config from personas."""
    agent_configs = []
    initial_posts = []

    for idx, p in enumerate(personas):
        # Agents with stronger stances are more active
        stance_lower = (p.initial_stance or "").lower()
        activity = 0.7 if any(w in stance_lower for w in ["strong", "support", "oppose"]) else 0.5

        agent_configs.append({
            "agent_id": idx,
            "entity_name": p.name,
            "entity_type": p.entity_type,
            "role": p.role,
            "active_hours": list(range(8, 23)),
            "activity_level": activity,
        })

    # The first agent posts the breaking news
    initial_posts.append({
        "poster_agent_id": 0,
        "content": f"BREAKING: {idea_description[:280]}",
    })

    # Some agents react immediately
    for idx in range(1, min(4, len(personas))):
        p = personas[idx]
        initial_posts.append({
            "poster_agent_id": idx,
            "content": f"As a {p.role}, my reaction to this news: {p.initial_stance or 'Watching closely.'}",
        })

    minutes_per_round = 30
    total_hours = (total_rounds * minutes_per_round) / 60

    config = {
        "simulation_id": f"sim_{uuid.uuid4().hex[:8]}",
        "idea_description": idea_description,
        "agent_configs": agent_configs,
        "event_config": {
            "initial_posts": initial_posts,
        },
        "time_config": {
            "total_simulation_hours": int(total_hours),
            "minutes_per_round": minutes_per_round,
            "peak_hours": [9, 10, 11, 14, 15, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "peak_activity_multiplier": 1.5,
            "off_peak_activity_multiplier": 0.3,
            "agents_per_hour_min": max(3, len(personas) // 3),
            "agents_per_hour_max": len(personas),
        },
        "llm_model": settings.GEMINI_MODEL_NAME,
    }

    return config


# ── OASIS Simulation Runner ──────────────────────────────

class SimulationRunner:
    """
    Runs OASIS Twitter + Reddit social media simulation.

    The OASIS engine creates a simulated social media environment where
    AI agents (powered by LLMs) post, like, repost, comment, and interact
    on Twitter and Reddit platforms.
    """

    def __init__(self, llm=None):
        self.llm = llm  # Kept for compatibility, OASIS uses its own model
        self._oasis_available = False
        self._check_oasis()

    def _check_oasis(self):
        """Check if OASIS is installed."""
        try:
            import oasis
            from camel.models import ModelFactory
            self._oasis_available = True
            logger.info("OASIS simulation engine available")
        except ImportError:
            self._oasis_available = False
            logger.warning("OASIS not installed. Install with: pip install oasis-ai camel-ai[all]")

    def run(
        self,
        graph: KnowledgeGraph,
        personas: list[AgentPersona],
        idea_description: str,
        total_rounds: int = 15,
        progress_callback: Optional[Callable] = None,
    ) -> list[RoundSummary]:
        """
        Run the OASIS simulation.

        If OASIS is installed: runs full Twitter+Reddit social media simulation.
        If not: falls back to a lightweight LLM-based simulation.
        """
        if self._oasis_available:
            return asyncio.run(self._run_oasis(
                graph, personas, idea_description, total_rounds, progress_callback
            ))
        else:
            logger.warning("OASIS not available — running lightweight LLM simulation")
            return self._run_lightweight(
                graph, personas, idea_description, total_rounds, progress_callback
            )

    async def _run_oasis(
        self,
        graph: KnowledgeGraph,
        personas: list[AgentPersona],
        idea_description: str,
        total_rounds: int,
        progress_callback: Optional[Callable],
    ) -> list[RoundSummary]:
        """Run full OASIS Twitter + Reddit simulation."""
        import oasis
        from oasis import (
            ActionType, LLMAction, ManualAction,
            generate_twitter_agent_graph, generate_reddit_agent_graph,
        )
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType

        logger.info(f"Starting OASIS simulation: {len(personas)} agents, {total_rounds} rounds")

        # Create simulation directory
        sim_id = f"sim_{uuid.uuid4().hex[:8]}"
        sim_dir = os.path.join(settings.DATA_DIR, "simulations", sim_id)
        os.makedirs(sim_dir, exist_ok=True)

        # Write profiles
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        write_twitter_profiles(personas, twitter_profile_path)
        write_reddit_profiles(personas, reddit_profile_path)

        # Generate simulation config
        config = generate_simulation_config(personas, idea_description, total_rounds)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Create LLM model for OASIS — route to the selected provider
        provider = self.llm.provider if self.llm else "gemini"
        logger.info(f"OASIS using LLM provider: {provider}")

        if provider == "ollama":
            # Ollama exposes an OpenAI-compatible API at /v1
            ollama_host = settings.OLLAMA_HOST or "http://localhost:11434"
            os.environ["OPENAI_API_KEY"] = "ollama"  # Ollama doesn't need a real key
            os.environ["OPENAI_API_BASE_URL"] = f"{ollama_host}/v1"
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=settings.OLLAMA_MODEL_NAME,
            )
        elif provider == "groq":
            os.environ["OPENAI_API_KEY"] = settings.GROQ_API_KEY
            os.environ["OPENAI_API_BASE_URL"] = "https://api.groq.com/openai/v1"
            # Use a standard Groq model (compound models don't work via OpenAI compat)
            groq_model = settings.GROQ_MODEL_NAME
            if "compound" in groq_model:
                groq_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=groq_model,
            )
        else:
            # Default: Gemini via OpenAI-compatible interface
            os.environ["OPENAI_API_KEY"] = settings.GEMINI_API_KEY
            os.environ["OPENAI_API_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=settings.GEMINI_MODEL_NAME,
            )

        # Agent name mapping
        agent_names = {i: p.name for i, p in enumerate(personas)}

        # Setup loggers
        log_manager = SimulationLogManager(sim_dir)
        twitter_logger = log_manager.get_twitter_logger()
        reddit_logger = log_manager.get_reddit_logger()

        if progress_callback:
            progress_callback(0, total_rounds, "Initializing OASIS environments...")

        # Available actions
        twitter_actions = [
            ActionType.CREATE_POST, ActionType.LIKE_POST, ActionType.REPOST,
            ActionType.FOLLOW, ActionType.DO_NOTHING, ActionType.QUOTE_POST,
        ]
        reddit_actions = [
            ActionType.LIKE_POST, ActionType.DISLIKE_POST, ActionType.CREATE_POST,
            ActionType.CREATE_COMMENT, ActionType.LIKE_COMMENT, ActionType.DISLIKE_COMMENT,
            ActionType.SEARCH_POSTS, ActionType.DO_NOTHING, ActionType.FOLLOW,
        ]

        # Initialize environments
        twitter_agent_graph = await generate_twitter_agent_graph(
            profile_path=twitter_profile_path,
            model=model,
            available_actions=twitter_actions,
        )
        reddit_agent_graph = await generate_reddit_agent_graph(
            profile_path=reddit_profile_path,
            model=model,
            available_actions=reddit_actions,
        )

        twitter_db = os.path.join(sim_dir, "twitter_simulation.db")
        reddit_db = os.path.join(sim_dir, "reddit_simulation.db")

        twitter_env = oasis.make(
            agent_graph=twitter_agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=twitter_db,
            semaphore=30,
        )
        reddit_env = oasis.make(
            agent_graph=reddit_agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=reddit_db,
            semaphore=30,
        )

        await twitter_env.reset()
        await reddit_env.reset()

        if progress_callback:
            progress_callback(0, total_rounds, "OASIS environments ready — injecting initial events")

        # Round 0: Initial posts
        round_summaries = []
        round0 = RoundSummary(round_number=0)

        initial_posts = config["event_config"]["initial_posts"]
        twitter_initial = {}
        reddit_initial = {}

        for post in initial_posts:
            agent_id = post["poster_agent_id"]
            content = post["content"]
            try:
                t_agent = twitter_env.agent_graph.get_agent(agent_id)
                twitter_initial[t_agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content},
                )
            except Exception:
                pass
            try:
                r_agent = reddit_env.agent_graph.get_agent(agent_id)
                reddit_initial[r_agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content},
                )
            except Exception:
                pass

            round0.actions.append(AgentAction(
                agent_id=agent_id,
                agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                round_number=0,
                action_type="CREATE_POST",
                content=content,
                platform="both",
            ))

        if twitter_initial:
            await twitter_env.step(twitter_initial)
        if reddit_initial:
            await reddit_env.step(reddit_initial)

        round0.key_events = [f"Breaking news posted: {idea_description[:100]}..."]
        round0.market_sentiment = "neutral"
        round_summaries.append(round0)

        # Main simulation loop
        twitter_last_rowid = 0
        reddit_last_rowid = 0

        time_config = config["time_config"]
        minutes_per_round = time_config["minutes_per_round"]

        for round_num in range(1, total_rounds + 1):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24

            # Get active agents
            twitter_active = get_active_agents_for_round(
                twitter_env, config, simulated_hour, round_num
            )
            reddit_active = get_active_agents_for_round(
                reddit_env, config, simulated_hour, round_num
            )

            # Execute actions
            if twitter_active:
                t_actions = {agent: LLMAction() for _, agent in twitter_active}
                await twitter_env.step(t_actions)

            if reddit_active:
                r_actions = {agent: LLMAction() for _, agent in reddit_active}
                await reddit_env.step(r_actions)

            # Fetch actions from databases
            round_summary = RoundSummary(round_number=round_num)

            t_actions_data, twitter_last_rowid = fetch_new_actions_from_db(
                twitter_db, twitter_last_rowid, agent_names
            )
            r_actions_data, reddit_last_rowid = fetch_new_actions_from_db(
                reddit_db, reddit_last_rowid, agent_names
            )

            for ad in t_actions_data:
                content = ad['action_args'].get('content', '')
                action = AgentAction(
                    agent_id=ad['agent_id'],
                    agent_name=ad['agent_name'],
                    round_number=round_num,
                    action_type=ad['action_type'],
                    content=content,
                    platform="twitter",
                )
                round_summary.actions.append(action)

                # Log to twitter action logger
                twitter_logger.log_action(
                    round_num=round_num,
                    agent_id=ad['agent_id'],
                    agent_name=ad['agent_name'],
                    action_type=ad['action_type'],
                    action_args=ad['action_args'],
                )

                # Update graph memory
                if content:
                    graph.add_memory(
                        str(ad['agent_id']),
                        f"[Twitter R{round_num}] {ad['action_type']}: {content[:200]}",
                        round_number=round_num,
                    )

            for ad in r_actions_data:
                content = ad['action_args'].get('content', '')
                action = AgentAction(
                    agent_id=ad['agent_id'],
                    agent_name=ad['agent_name'],
                    round_number=round_num,
                    action_type=ad['action_type'],
                    content=content,
                    platform="reddit",
                )
                round_summary.actions.append(action)

                reddit_logger.log_action(
                    round_num=round_num,
                    agent_id=ad['agent_id'],
                    agent_name=ad['agent_name'],
                    action_type=ad['action_type'],
                    action_args=ad['action_args'],
                )

                if content:
                    graph.add_memory(
                        str(ad['agent_id']),
                        f"[Reddit R{round_num}] {ad['action_type']}: {content[:200]}",
                        round_number=round_num,
                    )

            # Extract key events (posts with actual content)
            round_summary.key_events = [
                f"{a.agent_name} ({a.platform}): {a.content[:100]}"
                for a in round_summary.actions
                if a.action_type == "CREATE_POST" and a.content
            ][:5]
            round_summary.market_sentiment = self._analyze_sentiment(round_summary.actions)
            round_summaries.append(round_summary)

            if progress_callback:
                progress_callback(
                    round_num, total_rounds,
                    f"Round {round_num}: {len(round_summary.actions)} actions "
                    f"(T:{len(t_actions_data)}, R:{len(r_actions_data)}), "
                    f"sentiment={round_summary.market_sentiment}"
                )

        # Close environments
        await twitter_env.close()
        await reddit_env.close()

        twitter_logger.log_simulation_end(total_rounds, sum(
            len(r.actions) for r in round_summaries if any(a.platform == "twitter" for a in r.actions)
        ))
        reddit_logger.log_simulation_end(total_rounds, sum(
            len(r.actions) for r in round_summaries if any(a.platform == "reddit" for a in r.actions)
        ))

        logger.info(f"OASIS simulation complete: {len(round_summaries)} rounds, "
                     f"{sum(len(r.actions) for r in round_summaries)} total actions")

        return round_summaries

    # ── Lightweight Fallback (no OASIS) ──────────────────

    def _run_lightweight(
        self,
        graph: KnowledgeGraph,
        personas: list[AgentPersona],
        idea_description: str,
        total_rounds: int,
        progress_callback: Optional[Callable],
    ) -> list[RoundSummary]:
        """Fallback LLM-based simulation when OASIS is not installed.
        
        Uses batched rounds (3 per LLM call) for speed (Optimization #4).
        """
        from engine.llm_client import LLMClient
        llm = self.llm or LLMClient()

        BATCH_SIZE = 3
        logger.info(f"Starting lightweight simulation: {len(personas)} agents, {total_rounds} rounds (batch={BATCH_SIZE})")
        round_summaries = []

        # Round 0: Inject event
        round0 = self._inject_event(llm, graph, personas, idea_description)
        round_summaries.append(round0)
        if progress_callback:
            progress_callback(0, total_rounds, "Event injected — agents reacting")

        # Process remaining rounds in batches of BATCH_SIZE
        round_num = 1
        while round_num <= total_rounds:
            batch_end = min(round_num + BATCH_SIZE - 1, total_rounds)
            batch_rounds = list(range(round_num, batch_end + 1))
            
            batch_summaries = self._run_lightweight_batch(
                llm, graph, personas, idea_description, batch_rounds, round_summaries
            )
            round_summaries.extend(batch_summaries)
            
            if progress_callback:
                progress_callback(
                    batch_end, total_rounds,
                    f"Rounds {round_num}-{batch_end}: "
                    f"{sum(len(s.actions) for s in batch_summaries)} actions, "
                    f"sentiment={batch_summaries[-1].market_sentiment}"
                )
            
            round_num = batch_end + 1

        return round_summaries

    def _inject_event(self, llm, graph, personas, idea):
        """Inject breaking news and get initial reactions."""
        summary = RoundSummary(round_number=0)

        agent_list = "\n".join(f"- {p.name} ({p.role}): {p.initial_stance}" for p in personas)
        prompt = f"""You are simulating a market. Breaking news:

BREAKING NEWS: "{idea}"

Market participants:
{agent_list}

For EACH agent, generate their immediate reaction. Return JSON:
{{"reactions": [{{"agent": "Name", "reaction": "1-2 sentences", "sentiment": "positive/negative/neutral"}}]}}"""

        try:
            reactions = llm.chat_json(prompt, temperature=0.8, max_tokens=4096)
            reaction_list = reactions.get("reactions", reactions) if isinstance(reactions, dict) else reactions
        except Exception as e:
            logger.warning(f"Batch reaction failed: {e}")
            reaction_list = []

        for rx in (reaction_list if isinstance(reaction_list, list) else []):
            agent_name = rx.get("agent", "Unknown")
            persona = next((p for p in personas if p.name == agent_name), None)
            action = AgentAction(
                agent_id=personas.index(persona) if persona else -1,
                agent_name=agent_name,
                round_number=0,
                action_type="react",
                content=rx.get("reaction", ""),
                sentiment=rx.get("sentiment", "neutral"),
                platform="simulation",
            )
            summary.actions.append(action)
            if persona:
                graph.add_memory(persona.id, f"[R0] {rx.get('reaction', '')}", round_number=0)

        summary.key_events = [f"Breaking news: {idea[:100]}..."]
        summary.market_sentiment = self._analyze_sentiment(summary.actions)
        return summary

    def _run_lightweight_round(self, llm, graph, personas, idea, round_num, history):
        """Run a single lightweight simulation round."""
        summary = RoundSummary(round_number=round_num)

        recent = self._compact_history(history[-3:])
        agent_list = "\n".join(
            f"- {p.name} ({p.role}): Goals={', '.join(p.goals[:2])}" for p in personas
        )

        prompt = f"""Simulating Round {round_num}. IDEA: "{idea}"

AGENTS:
{agent_list}

RECENT HISTORY:
{recent}

For EACH agent, decide what they do. Actions: react, converse, decide, do_nothing.
Return JSON:
{{"actions": [{{"agent": "Name", "action_type": "react|converse|decide|do_nothing", "content": "1-2 sentences", "target": "", "sentiment": "positive/negative/neutral"}}]}}

Rules: Not all agents act every round. Be realistic. 40%+ should take action."""

        try:
            result = llm.chat_json(prompt, temperature=0.8, max_tokens=4000)
            actions_data = result.get("actions", result) if isinstance(result, dict) else result
        except Exception as e:
            logger.warning(f"Round {round_num} failed: {e}")
            actions_data = []

        for ad in (actions_data if isinstance(actions_data, list) else []):
            agent_name = ad.get("agent", "Unknown")
            persona = next((p for p in personas if p.name == agent_name), None)
            action = AgentAction(
                agent_id=personas.index(persona) if persona else -1,
                agent_name=agent_name,
                round_number=round_num,
                action_type=ad.get("action_type", "react"),
                content=ad.get("content", ""),
                target_agent=ad.get("target", ""),
                sentiment=ad.get("sentiment", "neutral"),
                platform="simulation",
            )
            summary.actions.append(action)
            if persona:
                graph.add_memory(persona.id, f"[R{round_num}] {action.content}", round_number=round_num)

            if action.target_agent and persona:
                target = next((p for p in personas if p.name == action.target_agent), None)
                if target:
                    rel = Relationship(
                        id=str(uuid.uuid4())[:8],
                        source_id=persona.id, target_id=target.id,
                        relation_type="interacted_with",
                        fact=f"R{round_num}: {persona.name} → {target.name}: {action.content[:100]}",
                        round_created=round_num,
                    )
                    graph.add_relationship(rel)

        summary.key_events = [a.content for a in summary.actions if a.action_type in ("decide", "react")][:5]
        summary.market_sentiment = self._analyze_sentiment(summary.actions)
        return summary

    def _run_lightweight_batch(self, llm, graph, personas, idea, batch_rounds, history):
        """Simulate multiple rounds in a single LLM call (Optimization #4)."""
        recent = self._compact_history(history[-3:])
        agent_list = "\n".join(
            f"- {p.name} ({p.role}): Goals={', '.join(p.goals[:2])}" for p in personas
        )
        round_nums_str = ", ".join(str(r) for r in batch_rounds)

        prompt = f"""Simulating Rounds {round_nums_str}. IDEA: "{idea}"

AGENTS:
{agent_list}

RECENT HISTORY:
{recent}

Simulate {len(batch_rounds)} sequential rounds. Each round builds on the previous.
Actions: react, converse, decide, do_nothing.
Rules: Not all agents act every round. Be realistic. 40%+ should take action per round.

Return JSON with one key per round:
{{
    "round_{batch_rounds[0]}": {{
        "actions": [{{"agent": "Name", "action_type": "react|converse|decide|do_nothing", "content": "1-2 sentences", "target": "", "sentiment": "positive/negative/neutral"}}]
    }},
    "round_{batch_rounds[-1]}": {{
        "actions": [...]
    }}
}}"""

        try:
            result = llm.chat_json(prompt, temperature=0.8, max_tokens=6000)
        except Exception as e:
            logger.warning(f"Batch round {round_nums_str} failed: {e}, falling back to single rounds")
            summaries = []
            for rn in batch_rounds:
                s = self._run_lightweight_round(llm, graph, personas, idea, rn, history + summaries)
                summaries.append(s)
            return summaries

        # Parse batched result
        summaries = []
        for round_num in batch_rounds:
            summary = RoundSummary(round_number=round_num)
            
            # Try multiple key formats
            round_data = (
                result.get(f"round_{round_num}")
                or result.get(str(round_num))
                or result.get(f"Round {round_num}")
                or {}
            )
            actions_data = round_data.get("actions", []) if isinstance(round_data, dict) else []

            for ad in (actions_data if isinstance(actions_data, list) else []):
                agent_name = ad.get("agent", "Unknown")
                persona = next((p for p in personas if p.name == agent_name), None)
                action = AgentAction(
                    agent_id=personas.index(persona) if persona else -1,
                    agent_name=agent_name,
                    round_number=round_num,
                    action_type=ad.get("action_type", "react"),
                    content=ad.get("content", ""),
                    target_agent=ad.get("target", ""),
                    sentiment=ad.get("sentiment", "neutral"),
                    platform="simulation",
                )
                summary.actions.append(action)
                if persona:
                    graph.add_memory(persona.id, f"[R{round_num}] {action.content}", round_number=round_num)

                if action.target_agent and persona:
                    target = next((p for p in personas if p.name == action.target_agent), None)
                    if target:
                        rel = Relationship(
                            id=str(uuid.uuid4())[:8],
                            source_id=persona.id, target_id=target.id,
                            relation_type="interacted_with",
                            fact=f"R{round_num}: {persona.name} → {target.name}: {action.content[:100]}",
                            round_created=round_num,
                        )
                        graph.add_relationship(rel)

            summary.key_events = [a.content for a in summary.actions if a.action_type in ("decide", "react")][:5]
            summary.market_sentiment = self._analyze_sentiment(summary.actions)
            summaries.append(summary)

        return summaries

    # ── Utilities ────────────────────────────────────────

    @staticmethod
    def _analyze_sentiment(actions: list[AgentAction]) -> str:
        """Compute overall sentiment from actions."""
        if not actions:
            return "neutral"
        sentiments = [a.sentiment for a in actions if a.sentiment]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        if pos > neg * 1.5:
            return "positive"
        elif neg > pos * 1.5:
            return "negative"
        return "mixed"

    @staticmethod
    def _compact_history(summaries: list[RoundSummary]) -> str:
        if not summaries:
            return "No prior history."
        parts = []
        for s in summaries:
            actions_text = "; ".join(f"{a.agent_name}: {a.content[:60]}" for a in s.actions[:6])
            parts.append(f"Round {s.round_number} ({s.market_sentiment}): {actions_text}")
        return "\n".join(parts)
