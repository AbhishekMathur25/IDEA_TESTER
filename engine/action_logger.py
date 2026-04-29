"""
Idea Tester — OASIS Simulation Action Logger
Tracks agent actions from OASIS Twitter/Reddit simulations.
Port of MiroFish's action_logger.py.
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("idea_tester.action_logger")


class PlatformActionLogger:
    """Logs agent actions to a JSONL file for a specific platform."""

    def __init__(self, log_dir: str, platform: str):
        """
        Args:
            log_dir: Directory to store the log file
            platform: "twitter" or "reddit"
        """
        self.platform = platform
        self.log_dir = os.path.join(log_dir, platform)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "actions.jsonl")

        # Clear previous log
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def _write(self, entry: dict):
        """Write a JSON entry to the log file."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_simulation_start(self, config: dict):
        self._write({
            "event": "simulation_start",
            "platform": self.platform,
            "agent_count": len(config.get("agent_configs", [])),
        })

    def log_simulation_end(self, total_rounds: int, total_actions: int):
        self._write({
            "event": "simulation_end",
            "platform": self.platform,
            "total_rounds": total_rounds,
            "total_actions": total_actions,
        })

    def log_round_start(self, round_num: int, simulated_hour: int):
        self._write({
            "event": "round_start",
            "round": round_num,
            "simulated_hour": simulated_hour,
        })

    def log_round_end(self, round_num: int, actions_count: int):
        self._write({
            "event": "round_end",
            "round": round_num,
            "actions_count": actions_count,
        })

    def log_action(
        self,
        round_num: int,
        agent_id: int,
        agent_name: str,
        action_type: str,
        action_args: dict,
    ):
        self._write({
            "event": "action",
            "round": round_num,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "action_type": action_type,
            "action_args": action_args,
        })


class SimulationLogManager:
    """Manages logging for the overall simulation process."""

    def __init__(self, simulation_dir: str):
        self.simulation_dir = simulation_dir
        os.makedirs(simulation_dir, exist_ok=True)

        self.log_path = os.path.join(simulation_dir, "simulation.log")
        self._logger = logging.getLogger(f"sim.{os.path.basename(simulation_dir)}")

        # File handler
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def info(self, msg: str):
        self._logger.info(msg)
        print(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def get_twitter_logger(self) -> PlatformActionLogger:
        return PlatformActionLogger(self.simulation_dir, "twitter")

    def get_reddit_logger(self) -> PlatformActionLogger:
        return PlatformActionLogger(self.simulation_dir, "reddit")
