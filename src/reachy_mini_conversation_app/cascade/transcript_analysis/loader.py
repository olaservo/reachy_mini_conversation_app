"""Load transcript reaction configurations from the active profile.

A profile opts into live reactions by adding a ``reactions.yaml`` and callback
modules alongside its ``instructions.txt``. Profiles without one simply disable
transcript analysis (the manager falls back to a no-op).
"""

from __future__ import annotations
import logging
import importlib.util
from typing import Any, Callable, Optional
from pathlib import Path

import yaml

from .base import TriggerConfig, ReactionConfig
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)


def get_profile_reactions() -> list[ReactionConfig] | None:
    """Load reactions from the active profile's reactions.yaml.

    Returns:
        List of ReactionConfig, or None if no profile or no reactions.yaml.

    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.debug("No profile set, transcript reactions disabled")
        return None

    profile_dir = config.PROFILES_DIRECTORY / profile
    reactions_file = profile_dir / "reactions.yaml"
    if not reactions_file.exists():
        logger.debug(f"No reactions.yaml in profile '{profile}'")
        return None

    try:
        with open(reactions_file) as f:
            yaml_config = yaml.safe_load(f)

        if not yaml_config or not isinstance(yaml_config, list):
            return None

        reactions: list[ReactionConfig] = []

        for entry in yaml_config:
            name = entry.get("name")
            callback_name = entry.get("callback")
            if not name or not callback_name:
                logger.warning(f"Skipping reaction entry missing name or callback: {entry}")
                continue

            callback = _import_callback(profile_dir, callback_name)
            if not callback:
                continue

            trigger = _parse_trigger(entry.get("trigger", {}))
            reactions.append(
                ReactionConfig(
                    name=name,
                    callback=callback,
                    trigger=trigger,
                    params=entry.get("params", {}),
                    repeatable=entry.get("repeatable", False),
                )
            )

        logger.info(f"Loaded {len(reactions)} reactions from profile '{profile}'")
        return reactions if reactions else None

    except Exception as e:
        logger.warning(f"Failed to load reactions from profile '{profile}': {e}")
        return None


def _parse_trigger(raw: dict[str, Any]) -> TriggerConfig:
    """Parse a trigger dict from YAML into a TriggerConfig."""
    all_groups = [
        TriggerConfig(words=group.get("words", []), entities=group.get("entities", []))
        for group in raw.get("all", [])
    ]
    return TriggerConfig(
        words=raw.get("words", []),
        entities=raw.get("entities", []),
        all=all_groups,
    )


def _import_callback(profile_dir: Path, callback_name: str) -> Optional[Callable[..., Any]]:
    """Import a reaction callback from ``<profile_dir>/<callback_name>.py``."""
    callback_file = profile_dir / f"{callback_name}.py"
    if not callback_file.is_file():
        logger.warning(f"Reaction callback file not found: {callback_file}")
        return None
    try:
        module_name = f"_cascade_reaction_{profile_dir.name}_{callback_name}"
        spec = importlib.util.spec_from_file_location(module_name, callback_file)
        if not (spec and spec.loader):
            logger.warning(f"Cannot load callback module from {callback_file}")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, callback_name, None)
    except Exception as e:
        logger.warning(f"Failed to import callback '{callback_name}' from {callback_file}: {e}")
        return None
