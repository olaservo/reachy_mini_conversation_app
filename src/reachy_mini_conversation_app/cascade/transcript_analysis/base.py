"""Base abstractions for transcript analysis."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable
from dataclasses import field, dataclass


@dataclass
class EntityMatch:
    """A single entity matched in transcript text."""

    text: str
    label: str
    confidence: float


@dataclass
class TriggerMatch:
    """What actually matched in the transcript, passed to callbacks."""

    words: list[str] = field(default_factory=list)
    entities: list[EntityMatch] = field(default_factory=list)


@dataclass
class TriggerConfig:
    """Trigger definition from YAML: which words/entities activate this reaction.

    Use `all` for boolean AND: every sub-trigger must match for the reaction to fire.
    """

    words: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    all: list[TriggerConfig] = field(default_factory=list)


@dataclass
class ReactionConfig:
    """A fully resolved reaction entry from YAML."""

    name: str
    callback: Callable[..., Awaitable[None]]
    trigger: TriggerConfig
    params: dict[str, Any] = field(default_factory=dict)
    repeatable: bool = False


class TranscriptAnalyzer(ABC):
    """Abstract base class for transcript analyzers."""

    @abstractmethod
    async def analyze(self, text: str, is_final: bool) -> Any:
        """Analyze transcript text and return matches."""

    def reset(self) -> None:
        """Reset analyzer state between conversations."""
