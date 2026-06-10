"""Manager for orchestrating transcript analyzers and executing reactions."""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Any, List

from .base import EntityMatch, TriggerMatch, ReactionConfig
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


class TranscriptAnalysisManager:
    """Orchestrates transcript analyzers and dispatches reaction callbacks.

    Owns:
    - Analyzer creation (partitions ReactionConfigs by trigger type)
    - Per-reaction deduplication (each reaction fires at most once per turn)
    - Callback dispatch with TriggerMatch + params

    """

    DEBOUNCE_INTERVAL = 0.4  # minimum seconds between partial analyses

    def __init__(self, reactions: list[ReactionConfig], deps: ToolDependencies):
        """Initialize with reaction configs and tool dependencies."""
        self.deps = deps
        self.reactions = {r.name: r for r in reactions}
        self._pending_tasks: List[asyncio.Task[Any]] = []
        self._last_analysis_time = 0.0
        self.triggered_reactions: set[str] = set()
        # For repeatable entity reactions: track (reaction_name, entity_text) already fired
        self._triggered_entity_keys: set[tuple[str, str]] = set()

        # Partition reactions by trigger type and build analyzers
        # _all_reaction_groups maps real name → list of synthetic group names
        self._all_reaction_groups: dict[str, list[str]] = {}
        self.keyword_analyzer = self._build_keyword_analyzer(reactions)
        self.entity_analyzer = self._build_entity_analyzer(reactions)

        # Map entity label → reaction names for dispatch
        self.entity_reaction_map: dict[str, list[str]] = {}
        for r in reactions:
            for label in r.trigger.entities:
                self.entity_reaction_map.setdefault(label, []).append(r.name)

        logger.info(f"TranscriptAnalysisManager initialized with {len(reactions)} reactions")

    def _build_keyword_analyzer(self, reactions: list[ReactionConfig]) -> Any | None:
        """Build KeywordAnalyzer from reactions with word triggers."""
        from .keyword import KeywordAnalyzer

        reaction_words: dict[str, list[str]] = {}
        for r in reactions:
            # Simple word triggers
            if r.trigger.words:
                reaction_words[r.name] = r.trigger.words

            # Boolean `all` triggers: register each group as a synthetic entry
            if r.trigger.all:
                group_names: list[str] = []
                for i, sub_trigger in enumerate(r.trigger.all):
                    if sub_trigger.words:
                        synthetic_name = f"{r.name}__all_{i}"
                        reaction_words[synthetic_name] = sub_trigger.words
                        group_names.append(synthetic_name)
                if group_names:
                    self._all_reaction_groups[r.name] = group_names

        if not reaction_words:
            return None
        return KeywordAnalyzer(reaction_words)

    def _build_entity_analyzer(self, reactions: list[ReactionConfig]) -> Any | None:
        """Build EntityAnalyzer from reactions with entity triggers."""
        all_labels: set[str] = set()
        for r in reactions:
            all_labels.update(r.trigger.entities)

        if not all_labels:
            return None

        try:
            from .entity import EntityAnalyzer

            return EntityAnalyzer(sorted(all_labels))
        except ImportError:
            logger.warning(
                "GLiNER not installed, skipping entity analyzer. "
                "Install with: pip install 'reachy_mini_conversation_app[cascade_gliner]'"
            )
            return None

    async def analyze_partial(self, text: str) -> None:
        """Analyze partial transcript with debouncing. Fire-and-forget."""
        current_time = time.time()
        if current_time - self._last_analysis_time < self.DEBOUNCE_INTERVAL:
            return

        self._last_analysis_time = current_time
        logger.info(f"Analyzing PARTIAL transcript: '{text[:100]}...' ({len(text)} chars)")

        task = asyncio.create_task(self._analyze_and_dispatch(text, is_final=False))
        self._pending_tasks.append(task)

    async def analyze_final(self, text: str) -> None:
        """Analyze final transcript. Blocks until analysis completes."""
        logger.info(f"Analyzing FINAL transcript: '{text[:100]}...' ({len(text)} chars)")
        await self._analyze_and_dispatch(text, is_final=True)

    async def _analyze_and_dispatch(self, text: str, is_final: bool) -> None:
        """Run analyzers and dispatch triggered reactions."""
        try:
            # Run keyword and entity analyzers in parallel
            tasks = []
            if self.keyword_analyzer:
                tasks.append(self.keyword_analyzer.analyze(text, is_final))
            if self.entity_analyzer:
                tasks.append(self.entity_analyzer.analyze(text, is_final))

            if not tasks:
                return

            results = await asyncio.gather(*tasks, return_exceptions=True)
            idx = 0

            # Process keyword results
            keyword_matches: dict[str, list[str]] = {}
            if self.keyword_analyzer:
                kw_result = results[idx]
                if isinstance(kw_result, BaseException):
                    logger.warning(f"Keyword analyzer error: {kw_result}")
                else:
                    keyword_matches = kw_result
                idx += 1

            # Process entity results
            entity_matches: list[EntityMatch] = []
            if self.entity_analyzer:
                ent_result = results[idx]
                if isinstance(ent_result, BaseException):
                    logger.warning(f"Entity analyzer error: {ent_result}")
                else:
                    entity_matches = ent_result

            # Evaluate boolean `all` triggers: merge synthetic groups into real reactions
            for real_name, group_names in self._all_reaction_groups.items():
                real_reaction = self.reactions[real_name]
                if not real_reaction.repeatable and real_name in self.triggered_reactions:
                    # Already fired — just strip synthetic entries
                    for g in group_names:
                        keyword_matches.pop(g, None)
                    continue
                if all(g in keyword_matches for g in group_names):
                    merged_words: list[str] = []
                    for g in group_names:
                        merged_words.extend(keyword_matches.pop(g))
                    keyword_matches[real_name] = merged_words
                else:
                    # Not all groups matched — strip synthetic entries
                    for g in group_names:
                        keyword_matches.pop(g, None)

            # Dispatch keyword-triggered reactions
            for reaction_name, matched_words in keyword_matches.items():
                if reaction_name in self.triggered_reactions:
                    continue

                reaction = self.reactions[reaction_name]
                if not reaction.repeatable:
                    self.triggered_reactions.add(reaction_name)
                match = TriggerMatch(words=matched_words)
                logger.info(f"Reaction triggered: {reaction_name} (words: {matched_words})")
                asyncio.create_task(self._execute(reaction, match))

            # Dispatch entity-triggered reactions
            for em in entity_matches:
                for reaction_name in self.entity_reaction_map.get(em.label, []):
                    if reaction_name in self.triggered_reactions:
                        continue

                    reaction = self.reactions[reaction_name]
                    if reaction.repeatable:
                        # Deduplicate by (reaction, entity_text) so each unique entity fires once
                        entity_key = (reaction_name, em.text.lower())
                        if entity_key in self._triggered_entity_keys:
                            continue
                        self._triggered_entity_keys.add(entity_key)
                    else:
                        self.triggered_reactions.add(reaction_name)
                    match = TriggerMatch(entities=[em])
                    logger.info(f"Reaction triggered: {reaction_name} (entity: {em.text} [{em.label}])")
                    asyncio.create_task(self._execute(reaction, match))

        except Exception as e:
            logger.exception(f"Error in transcript analysis: {e}")

    async def _execute(self, reaction: ReactionConfig, match: TriggerMatch) -> None:
        """Execute a reaction callback with deps, match, and params."""
        try:
            await reaction.callback(self.deps, match, **reaction.params)
        except Exception as e:
            logger.exception(f"Error executing reaction '{reaction.name}': {e}")

    def reset(self) -> None:
        """Reset deduplication state for next conversation turn."""
        logger.debug(f"Resetting manager ({len(self.triggered_reactions)} reactions triggered)")
        self.triggered_reactions.clear()
        self._triggered_entity_keys.clear()
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()
        self._last_analysis_time = 0.0


class NoOpTranscriptManager:
    """No-op manager when transcript analysis is disabled."""

    async def analyze_partial(self, text: str) -> None:
        """No-op."""

    async def analyze_final(self, text: str) -> None:
        """No-op."""

    def reset(self) -> None:
        """No-op."""
