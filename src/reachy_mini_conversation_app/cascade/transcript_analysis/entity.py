"""Named Entity Recognition (NER) based transcript analyzer using GLiNER."""

from __future__ import annotations
import asyncio
import logging

from .base import EntityMatch, TranscriptAnalyzer


logger = logging.getLogger(__name__)


class EntityAnalyzer(TranscriptAnalyzer):
    """Pure entity matcher using GLiNER. Returns list[EntityMatch].

    No callbacks, no deduplication — that's the manager's job.
    """

    def __init__(self, entity_labels: list[str]):
        """Initialize entity analyzer.

        Args:
            entity_labels: Entity labels to detect (e.g. ["food", "person"])

        """
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError(
                "GLiNER not installed. Install with: pip install 'reachy_mini_conversation_app[cascade_gliner]'"
            )

        from reachy_mini_conversation_app.cascade.config import get_config

        self.entity_labels = entity_labels
        self.model_name = get_config().gliner_model

        logger.info(f"Loading GLiNER model: {self.model_name}")
        self.model = GLiNER.from_pretrained(self.model_name)
        logger.info(f"EntityAnalyzer initialized: {len(entity_labels)} entity types")

    async def analyze(self, text: str, is_final: bool) -> list[EntityMatch]:
        """Return list of EntityMatch for entities found in text."""
        import time

        start_time = time.time()
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            None, lambda: self.model.predict_entities(text, self.entity_labels)
        )
        elapsed = time.time() - start_time

        logger.info(f"GLiNER analyzed '{text[:50]}...' in {elapsed * 1000:.0f}ms, found {len(entities)} entities")

        return [
            EntityMatch(
                text=e["text"],
                label=e["label"],
                confidence=e["score"],
            )
            for e in entities
        ]
