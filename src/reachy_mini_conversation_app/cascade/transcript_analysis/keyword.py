"""Keyword-based transcript analyzer."""

from __future__ import annotations
import logging
from fnmatch import fnmatch

from .base import TranscriptAnalyzer


logger = logging.getLogger(__name__)


def _is_glob(pattern: str) -> bool:
    """Return True if pattern contains wildcard characters."""
    return "*" in pattern or "?" in pattern


class KeywordAnalyzer(TranscriptAnalyzer):
    """Pure keyword matcher. Returns reaction_name → matched words.

    Supports literal substring matching and glob patterns (e.g. "danc*").
    No callbacks, no deduplication — that's the manager's job.
    """

    def __init__(self, reaction_words: dict[str, list[str]]):
        """Initialize keyword analyzer.

        Args:
            reaction_words: Dict mapping reaction_name to list of trigger words.
                Words containing * or ? are treated as glob patterns matched
                against individual tokens; plain words use substring matching.

        """
        self.literal_words: dict[str, list[str]] = {}
        self.glob_patterns: dict[str, list[str]] = {}

        for name, words in reaction_words.items():
            literals = []
            globs = []
            for w in words:
                w_lower = w.lower()
                if _is_glob(w_lower):
                    globs.append(w_lower)
                else:
                    literals.append(w_lower)
            if literals:
                self.literal_words[name] = literals
            if globs:
                self.glob_patterns[name] = globs

        total = sum(len(ws) for ws in reaction_words.values())
        logger.info(f"KeywordAnalyzer initialized: {total} words across {len(reaction_words)} reactions")

    async def analyze(self, text: str, is_final: bool) -> dict[str, list[str]]:
        """Return {reaction_name: [matched_words]} for words found in text."""
        text_lower = text.lower()
        # Tokenize once, only if there are glob patterns
        tokens = text_lower.split() if self.glob_patterns else []

        matches: dict[str, list[str]] = {}
        all_names = set(self.literal_words) | set(self.glob_patterns)

        for name in all_names:
            matched: list[str] = []

            # Literal substring matching
            for w in self.literal_words.get(name, []):
                if w in text_lower:
                    matched.append(w)

            # Glob pattern matching against tokens
            for pattern in self.glob_patterns.get(name, []):
                for token in tokens:
                    if fnmatch(token, pattern):
                        matched.append(token)
                        break  # one match per pattern is enough

            if matched:
                matches[name] = matched

        return matches
