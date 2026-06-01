"""Background dream runner.

Runs ``Dreamer.run()`` on a daemon thread during the conversation so memory
consolidation never blocks startup. The scheduler is framework-agnostic: it knows
nothing about audio or the realtime connection. It only calls two callbacks,
``on_start`` (just before the dream begins) and ``on_finish`` (once it ends, always,
even on error). The realtime backend wires those to the subtle chime and the hidden
"you just dreamed" context note [see ``base_realtime.py``].

See ``docs/memory-system-design.md``.
"""

from __future__ import annotations
import logging
import threading
from typing import Callable
from dataclasses import dataclass

from reachy_mini_conversation_app.memory.dreamer import Dreamer, DreamLogStats
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


logger = logging.getLogger(__name__)


# OPENAI_MODEL_NAME is typically a realtime alias ("gpt-realtime") which doesn't
# exist on the Responses API that the dreamer uses. Don't fall back to it; pick
# a chat-capable default instead.
DEFAULT_DREAMER_MODEL = "gpt-5.4"


@dataclass
class DreamSummary:
    """One-line outcome of a dream pass, used to phrase the awareness note."""

    logs_processed: int = 0
    created: int = 0
    updated: int = 0
    errored: bool = False

    @classmethod
    def from_stats(cls, stats: list[DreamLogStats]) -> "DreamSummary":
        """Fold the dreamer's per-log stats into a single summary."""
        return cls(
            logs_processed=len(stats),
            created=sum(s.created for s in stats),
            updated=sum(s.updated for s in stats),
            errored=any(s.errors for s in stats),
        )


class DreamScheduler:
    """Run a dream pass on a daemon thread, bracketed by start/finish callbacks.

    Usage::

        scheduler = DreamScheduler(
            memory_manager, model="gpt-5.4", api_key=KEY,
            on_start=play_start_chime, on_finish=play_end_chime,
        )
        scheduler.start()  # returns immediately; dream runs in the background
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        on_start: Callable[[], None],
        on_finish: Callable[[DreamSummary], None],
        dreamer_factory: Callable[[], Dreamer] | None = None,
    ) -> None:
        """Initialize the scheduler. Pass ``dreamer_factory`` in tests to stub the dreamer."""
        self._manager = memory_manager
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._on_start = on_start
        self._on_finish = on_finish
        self._dreamer_factory = dreamer_factory
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Spawn the dream thread if there is anything to dream about.

        Returns ``True`` if a thread was started, ``False`` if skipped (already
        running, or no pending logs). Never raises.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.info("[DREAM] A dream is already running; not starting another.")
            return False

        pending = self._manager.list_pending_logs(exclude_session=True)
        if not pending:
            logger.info("[DREAM] No pending logs; skipping background dream.")
            return False

        logger.info("[DREAM] Launching background dream over %d pending log(s).", len(pending))
        self._thread = threading.Thread(target=self._run, name="dream-scheduler", daemon=True)
        self._thread.start()
        return True

    def is_running(self) -> bool:
        """Whether a dream thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Thread body: on_start, run the dreamer, then always on_finish."""
        try:
            self._on_start()
        except Exception:
            logger.exception("[DREAM] on_start callback raised; continuing.")

        summary = DreamSummary(errored=True)
        try:
            dreamer = self._dreamer_factory() if self._dreamer_factory else self._build_dreamer()
            stats = dreamer.run()
            summary = DreamSummary.from_stats(stats)
            logger.info(
                "[DREAM] Background dream finished: %d log(s), created %d, updated %d.",
                summary.logs_processed,
                summary.created,
                summary.updated,
            )
        except Exception:
            logger.exception("[DREAM] Background dream failed; conversation is unaffected.")
        finally:
            try:
                self._on_finish(summary)
            except Exception:
                logger.exception("[DREAM] on_finish callback raised.")

    def _build_dreamer(self) -> Dreamer:
        return Dreamer(self._manager, model=self._model, api_key=self._api_key, base_url=self._base_url)
