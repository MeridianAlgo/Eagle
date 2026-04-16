"""
Eagle Core: Event Bus System
=============================
Async event-driven architecture for decoupled component communication.
All market data, signals, orders, and fills flow through this bus.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventType(Enum):
    """All event types flowing through the Eagle system."""

    # Market Data Events
    MARKET_DATA = "market_data"
    TICK = "tick"
    BAR = "bar"
    ORDER_BOOK = "order_book"

    # Signal Events
    SIGNAL = "signal"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_REJECTED = "signal_rejected"

    # Order Events
    ORDER_NEW = "order_new"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Portfolio Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    PORTFOLIO_UPDATED = "portfolio_updated"

    # Risk Events
    RISK_ALERT = "risk_alert"
    RISK_BREACH = "risk_breach"
    DRAWDOWN_ALERT = "drawdown_alert"
    TRADING_HALTED = "trading_halted"

    # System Events
    ENGINE_START = "engine_start"
    ENGINE_STOP = "engine_stop"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

    # Model Events
    MODEL_TRAINED = "model_trained"
    MODEL_PREDICTION = "model_prediction"
    RETRAIN_TRIGGERED = "retrain_triggered"


@dataclass
class Event:
    """Base event object flowing through the event bus."""

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    source: str = ""
    priority: int = 5  # 1 = highest, 10 = lowest

    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, source={self.source}, id={self.event_id[:8]})"


# Type alias for event handler functions
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async event bus for decoupled communication between Eagle components.

    Features:
        - Async publish/subscribe pattern
        - Priority-based event processing
        - Event filtering and routing
        - Dead letter queue for failed events
        - Event history for debugging
        - Wildcard subscriptions
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: dict[EventType, list[tuple[int, EventHandler]]] = defaultdict(list)
        self._wildcard_subscribers: list[tuple[int, EventHandler]] = []
        self._queue: asyncio.PriorityQueue[tuple[int, float, Event]] = asyncio.PriorityQueue()
        self._history: list[Event] = []
        self._max_history = max_history
        self._dead_letter: list[tuple[Event, Exception]] = []
        self._running = False
        self._stats: dict[str, int] = defaultdict(int)

    def subscribe(self, event_type: EventType, handler: EventHandler, priority: int = 5) -> None:
        """Subscribe a handler to a specific event type."""
        self._subscribers[event_type].append((priority, handler))
        self._subscribers[event_type].sort(key=lambda x: x[0])
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type.value}")

    def subscribe_all(self, handler: EventHandler, priority: int = 5) -> None:
        """Subscribe a handler to ALL event types (wildcard)."""
        self._wildcard_subscribers.append((priority, handler))
        self._wildcard_subscribers.sort(key=lambda x: x[0])
        logger.debug(f"Subscribed {handler.__qualname__} to ALL events")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from a specific event type."""
        self._subscribers[event_type] = [
            (p, h) for p, h in self._subscribers[event_type] if h != handler
        ]

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus for async processing."""
        await self._queue.put((event.priority, event.timestamp.timestamp(), event))
        self._stats["published"] += 1

    async def emit(self, event: Event) -> None:
        """Directly dispatch an event to handlers (synchronous dispatch)."""
        self._store_history(event)
        self._stats["emitted"] += 1

        handlers = [
            *self._subscribers.get(event.event_type, []),
            *self._wildcard_subscribers,
        ]

        for _priority, handler in handlers:
            try:
                await handler(event)
                self._stats["handled"] += 1
            except Exception as e:
                logger.error(f"Handler {handler.__qualname__} failed for {event}: {e}")
                self._dead_letter.append((event, e))
                self._stats["failed"] += 1

    async def process_queue(self) -> None:
        """Process events from the priority queue. Runs as a background task."""
        self._running = True
        logger.info("Event bus processing started")

        while self._running:
            try:
                _priority, _ts, event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                await self.emit(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

        logger.info("Event bus processing stopped")

    def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False

    def _store_history(self, event: Event) -> None:
        """Store event in rolling history buffer."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def stats(self) -> dict[str, int]:
        """Return event processing statistics."""
        return dict(self._stats)

    @property
    def pending_count(self) -> int:
        """Number of events waiting in the queue."""
        return self._queue.qsize()

    @property
    def dead_letters(self) -> list[tuple[Event, Exception]]:
        """Events that failed processing."""
        return list(self._dead_letter)

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """Retrieve event history, optionally filtered by type."""
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]
