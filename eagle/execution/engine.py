"""
Eagle Execution: Order Execution Engine
==========================================
Order management with broker adapters, paper trading simulation,
slippage modeling, and fill tracking.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from eagle.core.config import EagleConfig
from eagle.core.events import Event, EventBus, EventType
from eagle.strategies.manager import TradeSignal

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Represents a trading order."""

    order_id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: str = "buy"
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_quantity: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: datetime | None = None
    signal: TradeSignal | None = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        ...


class PaperBroker(BrokerAdapter):
    """
    Paper trading broker for simulation.

    Simulates order fills with configurable slippage and commission models.
    """

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._cfg = config.execution.paper
        self._orders: dict[str, Order] = {}
        self._last_prices: dict[str, float] = {}

    async def connect(self) -> None:
        logger.info("Paper broker connected")

    async def disconnect(self) -> None:
        logger.info("Paper broker disconnected")

    async def submit_order(self, order: Order) -> Order:
        """Simulate order execution with slippage."""
        # Simulate fill
        if order.order_type == OrderType.MARKET:
            base_price = order.limit_price or self._last_prices.get(order.symbol, 100.0)
            slippage = base_price * self._cfg.slippage_pct
            if order.side == "buy":
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage

            order.fill_price = fill_price
            order.fill_quantity = order.quantity
            order.commission = fill_price * order.quantity * self._cfg.commission_pct
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()

        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill at limit price (simplified)
            order.fill_price = order.limit_price
            order.fill_quantity = order.quantity
            order.commission = (order.limit_price or 0) * order.quantity * self._cfg.commission_pct
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()

        self._orders[order.order_id] = order

        logger.info(
            f"Paper fill: {order.side.upper()} {order.symbol} "
            f"x{order.quantity:.2f} @ {order.fill_price:.2f} "
            f"(commission: {order.commission:.2f})"
        )
        return order

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders and self._orders[order_id].is_active:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_account_info(self) -> dict[str, Any]:
        return {
            "broker": "paper",
            "balance": self._cfg.initial_capital,
            "orders": len(self._orders),
        }

    def set_price(self, symbol: str, price: float) -> None:
        """Set current market price for simulation."""
        self._last_prices[symbol] = price


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca Markets broker adapter.

    Supports both paper and live trading via the Alpaca API.
    """

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._client: Any = None

    async def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient

            alpaca_cfg = self.config.data.providers.alpaca
            self._client = TradingClient(
                api_key=alpaca_cfg.api_key,
                secret_key=alpaca_cfg.api_secret,
                paper=alpaca_cfg.paper,
            )
            logger.info(f"Alpaca broker connected (paper={alpaca_cfg.paper})")
        except ImportError:
            logger.warning("alpaca-py not installed, Alpaca broker unavailable")
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")

    async def disconnect(self) -> None:
        self._client = None
        logger.info("Alpaca broker disconnected")

    async def submit_order(self, order: Order) -> Order:
        if not self._client:
            order.status = OrderStatus.REJECTED
            return order

        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            alpaca_order = self._client.submit_order(request)
            order.order_id = str(alpaca_order.id)
            order.status = OrderStatus.SUBMITTED
            logger.info(f"Alpaca order submitted: {order.order_id}")

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            order.status = OrderStatus.REJECTED

        return order

    async def cancel_order(self, order_id: str) -> bool:
        if self._client:
            try:
                self._client.cancel_order_by_id(order_id)
                return True
            except Exception as e:
                logger.error(f"Cancel failed: {e}")
        return False

    async def get_account_info(self) -> dict[str, Any]:
        if self._client:
            account = self._client.get_account()
            return {
                "broker": "alpaca",
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            }
        return {"broker": "alpaca", "status": "disconnected"}


class ExecutionEngine:
    """
    Central execution engine that translates trade signals into orders,
    routes them to the appropriate broker, and tracks fills.
    """

    def __init__(self, config: EagleConfig, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._broker: BrokerAdapter | None = None
        self._order_history: list[Order] = []

    async def initialize(self) -> None:
        """Initialize the appropriate broker adapter."""
        broker_map: dict[str, type[BrokerAdapter]] = {
            "paper": PaperBroker,
            "alpaca": AlpacaBroker,
        }

        broker_name = self.config.execution.broker
        broker_class = broker_map.get(broker_name, PaperBroker)
        self._broker = broker_class(self.config)
        await self._broker.connect()
        logger.info(f"Execution engine ready (broker: {broker_name})")

    async def execute(self, signal: TradeSignal) -> Order:
        """Execute a trade signal by creating and submitting an order."""
        if not self._broker:
            raise RuntimeError("No broker connected")

        # Create order from signal
        order = Order(
            symbol=signal.symbol,
            side=signal.side.value,
            order_type=OrderType(self.config.execution.order_type),
            quantity=self._calculate_quantity(signal),
            signal=signal,
        )

        # Set limit price if applicable
        if order.order_type == OrderType.LIMIT and signal.metadata.get("current_price"):
            offset = signal.metadata["current_price"] * self.config.execution.limit_offset_pct
            if signal.is_buy:
                order.limit_price = signal.metadata["current_price"] + offset
            else:
                order.limit_price = signal.metadata["current_price"] - offset

        # Submit to broker
        order = await self._broker.submit_order(order)
        self._order_history.append(order)

        # Emit order event
        event_type = {
            OrderStatus.FILLED: EventType.ORDER_FILLED,
            OrderStatus.SUBMITTED: EventType.ORDER_SUBMITTED,
            OrderStatus.REJECTED: EventType.ORDER_REJECTED,
        }.get(order.status, EventType.ORDER_NEW)

        await self.event_bus.emit(Event(
            event_type=event_type,
            source="execution",
            data={
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
                "status": order.status.value,
            },
        ))

        return order

    def _calculate_quantity(self, signal: TradeSignal) -> float:
        """Calculate order quantity from signal size hint."""
        # This is a simplified version; real implementation would
        # query portfolio for current equity
        base_capital = self.config.execution.paper.initial_capital
        size_frac = signal.size_hint or self.config.risk.max_portfolio_risk
        position_value = base_capital * size_frac

        # We'd need the current price to convert to shares
        # For now, return the position value (to be refined)
        return round(position_value, 2)

    async def shutdown(self) -> None:
        """Disconnect from broker."""
        if self._broker:
            await self._broker.disconnect()

    @property
    def order_history(self) -> list[Order]:
        return list(self._order_history)
