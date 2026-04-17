"""
Eagle Display: Live Terminal Dashboard
========================================
Renders a real-time trading dashboard in the terminal using Rich.

Layout:
    ┌─────────── HEADER ───────────┐
    │  BTC/USDT price + changes    │
    ├────────────┬─────────────────┤
    │ Indicators │ Strategy Signals │
    ├────────────┴─────────────────┤
    │     Trade Recommendation     │
    ├──────────────┬───────────────┤
    │  Portfolio   │ Trade History  │
    └──────────────┴───────────────┘

The ``LiveDashboard`` class holds a snapshot of the latest data and
exposes ``render()`` which returns a Rich ``Layout`` ready to be
displayed inside a ``rich.live.Live`` context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eagle.execution.paper_account import PaperAccount, Trade
from eagle.indicators.realtime_calculator import Indicators
from eagle.learning.weight_adapter import WeightAdapter
from eagle.strategies.realtime.aggregator import TradeRecommendation
from eagle.strategies.realtime.base import SignalDirection


# ── colour helpers ─────────────────────────────────────────────────────────

def _pct_color(val: float) -> str:
    if val > 0.05:
        return "bright_green"
    if val > 0:
        return "green"
    if val < -0.05:
        return "bright_red"
    if val < 0:
        return "red"
    return "white"


def _score_color(score: float) -> str:
    if score >= 0.55:
        return "bright_green"
    if score >= 0.30:
        return "green"
    if score <= -0.55:
        return "bright_red"
    if score <= -0.30:
        return "red"
    return "yellow"


def _rsi_color(rsi: float) -> str:
    if rsi < 30:
        return "bright_green"
    if rsi > 70:
        return "bright_red"
    return "white"


# ── bar helpers ─────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 20) -> str:
    """Convert -1..+1 score to a visual bar like ◀▓▓▓▓░░░░▶."""
    half = width // 2
    pos = int((score + 1) / 2 * width)
    bar = "░" * width
    bar = bar[:pos] + "█" + bar[pos + 1:]
    mid = half
    return bar[:mid] + "|" + bar[mid:]


# ── snapshot dataclass ──────────────────────────────────────────────────────

@dataclass
class DashboardSnapshot:
    """All data needed to render one frame of the dashboard."""

    price: float = 0.0
    indicators: Optional[Indicators] = None
    recommendation: Optional[TradeRecommendation] = None
    last_trade: Optional[Trade] = None
    account: Optional[PaperAccount] = None
    weight_adapter: Optional[WeightAdapter] = None
    candles_received: int = 0
    connected: bool = False
    status_msg: str = "Connecting..."
    last_update: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_learn_event: str = ""


# ── main dashboard class ────────────────────────────────────────────────────

class LiveDashboard:
    """
    Maintains a ``DashboardSnapshot`` and renders it on demand.

    Usage (inside asyncio + rich.live.Live)::

        dashboard = LiveDashboard()
        with Live(dashboard.render(), refresh_per_second=2) as live:
            while True:
                dashboard.update(snapshot)
                live.update(dashboard.render())
                await asyncio.sleep(0.5)
    """

    def __init__(self) -> None:
        self._snap = DashboardSnapshot()

    def update(self, snap: DashboardSnapshot) -> None:
        self._snap = snap

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=5),
            Layout(name="middle", ratio=1),
            Layout(self._recommendation(), name="rec", size=5),
            Layout(name="bottom", ratio=1),
            Layout(self._learning_panel(), name="learn", size=9),
        )
        layout["middle"].split_row(
            Layout(self._indicators(), name="indicators"),
            Layout(self._signals(), name="signals"),
        )
        layout["bottom"].split_row(
            Layout(self._portfolio(), name="portfolio"),
            Layout(self._trade_history(), name="history"),
        )
        return layout

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _header(self) -> Panel:
        snap = self._snap
        status_color = "bright_green" if snap.connected else "bright_red"
        status = "● LIVE" if snap.connected else "○ CONNECTING"

        price_text = Text(f"${snap.price:>12,.2f}", style="bold bright_white", justify="right")

        chg_parts: list[str] = []
        if snap.indicators:
            ind = snap.indicators
            for label, val in [("1m", ind.price_change_1m), ("5m", ind.price_change_5m), ("15m", ind.price_change_15m)]:
                arrow = "▲" if val >= 0 else "▼"
                color = _pct_color(val / 100)
                chg_parts.append(f"[{color}]{arrow}{abs(val):.3f}%[/{color}] ({label})")
        chg = "  ".join(chg_parts) if chg_parts else ""

        ts = snap.last_update.strftime("%H:%M:%S UTC")
        header = Text.assemble(
            Text(f"  [{status_color}]{status}[/{status_color}]  "),
            Text("BTC/USDT  ", style="bold cyan"),
            price_text,
            Text(f"    {chg}", style=""),
            Text(f"    candles: {snap.candles_received}    {ts}", style="dim"),
        )
        return Panel(header, title="[bold yellow]🦅 EAGLE — Bitcoin Real-Time Trading Bot[/bold yellow]", border_style="yellow")

    def _indicators(self) -> Panel:
        snap = self._snap
        if not snap.indicators:
            return Panel("[dim]Waiting for enough data…[/dim]", title="Indicators", border_style="blue")

        ind = snap.indicators
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", width=18)
        t.add_column(justify="right", width=12)
        t.add_column(width=25)

        def row(label: str, value: str, note: str = "") -> None:
            t.add_row(label, value, f"[dim]{note}[/dim]")

        # RSI
        rsi_color = _rsi_color(ind.rsi)
        row("RSI(14)", f"[{rsi_color}]{ind.rsi:.1f}[/{rsi_color}]",
            "oversold<30 | overbought>70")

        # MACD
        hist_color = "green" if ind.macd_hist > 0 else "red"
        row("MACD", f"{ind.macd:.2f}", f"signal {ind.macd_signal:.2f}")
        row("  histogram", f"[{hist_color}]{ind.macd_hist:+.2f}[/{hist_color}]",
            "crossing zero = signal")

        # Bollinger
        bb_color = _pct_color((0.5 - ind.bb_pct) * 2)
        row("BB %B", f"[{bb_color}]{ind.bb_pct:.2f}[/{bb_color}]",
            f"L={ind.bb_lower:,.0f} M={ind.bb_mid:,.0f} U={ind.bb_upper:,.0f}")
        row("  width", f"{ind.bb_width*100:.2f}%", "higher = more volatile")

        # EMAs
        e9_c = "green" if ind.price > ind.ema_9 else "red"
        e21_c = "green" if ind.price > ind.ema_21 else "red"
        e50_c = "green" if ind.price > ind.ema_50 else "red"
        row("EMA 9", f"[{e9_c}]{ind.ema_9:,.0f}[/{e9_c}]", "")
        row("EMA 21", f"[{e21_c}]{ind.ema_21:,.0f}[/{e21_c}]", "")
        row("EMA 50", f"[{e50_c}]{ind.ema_50:,.0f}[/{e50_c}]", "")

        # Volume
        vol_color = "bright_green" if ind.volume_ratio > 1.5 else ("green" if ind.volume_ratio > 1.0 else "dim")
        row("Volume ratio", f"[{vol_color}]{ind.volume_ratio:.2f}x[/{vol_color}]",
            f"avg {ind.volume_avg_20:.2f} BTC/min")

        # ATR
        row("ATR(14)", f"{ind.atr:.2f}", f"{ind.atr_pct*100:.3f}% of price")

        return Panel(t, title="[bold blue]Indicators[/bold blue]", border_style="blue")

    def _signals(self) -> Panel:
        snap = self._snap
        if not snap.recommendation:
            return Panel("[dim]Waiting…[/dim]", title="Strategy Signals", border_style="cyan")

        rec = snap.recommendation
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        t.add_column("Strategy", style="dim", width=16)
        t.add_column("Signal", width=6, justify="center")
        t.add_column("Score", width=7, justify="right")
        t.add_column("Bar", width=22)
        t.add_column("Reason", style="dim")

        for sig in rec.signals:
            sc = sig.score
            color = _score_color(sc)
            dir_str = sig.direction.value
            dir_color = "green" if dir_str == "BUY" else ("red" if dir_str == "SELL" else "yellow")
            t.add_row(
                sig.strategy_name,
                f"[{dir_color}]{dir_str}[/{dir_color}]",
                f"[{color}]{sc:+.2f}[/{color}]",
                f"[{color}]{_score_bar(sc)}[/{color}]",
                sig.reason[:40],
            )

        return Panel(t, title="[bold cyan]Strategy Signals[/bold cyan]", border_style="cyan")

    def _recommendation(self) -> Panel:
        snap = self._snap
        if not snap.recommendation:
            return Panel(
                Text("Accumulating data — need 50 closed candles…", style="dim", justify="center"),
                title="Trade Recommendation",
                border_style="white",
            )

        rec = snap.recommendation
        label = rec.label
        score = rec.score
        conf = rec.confidence

        color_map = {
            "STRONG BUY": "bright_green",
            "BUY": "green",
            "HOLD": "yellow",
            "SELL": "red",
            "STRONG SELL": "bright_red",
        }
        color = color_map.get(label, "white")

        conf_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        rec_text = Text.assemble(
            Text(f"  {rec.emoji}  ", style="bold"),
            Text(f"{label}", style=f"bold {color}"),
            Text(f"    score: [{color}]{score:+.3f}[/{color}]   confidence: {conf_bar} {conf*100:.0f}%   "),
            Text(f"{rec.summary}", style="dim"),
        )
        return Panel(rec_text, title="[bold white]● BEST TRADE NOW[/bold white]", border_style=color)

    def _portfolio(self) -> Panel:
        snap = self._snap
        if not snap.account:
            return Panel("[dim]No account[/dim]", title="Portfolio", border_style="magenta")

        acc = snap.account
        price = snap.price or 1.0
        summary = acc.summary(price)

        equity = summary["equity_usd"]
        pnl = summary["total_pnl_usd"]
        pnl_pct = summary["total_pnl_pct"]
        unreal = summary["unrealised_pnl"]
        pnl_color = "bright_green" if pnl >= 0 else "bright_red"
        unreal_color = "green" if unreal >= 0 else "red"

        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", width=18)
        t.add_column(justify="right")

        t.add_row("Cash (USD)", f"${summary['cash_usd']:>12,.2f}")
        t.add_row("BTC held", f"{summary['btc_held']:>.8f} BTC")
        t.add_row("BTC value", f"${summary['btc_value_usd']:>12,.2f}")
        t.add_row("─" * 18, "─" * 14)
        t.add_row("Total equity", f"[bold]${equity:>12,.2f}[/bold]")
        t.add_row("Unrealised P&L", f"[{unreal_color}]{unreal:>+12,.2f}[/{unreal_color}]")
        t.add_row("Total P&L", f"[{pnl_color}]{pnl:>+12,.2f}[/{pnl_color}]")
        t.add_row("Return", f"[{pnl_color}]{pnl_pct:>+11.2f}%[/{pnl_color}]")
        t.add_row("# Trades", f"{summary['total_trades']:>14}")

        return Panel(t, title="[bold magenta]Portfolio[/bold magenta]", border_style="magenta")

    def _trade_history(self) -> Panel:
        snap = self._snap
        if not snap.account or not snap.account.recent_trades:
            return Panel("[dim]No trades yet[/dim]", title="Recent Trades", border_style="white")

        t = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        t.add_column("#", width=4, justify="right")
        t.add_column("Side", width=5, justify="center")
        t.add_column("Price", width=12, justify="right")
        t.add_column("BTC", width=10, justify="right")
        t.add_column("P&L", width=10, justify="right")
        t.add_column("Label", width=12)
        t.add_column("Time", width=9)

        for trade in reversed(snap.account.recent_trades):
            side_color = "green" if trade.side == "BUY" else "red"
            pnl_str = f"${trade.realised_pnl:+,.2f}" if trade.side == "SELL" else "—"
            pnl_color = "green" if trade.realised_pnl >= 0 else "red"
            t.add_row(
                str(trade.trade_id),
                f"[{side_color}]{trade.side}[/{side_color}]",
                f"${trade.price:,.2f}",
                f"{trade.btc_qty:.5f}",
                f"[{pnl_color}]{pnl_str}[/{pnl_color}]",
                f"[dim]{trade.label}[/dim]",
                trade.timestamp.strftime("%H:%M:%S"),
            )

        return Panel(t, title="[bold white]Recent Trades[/bold white]", border_style="white")

    def _learning_panel(self) -> Panel:
        snap = self._snap
        wa = snap.weight_adapter

        if wa is None or wa.total_learned == 0:
            content = Text(
                f"  Self-learning active — waiting for first completed trade to start grading strategies...  "
                f"  {snap.last_learn_event or ''}",
                style="dim", justify="center",
            )
            return Panel(content, title="[bold yellow]Self-Learning Engine[/bold yellow]", border_style="yellow")

        rows = wa.summary_table()  # [(name, accuracy%, weight%, graded)]

        t = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow", expand=True)
        t.add_column("Strategy", width=18)
        t.add_column("Accuracy", width=10, justify="right")
        t.add_column("Weight", width=9, justify="right")
        t.add_column("Graded", width=8, justify="right")
        t.add_column("Accuracy Bar", min_width=24)
        t.add_column("Status", width=16)

        for name, acc_pct, w_pct, graded in rows:
            acc_color = "bright_green" if acc_pct >= 60 else ("yellow" if acc_pct >= 45 else "bright_red")
            w_color = "cyan" if w_pct >= 28 else "dim"
            bar_filled = int(acc_pct / 100 * 24)
            bar = f"[{acc_color}]{'█' * bar_filled}[/{acc_color}][dim]{'░' * (24 - bar_filled)}[/dim]"
            if graded == 0:
                status = "[dim]no data yet[/dim]"
            elif acc_pct >= 60:
                status = "[green]trusted[/green]"
            elif acc_pct >= 45:
                status = "[yellow]neutral[/yellow]"
            else:
                status = "[red]down-weighted[/red]"
            t.add_row(
                f"[bold]{name}[/bold]",
                f"[{acc_color}]{acc_pct:.1f}%[/{acc_color}]",
                f"[{w_color}]{w_pct:.1f}%[/{w_color}]",
                str(graded),
                bar,
                status,
            )

        learn_line = Text.assemble(
            Text(f"  Trades learned: {wa.total_learned}   ", style="bold"),
            Text(snap.last_learn_event, style="dim italic"),
        )
        return Panel(
            Group(learn_line, t),
            title="[bold yellow]Self-Learning Engine  (weights update after every closed trade)[/bold yellow]",
            border_style="yellow",
        )
