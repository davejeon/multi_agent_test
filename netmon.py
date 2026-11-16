#!/usr/bin/env python3
"""
netmon.py - a small live network traffic monitor for the terminal.

Reads the OS per-interface traffic counters through psutil (no root needed)
and draws throughput plus sparkline history with rich.

    pip install psutil rich
    python netmon.py

Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from collections import deque

try:
    import psutil
except ImportError:
    sys.exit("netmon needs psutil:  pip install psutil")

try:
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("netmon needs rich:  pip install rich")


SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

PALETTE = [
    "cyan", "magenta", "green", "yellow", "blue", "red",
    "bright_cyan", "bright_magenta", "bright_green", "bright_yellow",
    "bright_blue", "bright_red",
]


# --------------------------------------------------------------- formatting

def human_size(n: float) -> str:
    n = max(float(n), 0.0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def human_rate(n: float) -> str:
    return f"{human_size(n)}/s"


def human_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{mins:02d}m"
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def sparkline(values, width: int, scale: float) -> str:
    """Unicode sparkline: oldest on the left, newest on the right."""
    if width <= 0:
        return ""
    vals = list(values)[-width:]
    if not vals:
        return " " * width
    top = len(SPARK) - 1
    scale = max(scale, 1e-9)
    out = []
    for v in vals:
        step = int((v / scale) * top + 0.5)
        out.append(SPARK[max(0, min(top, step))])
    return "".join(out).rjust(width, " ")


# ------------------------------------------------------------------- state

class Iface:
    """Rolling state for one network interface."""

    __slots__ = ("name", "color", "rx_hist", "tx_hist", "rx_rate", "tx_rate",
                 "bytes_recv", "bytes_sent", "errs", "drops")

    def __init__(self, name: str, history: int, color: str):
        self.name = name
        self.color = color
        self.rx_hist = deque(maxlen=history)
        self.tx_hist = deque(maxlen=history)
        self.rx_rate = 0.0
        self.tx_rate = 0.0
        self.bytes_recv = 0
        self.bytes_sent = 0
        self.errs = 0
        self.drops = 0


class Monitor:
    def __init__(self, history: int = 60, exclude=None):
        self.history = history
        self.exclude = exclude
        self.ifaces: dict[str, Iface] = {}
        self._prev = psutil.net_io_counters(pernic=True)
        self._prev_t = time.monotonic()
        self.started = self._prev_t
        self._scale_rx = 0.0
        self._scale_tx = 0.0
        self._color = 0
        self.samples = 0

    def sample(self) -> None:
        now = time.monotonic()
        elapsed = max(now - self._prev_t, 1e-6)
        cur = psutil.net_io_counters(pernic=True)
        prev = self._prev
        self._prev = cur
        self._prev_t = now

        for name, counters in cur.items():
            if self.exclude is not None and self.exclude.search(name):
                continue

            iface = self.ifaces.get(name)
            if iface is None:
                iface = Iface(name, self.history,
                              PALETTE[self._color % len(PALETTE)])
                self._color += 1
                self.ifaces[name] = iface

            old = prev.get(name)
            if old is None:
                rx_delta = tx_delta = 0.0
            else:
                # counters are cumulative since boot and reset if the link
                # bounces, so clamp negatives rather than showing garbage
                rx_delta = max(counters.bytes_recv - old.bytes_recv, 0)
                tx_delta = max(counters.bytes_sent - old.bytes_sent, 0)

            iface.bytes_recv = counters.bytes_recv
            iface.bytes_sent = counters.bytes_sent
            iface.errs = counters.errin + counters.errout
            iface.drops = counters.dropin + counters.dropout
            iface.rx_rate = rx_delta / elapsed
            iface.tx_rate = tx_delta / elapsed
            iface.rx_hist.append(iface.rx_rate)
            iface.tx_hist.append(iface.tx_rate)

        # forget interfaces that went away (USB dongles, VPN tunnels, ...)
        for name in [n for n in self.ifaces if n not in cur]:
            del self.ifaces[name]

        self.samples += 1

    @staticmethod
    def _sticky(current: float, peak: float) -> float:
        """Rescale the graph, but only when it really changes, to avoid jitter."""
        if peak > current or peak < current / 2.0:
            return peak
        return current

    def render(self, console: Console) -> Panel:
        states = sorted(self.ifaces.values(), key=lambda i: i.name)
        self._scale_rx = self._sticky(
            self._scale_rx,
            max((max(i.rx_hist, default=0.0) for i in states), default=0.0))
        self._scale_tx = self._sticky(
            self._scale_tx,
            max((max(i.tx_hist, default=0.0) for i in states), default=0.0))

        # responsive layout
        width = console.width
        show_totals = width >= 100
        ncols = 7 if show_totals else 5
        iface_w = 14 if width >= 90 else (12 if width >= 70 else 9)
        overhead = iface_w + 22 + (20 if show_totals else 0) + 2 * ncols + 4
        gw = max(6, min(40, (width - overhead) // 2))

        table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), show_edge=False)
        table.add_column("interface", style="bold", no_wrap=True,
                         overflow="ellipsis", width=iface_w)
        table.add_column("\u2193 rx/s", justify="right", width=11, no_wrap=True)
        table.add_column("\u2191 tx/s", justify="right", width=11, no_wrap=True)
        if show_totals:
            table.add_column("\u2193 total", justify="right", width=9, no_wrap=True)
            table.add_column("\u2191 total", justify="right", width=9, no_wrap=True)
        table.add_column("receive history", no_wrap=True, width=gw)
        table.add_column("transmit history", no_wrap=True, width=gw)

        total_rx = 0.0
        total_tx = 0.0
        for i in states:
            total_rx += i.rx_rate
            total_tx += i.tx_rate
            row = [
                Text(i.name, style=i.color),
                Text(human_rate(i.rx_rate), style="cyan" if i.rx_rate else "dim"),
                Text(human_rate(i.tx_rate), style="magenta" if i.tx_rate else "dim"),
            ]
            if show_totals:
                row.append(Text(human_size(i.bytes_recv), style="dim"))
                row.append(Text(human_size(i.bytes_sent), style="dim"))
            row.append(Text(sparkline(i.rx_hist, gw, self._scale_rx), style="cyan"))
            row.append(Text(sparkline(i.tx_hist, gw, self._scale_tx), style="magenta"))
            table.add_row(*row)

        if not states:
            table.add_row(Text("no interfaces matched", style="dim italic"),
                          *([""] * (ncols - 1)))

        footer = Text()
        footer.append("\u2193 ", style="cyan")
        footer.append(human_rate(total_rx), style="bold cyan")
        footer.append("    \u2191 ", style="magenta")
        footer.append(human_rate(total_tx), style="bold magenta")
        footer.append("    scale ", style="dim")
        footer.append(human_rate(max(self._scale_rx, self._scale_tx, 0.0)),
                      style="dim")
        errs = sum(i.errs for i in states)
        drops = sum(i.drops for i in states)
        if errs or drops:
            footer.append(f"    errors {errs} \u00b7 drops {drops}", style="yellow")
        footer.append(f"    up {human_duration(time.monotonic() - self.started)}",
                      style="dim")

        return Panel(
            Group(table, footer),
            title=f"[bold]netmon[/] [dim]\u00b7 {socket.gethostname()}[/]",
            subtitle=f"[dim]{len(states)} ifaces \u00b7 sample {self.samples}[/]",
            box=box.ROUNDED,
            padding=(0, 1),
        )


# -------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="netmon",
        description="Live per-interface network traffic in your terminal.")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="seconds between samples (default: 1.0)")
    p.add_argument("-H", "--history", type=int, default=60,
                   help="samples kept per sparkline (default: 60)")
    p.add_argument("-x", "--exclude", metavar="REGEX", default=None,
                   help="hide interfaces matching REGEX, e.g. '^lo$'")
    p.add_argument("-n", "--samples", type=int, default=0,
                   help="exit after N samples (default: run until Ctrl+C)")
    p.add_argument("--inline", action="store_true",
                   help="draw in the normal screen instead of the alternate screen")
    args = p.parse_args(argv)
    if args.interval <= 0:
        p.error("--interval must be greater than 0")
    if args.history < 1:
        p.error("--history must be at least 1")
    if args.samples < 0:
        p.error("--samples must be 0 or greater")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    exclude = re.compile(args.exclude) if args.exclude else None
    console = Console()
    mon = Monitor(history=args.history, exclude=exclude)
    use_screen = console.is_terminal and not args.inline

    try:
        with Live(console=console, refresh_per_second=8, screen=use_screen,
                  transient=not use_screen, auto_refresh=False) as live:
            mon.sample()
            live.update(mon.render(console), refresh=True)
            while args.samples == 0 or mon.samples < args.samples:
                time.sleep(args.interval)
                mon.sample()
                live.update(mon.render(console), refresh=True)
    except KeyboardInterrupt:
        return 0
    except psutil.Error as exc:
        console.print(f"[red]psutil error:[/] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
