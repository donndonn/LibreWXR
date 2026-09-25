#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Benchmark NowcastStore memmap writes vs asyncio event-loop lag.

Compares today's sync ``_to_memmap`` under ``asyncio.Lock`` with the
proposed ``asyncio.to_thread`` pattern (matching ``FrameStore.add_frame``)
so we can quantify the responsiveness win before/after moving nowcast
publishes off the event loop.

Success criteria for the fix:
  * Replace wall time stays within ~15% of the sync baseline (same I/O).
  * Event-loop lag during replace drops from ~replace duration toward
    the ticker interval (tens of ms).

Usage (from repo root, with the project venv active)::

    python scripts/bench_nowcast_memmap_event_loop.py
    python scripts/bench_nowcast_memmap_event_loop.py --runs 5 --height 2000 --width 4000

Synthetic USCOMP-ish uint8 regions — no network, no optical flow.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import numpy as np

from librewxr.data.nowcast import NowcastFrame, NowcastStore

# Default sizing: large enough to show clear event-loop stalls without
# allocating a full USCOMP (12200×5400 ≈ 66 MB) × 6 frames on every run.
_DEFAULT_HEIGHT = 2000
_DEFAULT_WIDTH = 4000
_DEFAULT_FRAMES = 6
_DEFAULT_REGIONS = ("USCOMP",)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:8.1f} ms"


def _summarize(label: str, samples: list[float]) -> None:
    med = statistics.median(samples)
    p95 = _percentile(samples, 95.0)
    print(
        f"  {label:22}  n={len(samples):2d}  "
        f"median={_fmt_ms(med)}  p95={_fmt_ms(p95)}  "
        f"max={_fmt_ms(max(samples))}"
    )


def _make_frames(
    *,
    n_frames: int,
    height: int,
    width: int,
    region_names: tuple[str, ...],
) -> list[NowcastFrame]:
    rng = np.random.default_rng(0)
    frames: list[NowcastFrame] = []
    base_ts = 1_700_000_000
    for i in range(n_frames):
        regions = {
            name: rng.integers(0, 200, size=(height, width), dtype=np.uint8)
            for name in region_names
        }
        frames.append(
            NowcastFrame(
                timestamp=base_ts + i * 600,
                regions=regions,
                blend_weight=max(0.1, 1.0 - i * 0.15),
            )
        )
    return frames


async def _replace_sync(store: NowcastStore, frames: list[NowcastFrame]) -> None:
    """Today's NowcastStore.replace_all body: sync memmap under the lock."""
    async with store._lock:
        for path in store._memmap_dir.glob("frame_*.dat"):
            try:
                path.unlink()
            except OSError:
                pass
        for frame in frames:
            for name, data in list(frame.regions.items()):
                frame.regions[name] = store._to_memmap(
                    f"frame_{frame.timestamp}_{name}", data,
                )
        store._frames = {f.timestamp: f for f in frames}


async def _replace_threaded(
    store: NowcastStore, frames: list[NowcastFrame],
) -> None:
    """Proposed pattern: memmap writes via to_thread (lock still held)."""
    async with store._lock:
        for path in store._memmap_dir.glob("frame_*.dat"):
            try:
                path.unlink()
            except OSError:
                pass
        for frame in frames:
            for name, data in list(frame.regions.items()):
                frame.regions[name] = await asyncio.to_thread(
                    store._to_memmap,
                    f"frame_{frame.timestamp}_{name}",
                    data,
                )
        store._frames = {f.timestamp: f for f in frames}


async def _measure_lag(
    store: NowcastStore,
    frames: list[NowcastFrame],
    *,
    use_thread: bool,
    ticker_s: float,
) -> tuple[float, list[float]]:
    lags: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(stop.wait(), timeout=ticker_s)
                break
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - t0
                lags.append(elapsed - ticker_s)

    tick_task = asyncio.create_task(ticker())
    await asyncio.sleep(ticker_s)

    # Fresh region array copies so each run pays full write cost (avoid
    # re-using already-memmapped views from a prior iteration).
    run_frames = [
        NowcastFrame(
            timestamp=f.timestamp,
            regions={k: np.array(v, copy=True) for k, v in f.regions.items()},
            blend_weight=f.blend_weight,
        )
        for f in frames
    ]

    t0 = time.perf_counter()
    if use_thread:
        await _replace_threaded(store, run_frames)
    else:
        await _replace_sync(store, run_frames)
    wall_s = time.perf_counter() - t0

    stop.set()
    await tick_task
    return wall_s, lags


async def _run_mode(
    cache_dir: Path,
    frames: list[NowcastFrame],
    *,
    use_thread: bool,
    runs: int,
    ticker_s: float,
) -> tuple[list[float], list[float]]:
    store = NowcastStore(cache_dir=cache_dir)
    await _measure_lag(
        store, frames, use_thread=use_thread, ticker_s=ticker_s,
    )

    walls: list[float] = []
    all_lags: list[float] = []
    for _ in range(runs):
        wall_s, lags = await _measure_lag(
            store, frames, use_thread=use_thread, ticker_s=ticker_s,
        )
        walls.append(wall_s)
        all_lags.extend(lags)
    store.clear()
    return walls, all_lags


async def _main_async(args: argparse.Namespace) -> int:
    region_names = tuple(args.regions.split(",")) if args.regions else _DEFAULT_REGIONS
    frames = _make_frames(
        n_frames=args.frames,
        height=args.height,
        width=args.width,
        region_names=region_names,
    )
    bytes_per_replace = sum(
        arr.nbytes for f in frames for arr in f.regions.values()
    )
    print(
        f"Payload: {args.frames} frames × {len(region_names)} region(s) "
        f"at {args.width}×{args.height} uint8 "
        f"({bytes_per_replace / 1e6:.1f} MB per replace)\n",
        flush=True,
    )

    ticker_s = args.ticker_ms / 1000.0
    modes = (
        ("sync-under-lock (baseline)", False),
        ("to_thread (proposed)", True),
    )

    results: dict[str, tuple[list[float], list[float]]] = {}
    for label, use_thread in modes:
        print(f"== {label} ==")
        cache = Path(args.cache_dir) / ("thread" if use_thread else "sync")
        cache.mkdir(parents=True, exist_ok=True)
        walls, lags = await _run_mode(
            cache,
            frames,
            use_thread=use_thread,
            runs=args.runs,
            ticker_s=ticker_s,
        )
        results[label] = (walls, lags)
        _summarize("replace wall", walls)
        _summarize("event-loop lag", lags)
        print()

    on_wall, on_lag = results["sync-under-lock (baseline)"]
    th_wall, th_lag = results["to_thread (proposed)"]
    on_wall_med = statistics.median(on_wall)
    th_wall_med = statistics.median(th_wall)
    on_lag_med = statistics.median(on_lag) if on_lag else float("nan")
    th_lag_med = statistics.median(th_lag) if th_lag else float("nan")
    on_lag_p95 = _percentile(on_lag, 95.0)
    th_lag_p95 = _percentile(th_lag, 95.0)

    wall_delta_pct = (
        (th_wall_med - on_wall_med) / on_wall_med * 100.0
        if on_wall_med > 0
        else float("nan")
    )

    print("== verdict ==")
    print(
        f"  replace median delta: {wall_delta_pct:+.1f}% "
        f"(target: within ~15%)"
    )
    print(
        f"  lag median: {_fmt_ms(on_lag_med).strip()} → "
        f"{_fmt_ms(th_lag_med).strip()}"
    )
    print(
        f"  lag p95:    {_fmt_ms(on_lag_p95).strip()} → "
        f"{_fmt_ms(th_lag_p95).strip()}"
    )

    wall_ok = abs(wall_delta_pct) <= 15.0 or (
        wall_delta_pct <= 30.0 and th_lag_p95 < on_lag_p95 * 0.25
    )
    lag_ok = th_lag_p95 < max(ticker_s * 4.0, 0.050) or (
        on_lag_p95 > 0 and th_lag_p95 < on_lag_p95 * 0.25
    )

    if wall_ok and lag_ok:
        print("  PASS: to_thread keeps replace cost flat and clears loop lag.")
        return 0

    print("  FAIL: criteria not met — inspect numbers before shipping.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--ticker-ms", type=float, default=50.0)
    parser.add_argument("--frames", type=int, default=_DEFAULT_FRAMES)
    parser.add_argument("--height", type=int, default=_DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=_DEFAULT_WIDTH)
    parser.add_argument(
        "--regions",
        default=",".join(_DEFAULT_REGIONS),
        help="Comma-separated region names to materialise per frame",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/librewxr_bench_nowcast_memmap",
        help="Scratch directory for memmap files",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
