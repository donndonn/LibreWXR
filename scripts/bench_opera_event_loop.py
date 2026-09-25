#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Compare OPERA parse time and event-loop lag on the loop and in its pool."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import statistics
import sys
import time
import types
from pathlib import Path

import h5py
import numpy as np


def _load_opera():
    root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(root))
    import librewxr  # noqa: F401

    # Avoid source discovery while importing the parser for this benchmark.
    sources = types.ModuleType("librewxr.sources")
    sources.__path__ = [str(root / "librewxr" / "sources")]
    sources.iter_source_packages = lambda: iter(())
    sources.RADAR_PROVIDERS = []
    sources.NWP_PROVIDERS = []
    sources.SATELLITE_PROVIDERS = []
    sources.NOWCAST_PROVIDERS = []
    sys.modules["librewxr.sources"] = sources

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / "librewxr" / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("librewxr.sources._helpers", "sources/_helpers.py")
    return load(
        "librewxr.sources.regional.europe.radar.opera.source",
        "sources/regional/europe/radar/opera/source.py",
    )


def _payload():
    rng = np.random.default_rng(0)
    data = rng.uniform(-10, 55, size=(4400, 3800))
    data[::3] = -9999000.0
    data[1::5] = -8888000.0
    out = io.BytesIO()
    with h5py.File(out, "w") as file:
        item = file.create_group("dataset1/data1")
        item.create_dataset("data", data=data)
        what = item.create_group("what")
        what.attrs["gain"] = 1.0
        what.attrs["offset"] = 0.0
        what.attrs["nodata"] = -9999000.0
        what.attrs["undetect"] = -8888000.0
    return out.getvalue()


async def _sample(opera, payload, interval, threaded):
    stop = asyncio.Event()
    lags = []

    async def ticker():
        while True:
            started = time.perf_counter()
            try:
                await asyncio.wait_for(stop.wait(), interval)
                return
            except asyncio.TimeoutError:
                lags.append(time.perf_counter() - started - interval)

    tick_task = asyncio.create_task(ticker())
    await asyncio.sleep(interval)
    started = time.perf_counter()
    if threaded:
        result = await asyncio.get_running_loop().run_in_executor(
            opera._PARSE_EXECUTOR, opera._parse_opera_hdf5, payload,
        )
    else:
        result = opera._parse_opera_hdf5(payload)
    wall = time.perf_counter() - started
    stop.set()
    await tick_task
    assert result is not None
    return wall, max(lags, default=0.0)


async def _main(runs, interval):
    opera = _load_opera()
    payload = _payload()
    print(f"OPERA grid: 3800×4400; payload: {len(payload) / 1e6:.1f} MB")
    results = {False: [], True: []}
    await _sample(opera, payload, interval, False)
    await _sample(opera, payload, interval, True)
    for run in range(runs):
        for threaded in (False, True) if run % 2 == 0 else (True, False):
            results[threaded].append(await _sample(opera, payload, interval, threaded))
    for threaded, label in ((False, "on loop"), (True, "OPERA pool")):
        values = results[threaded]
        wall = statistics.median(value[0] for value in values) * 1000
        lag = statistics.median(value[1] for value in values) * 1000
        print(f"{label:10} median parse {wall:6.1f} ms; median max loop lag {lag:6.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--ticker-ms", type=float, default=50)
    args = parser.parse_args()
    asyncio.run(_main(args.runs, args.ticker_ms / 1000))
