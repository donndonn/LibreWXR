# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.sources.regional.europe.radar.opera import (
    OperaSource,
    _parse_opera_hdf5,
)
from librewxr.sources.regional.europe.radar.opera import source as opera_source
from librewxr.tiles.coordinates import tile_overlaps_region


class TestOperaRegion:
    def test_opera_in_regions(self):
        assert "OPERA" in REGIONS

    def test_opera_is_laea(self):
        r = REGIONS["OPERA"]
        assert r.proj == "laea"
        assert r.group == "EUROPE"

    def test_opera_dimensions(self):
        r = REGIONS["OPERA"]
        assert r.width == 3800
        assert r.height == 4400

    def test_europe_group_resolution(self):
        # EUROPE bundles OPERA + ITCOMP (DPC Italian composite).  Group
        # membership is stored alphabetically by the discovery walker.
        assert resolve_regions("EUROPE") == ["ITCOMP", "OPERA"]

    def test_all_includes_opera(self):
        assert "OPERA" in resolve_regions("ALL")

    def test_mixed_regions(self):
        result = resolve_regions("EUROPE,CANADA")
        assert "OPERA" in result
        assert "CACOMP" in result


class TestOperaLAEA:
    def test_tile_over_europe_overlaps(self):
        region = REGIONS["OPERA"]
        # z=4, x=8, y=5 is central Europe
        assert tile_overlaps_region(region, z=4, x=8, y=5)

    def test_tile_over_america_does_not_overlap(self):
        region = REGIONS["OPERA"]
        # z=4, x=3, y=5 is eastern US
        assert not tile_overlaps_region(region, z=4, x=3, y=5)


class TestOperaHDF5Parser:
    def _make_opera_hdf5(
        self,
        data: np.ndarray,
        nodata: float = -9999000.0,
        undetect: float = -8888000.0,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> bytes:
        """Build a minimal OPERA-style ODIM HDF5 in memory."""
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            ds = f.create_group("dataset1")
            d1 = ds.create_group("data1")
            d1.create_dataset("data", data=data)
            what = d1.create_group("what")
            what.attrs["gain"] = gain
            what.attrs["offset"] = offset
            what.attrs["nodata"] = nodata
            what.attrs["undetect"] = undetect
            what.attrs["quantity"] = b"DBZH"
        return buf.getvalue()

    def test_basic_parsing(self):
        data = np.array([[20.0, 30.0], [-9999000.0, -8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result is not None
        assert result.shape == (2, 2)

    def test_nodata_becomes_zero(self):
        data = np.array([[-9999000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_undetect_becomes_zero(self):
        data = np.array([[-8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_valid_dbz_encoding(self):
        # 20 dBZ -> (20+32)*2 = 104
        data = np.array([[20.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 104

    def test_high_dbz_encoding(self):
        # 60 dBZ -> (60+32)*2 = 184
        data = np.array([[60.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 184

    def test_negative_dbz_encoding(self):
        # -10 dBZ -> (-10+32)*2 = 44
        data = np.array([[-10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 44

    def test_bad_data_returns_none(self):
        result = _parse_opera_hdf5(b"not hdf5")
        assert result is None

    def test_coverage_distinction(self):
        """Both nodata and undetect map to 0; only real precip is non-zero."""
        data = np.array([[-9999000.0, -8888000.0, 10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0    # nodata → 0
        assert result[0, 1] == 0    # undetect → 0 (gap-filler, let ECMWF fill)
        assert result[0, 2] >= 2    # real dBZ


class TestOperaSourceURL:
    def test_url_construction(self):
        src = OperaSource("https://example.com")
        # 2026-04-10 09:05 UTC → ts=1775991900
        from datetime import datetime, timezone
        dt = datetime(2026, 4, 10, 9, 5, tzinfo=timezone.utc)
        ts = int(dt.timestamp())
        url = src._url_for_timestamp(ts)
        assert "OPERA@20260410T0905@0@DBZH.h5" in url
        assert url.startswith("https://example.com/openradar-24h/2026/04/10/OPERA/COMP/")

    def test_url_rounds_to_5min(self):
        src = OperaSource("https://example.com")
        from datetime import datetime, timezone
        # 09:07 should round to 09:05
        dt = datetime(2026, 4, 10, 9, 7, tzinfo=timezone.utc)
        url = src._url_for_timestamp(int(dt.timestamp()))
        assert "T0905@" in url


class TestOperaFetchOffLoop:
    @staticmethod
    async def _fake_retry_get(client, url, log_name="OPERA"):
        class Response:
            status_code = 200
            content = b"hdf5 payload"

        return Response()

    async def test_parse_does_not_block_event_loop(self, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def parse(data):
            started.set()
            release.wait(timeout=3)
            return threading.get_ident(), data

        monkeypatch.setattr(opera_source, "retry_get", self._fake_retry_get)
        monkeypatch.setattr(opera_source, "_parse_opera_hdf5", parse)
        fetch = asyncio.create_task(OperaSource()._fetch_hdf5(1_700_000_000))
        try:
            assert await asyncio.wait_for(asyncio.to_thread(started.wait, 3), 4)
            assert not fetch.done()
        finally:
            release.set()

        result = await fetch

        worker_id, payload = result
        assert payload == b"hdf5 payload"
        assert worker_id != threading.get_ident()

    async def test_cancellation_keeps_parse_limit(self, monkeypatch):
        monkeypatch.setattr(opera_source, "retry_get", self._fake_retry_get)
        release = threading.Event()
        lock = threading.Lock()
        active = started = peak = 0

        def parse(data):
            nonlocal active, started, peak
            with lock:
                active += 1
                started += 1
                peak = max(peak, active)
            try:
                release.wait(timeout=5)
                return np.zeros((1, 1), dtype=np.uint8)
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(opera_source, "_parse_opera_hdf5", parse)
        with ThreadPoolExecutor(max_workers=2) as pool:
            monkeypatch.setattr(opera_source, "_PARSE_EXECUTOR", pool)
            source = OperaSource()
            tasks = [
                asyncio.create_task(source._fetch_hdf5(1_700_000_000 + i))
                for i in range(3)
            ]
            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + 2
                while started < 2 and loop.time() < deadline:
                    await asyncio.sleep(0.01)
                assert started == 2

                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                await asyncio.sleep(0.05)
                assert started == 2
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

        assert started == 3
        assert peak == 2
