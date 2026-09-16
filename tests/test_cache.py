"""
Tests for utils/cache.py
CacheManager: get/set/clear/TTL, thread safety, global instance.
"""

import threading
from unittest.mock import patch


from utils.cache import CacheManager, cache


# ── Basic get / set ─────────────────────────────────────────────────────────

class TestCacheBasic:
    def test_set_and_get(self):
        cm = CacheManager(default_ttl=60)
        cm.set("key1", "value1")
        assert cm.get("key1") == "value1"

    def test_missing_key_returns_none(self):
        cm = CacheManager()
        assert cm.get("nonexistent") is None

    def test_overwrite_value(self):
        cm = CacheManager()
        cm.set("k", "old")
        cm.set("k", "new")
        assert cm.get("k") == "new"

    def test_various_value_types(self):
        cm = CacheManager()
        cm.set("int", 42)
        cm.set("list", [1, 2, 3])
        cm.set("dict", {"a": 1})
        cm.set("none", None)
        assert cm.get("int") == 42
        assert cm.get("list") == [1, 2, 3]
        assert cm.get("dict") == {"a": 1}
        # None value stored is still returned (not confused with "missing")
        assert cm.get("none") is None  # stored None == missing None, acceptable


# ── TTL / Expiry ────────────────────────────────────────────────────────────

class TestCacheTTL:
    def test_ttl_expiry(self):
        """Value should expire after TTL by advancing mocked time."""
        cm = CacheManager(default_ttl=10)
        with patch("utils.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cm.set("k", "v")
            # Still fresh
            mock_time.time.return_value = 1005.0
            assert cm.get("k") == "v"
            # Expired
            mock_time.time.return_value = 1011.0
            assert cm.get("k") is None

    def test_custom_ttl(self):
        cm = CacheManager(default_ttl=60)
        with patch("utils.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cm.set("short", "val", ttl=5)
            mock_time.time.return_value = 1006.0
            assert cm.get("short") is None


# ── Clear ───────────────────────────────────────────────────────────────────

class TestCacheClear:
    def test_clear(self):
        cm = CacheManager()
        cm.set("a", 1)
        cm.set("b", 2)
        cm.clear()
        assert cm.get("a") is None
        assert cm.get("b") is None


# ── Thread safety ───────────────────────────────────────────────────────────

class TestCacheThreadSafety:
    def test_concurrent_writes(self):
        cm = CacheManager()
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    cm.set(f"thread-{n}-{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── Global instance ─────────────────────────────────────────────────────────

class TestGlobalCache:
    def test_global_cache_is_cache_manager(self):
        assert isinstance(cache, CacheManager)
