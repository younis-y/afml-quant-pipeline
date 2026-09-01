"""
Cache Manager
Simple in-memory cache with TTL for market data.
"""

import time
from typing import Dict, Any, Optional
from threading import Lock

class CacheManager:
    def __init__(self, default_ttl: int = 60):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and hasn't expired."""
        with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if time.time() < item['expires_at']:
                    return item['value']
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL."""
        ttl = ttl or self._default_ttl
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }
    
    def clear(self) -> None:
        """Clear all cache items."""
        with self._lock:
            self._cache.clear()

# Global cache instance
cache = CacheManager()
