# =============================================================
# ratelimit.py — Rate limiting shared by the API and the dashboard
#
# With REDIS_URL set (e.g. redis://redis:6379/0), counters live in Redis, so
# limits hold across every replica/worker. Without it, counters are in this
# process's memory: fine for one Streamlit Cloud app or one uvicorn worker.
#
# Fixed-window counters: at most `limit` hits per key per `window_s` seconds.
# =============================================================

import os
import threading
import time

_local, _lock = {}, threading.Lock()
_redis = None


def _client():
    global _redis
    if _redis is None and os.getenv("REDIS_URL"):
        import redis
        _redis = redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=2)
    return _redis


def hit(key, limit, window_s):
    """Records one hit for `key`; returns True if it is within the limit."""
    bucket = f"pisa:rl:{key}:{int(time.time() // window_s)}"
    r = _client()
    if r is not None:
        try:
            pipe = r.pipeline()
            pipe.incr(bucket)
            pipe.expire(bucket, window_s + 5)
            count = pipe.execute()[0]
            return count <= limit
        except Exception as e:  # Redis down → fall back to local counting rather than block everyone
            print(f"   ⚠️  Redis rate limiter unavailable ({e}); using in-memory limits")
    with _lock:
        if len(_local) > 50_000:  # many distinct IPs: drop windows that have already ended
            for k in [k for k in _local if not k.endswith(bucket.rsplit(":", 1)[1])]:
                del _local[k]
        local_key = f"{key}:{window_s}:{bucket.rsplit(':', 1)[1]}"
        _local[local_key] = _local.get(local_key, 0) + 1
        return _local[local_key] <= limit


def reset():
    """Clears in-memory counters (used by tests)."""
    with _lock:
        _local.clear()
