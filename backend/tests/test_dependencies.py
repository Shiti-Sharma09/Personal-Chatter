"""Regression coverage for dependencies.py's concurrency handling.

llama-cpp-python's Llama isn't safe for concurrent generate calls from
multiple threads on the same instance, but FastAPI's sync `def`
endpoints run on a threadpool -- two browser tabs, or a double-click on
Send, would otherwise call into the same shared model simultaneously.
"""
import threading
import time

from app.dependencies import _SerializedLLM


class _SlowLLM:
    """Records how many callers are inside stream() at once, so a test
    can prove they never overlap."""

    def __init__(self):
        self.concurrent = 0
        self.max_concurrent = 0
        self._state_lock = threading.Lock()

    def stream(self, messages, stop=None):
        with self._state_lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        time.sleep(0.05)
        with self._state_lock:
            self.concurrent -= 1
        yield "ok"

    def invoke(self, messages, stop=None):
        return "ok"


def test_serialized_llm_prevents_concurrent_generation():
    inner = _SlowLLM()
    llm = _SerializedLLM(inner, threading.Lock())

    def consume():
        list(llm.stream([{"role": "user", "content": "hi"}]))

    threads = [threading.Thread(target=consume) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert inner.max_concurrent == 1
