"""E9 efficiency measurements (proposal: requirement R5), reusable by every model.

    from emotionshift.eval.efficiency import latency_ms, file_size_mb, machine
    latency_ms(model.predict, test_texts)   # per-utterance CPU latency

Latency is measured the way the demo uses a model: one utterance per call,
after a warm-up, timed with ``time.perf_counter``. Report the median and p95.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np


def latency_ms(predict_one_batch, texts, n: int = 500, warmup: int = 20, seed: int = 0) -> dict:
    """Per-utterance latency of ``predict_one_batch([text])`` in milliseconds."""
    rng = np.random.default_rng(seed)
    texts = list(texts)
    pick = [texts[i] for i in rng.choice(len(texts), size=min(n, len(texts)), replace=False)]
    for t in pick[:warmup]:
        predict_one_batch([t])
    times = []
    for t in pick:
        t0 = time.perf_counter()
        predict_one_batch([t])
        times.append((time.perf_counter() - t0) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def throughput(predict_batch, texts) -> dict:
    """Utterances per second when predicting a whole split in one call."""
    texts = list(texts)
    t0 = time.perf_counter()
    predict_batch(texts)
    sec = time.perf_counter() - t0
    return {"n": len(texts), "seconds": sec, "utterances_per_second": len(texts) / sec}


def file_size_mb(path) -> float:
    return Path(path).stat().st_size / 1e6


def machine() -> dict:
    """Where the numbers were measured (they depend on the CPU)."""
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}
