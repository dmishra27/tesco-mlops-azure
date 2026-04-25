"""
E2E test fixtures. Generates a small, signal-rich dataset and runs the
pipeline once per session (cached). Uses n_optuna_trials=5 for speed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


FAST_CONFIG = {
    "n_customers":      1_500,
    "n_transactions":   15_000,
    "n_optuna_trials":  20,
    "seed":             42,
    "nonlinear":        True,   # inject AND-gate signal so tree models win clearly
    "out_dir":          "data/e2e",
    "kmeans_n_init":    20,
}


@pytest.fixture(scope="session")
def e2e_result():
    """Runs the full pipeline with fast config. Returns the results dict."""
    from ml.local.run_pipeline import run_pipeline
    result = run_pipeline(config=FAST_CONFIG)
    return result
