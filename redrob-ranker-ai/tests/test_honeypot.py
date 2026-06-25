"""
Tests for src/honeypot.py

Covers:
    - Clean candidate → not flagged
    - Expert skill with 0 duration × 3 → flagged
    - Salary min > max → flagged
    - Career months >> years_of_experience → flagged
    - Graduation year > career start year → flagged
    - False positive rate on sample_candidates.json is low
"""

from __future__ import annotations

import pytest

# TODO: implement tests in Phase 2 after honeypot.py is complete


def test_placeholder():
    """Placeholder — replace with real tests when honeypot.py is implemented."""
    assert True
