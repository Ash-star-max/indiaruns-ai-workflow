"""
Tests for src/features.py

Covers:
    - Feature vector has expected shape and column names
    - Consulting-only candidate → consulting_only_flag == 1
    - Non-technical title → nontechnical_title_flag == 1
    - Tier-1 skill count is correct for a known candidate
    - Location score is 1.0 for India + preferred city
    - Experience fit Gaussian peaks at 7 years
"""

from __future__ import annotations

import pytest

# TODO: implement tests in Phase 2 after features.py is complete


def test_placeholder():
    """Placeholder — replace with real tests when features.py is implemented."""
    assert True
