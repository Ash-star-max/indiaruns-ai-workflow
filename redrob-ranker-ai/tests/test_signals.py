"""
Tests for src/signals.py

Covers:
    - Fully engaged candidate → behavioral_multiplier near 1.0
    - Inactive candidate (last_active > 90d, open_to_work=False) → capped at 0.30
    - offer_acceptance_rate = -1 → treated as neutral
    - github_activity_score = -1 → treated as neutral
    - recruiter_response_rate = 0.0 → low engagement score
"""

from __future__ import annotations

import pytest

# TODO: implement tests in Phase 2 after signals.py is complete


def test_placeholder():
    """Placeholder — replace with real tests when signals.py is implemented."""
    assert True
