"""
parser.py — Streaming parser for candidates.jsonl

Handles 100K+ candidate records without loading all into memory at once.
Uses ijson for streaming and Pydantic for validation.
"""

from __future__ import annotations

# TODO: implement in Phase 2
# Planned exports:
#   - CandidateRecord (Pydantic model mirroring candidate_schema.json)
#   - iter_candidates(path: Path) -> Iterator[CandidateRecord]
#   - load_candidates(path: Path) -> list[CandidateRecord]
