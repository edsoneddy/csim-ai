"""Shared types for the L1-L6 synthetic plagiarism generator."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MutationResult:
    level: str  # "L1".."L6"
    rule: str  # which rule(s) fired, e.g. "strip_docstrings+strip_comments"
    code: str | None  # transformed source, or None/unchanged if nothing fired
    applied: bool  # did at least one rule actually change the code
