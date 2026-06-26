"""Strict relationship types for the Knowledge Graph. Used for Cypher and validation."""

from enum import Enum


class RelationType(str, Enum):
    """
    Canonical relationship types. Use only these when writing to Neo4j
    to avoid injection and ensure consistent schema.
    """

    TESTS = "TESTS"         
    USED_METHOD = "USED_METHOD"
    USED_SAMPLE = "USED_SAMPLE"
    YIELDED = "YIELDED"

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    INCONCLUSIVE = "INCONCLUSIVE"

    DESCRIBES_EXP = "DESCRIBES_EXP"
    MAKES_CLAIM = "MAKES_CLAIM"


JUDGE_RELATION_TYPES: frozenset[RelationType] = frozenset({RelationType.SUPPORTS, RelationType.CONTRADICTS, RelationType.INCONCLUSIVE})


def is_valid_relation_type(value: str) -> bool:
    """Validate that a string is a known RelationType (prevents Cypher injection)."""
    try:
        RelationType(value)
        return True
    except ValueError:
        return False
