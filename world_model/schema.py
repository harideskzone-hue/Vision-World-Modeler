# world_model/schema.py
from typing import Set, Dict, FrozenSet
from shared.enums import RelationType

STATE_CONFLICTS: Dict[str, FrozenSet[str]] = {
    "open_closed": frozenset({"open", "closed"}),
    "switch": frozenset({"on", "off"}),
    "fill": frozenset({"empty", "full", "half_full"})
}

def are_states_conflicting(state_a: str, state_b: str) -> bool:
    if state_a == state_b:
        return False
    for group in STATE_CONFLICTS.values():
        if state_a in group and state_b in group:
            return True
    return False

def normalize_entity_name(name: str) -> str:
    articles = {"the", "a", "an"}
    words = name.strip().lower().split()
    words = [w for w in words if w not in articles]
    base_name = " ".join(words) if words else name.strip().lower()
    return base_name.replace(" ", "_")

def is_functional_relation(relation: RelationType) -> bool:
    """
    Check if a relation type is functional (at most 1 active edge per subject).
    """
    return relation in {RelationType.LOCATED_IN, RelationType.HAS_STATE, RelationType.IS_TYPE}
