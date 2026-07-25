# world_model/schema.py
from typing import Set, Dict, FrozenSet
from shared.enums import RelationType

STATE_CONFLICTS: Dict[str, FrozenSet[str]] = {
    "portal": frozenset({"open", "closed", "ajar", "locked", "unlocked", "sealed"}),
    "power": frozenset({"on", "off", "standby", "powered_off", "powered_on"}),
    "occupancy": frozenset({"occupied", "vacant", "unoccupied", "empty", "crowded"}),
    "cleanliness": frozenset({"clean", "dirty", "filthy", "pristine", "stained", "dusty"}),
    "moisture": frozenset({"wet", "dry", "damp", "soaked", "arid"}),
    "integrity": frozenset({"broken", "intact", "damaged", "operational", "shattered", "working"}),
    "motion": frozenset({"moving", "stationary", "stopped", "running", "still", "static"}),
    "posture": frozenset({"standing", "seated", "lying", "collapsed", "upright", "tilted", "inverted"}),
    "illumination": frozenset({"illuminated", "dark", "dim", "bright", "lit", "unlit"}),
    "temperature": frozenset({"hot", "cold", "warm", "freezing", "boiling", "chilled"}),
    "fill_level": frozenset({"empty", "full", "half_full", "overflowing", "nearly_empty", "depleted"}),
    "visibility": frozenset({"visible", "hidden", "occluded", "obscured", "unseen", "exposed"}),
    "security": frozenset({"secured", "unsecured", "locked", "unlocked", "alarmed", "disarmed"}),
    "alignment": frozenset({"aligned", "misaligned", "crooked", "straight", "tilted", "centered"}),
    "connection": frozenset({"connected", "disconnected", "attached", "detached", "unplugged", "plugged_in"})
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
