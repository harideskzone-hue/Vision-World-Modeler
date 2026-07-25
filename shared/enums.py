# shared/enums.py
from enum import Enum
from typing import Dict, FrozenSet

class NodeType(Enum):
    SCENE = "scene"
    ENTITY = "entity"
    CHARACTER = "character"
    CAMERA = "camera"

class NodeStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"

class EdgeStatus(Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"

class RelationType(Enum):
    CONTAINS = "contains"
    HAS_STATE = "has_state"
    IS_TYPE = "is_type"
    LOCATED_IN = "located_in"
    CONNECTS_TO = "connects_to"
    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"

class ExtractionMethod(Enum):
    VLM = "vlm"
    YOLO = "yolo"
    FUSED = "fused"
    RULE = "rule"

class ContradictionType(Enum):
    EXPAND = "expand"
    CORROBORATE = "corroborate"
    REVISE = "revise"

class ConflictCategory(Enum):
    STATE_MUTUAL_EXCLUSION = "state_mutual_exclusion"
    FUNCTIONAL_EDGE_VIOLATION = "functional_edge_violation"

class RevisionActionType(Enum):
    EXPAND = "expand"
    CORROBORATE = "corroborate"
    SUPERSEDE = "supersede"
    PROVISIONAL_SUPERSEDE = "provisional_supersede"
