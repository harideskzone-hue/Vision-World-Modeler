# shared/models.py
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from shared.enums import (
    NodeType, NodeStatus, EdgeStatus, RelationType,
    ExtractionMethod, ContradictionType, ConflictCategory,
    RevisionActionType
)

@dataclass
class Node:
    id: str
    name: str
    node_type: NodeType
    confidence: float
    status: NodeStatus
    first_observed_frame: int
    last_observed_frame: int
    observation_count: int
    category: str

@dataclass
class Edge:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str = ""
    relation: RelationType = RelationType.CONTAINS
    object: str = ""
    confidence: float = 0.5
    extraction_method: ExtractionMethod = ExtractionMethod.VLM
    source_frame_id: int = 0
    t_observed: int = 0
    t_valid_from: int = 0
    t_valid_until: Optional[int] = None
    status: EdgeStatus = EdgeStatus.ACTIVE
    corroboration_count: int = 1
    superseded_by: Optional[str] = None
    revision_reason: Optional[str] = None

@dataclass
class CandidateFact:
    subject: str
    relation: RelationType
    object: str
    confidence: float
    source_frame_id: int
    extraction_method: ExtractionMethod

@dataclass
class SceneObservation:
    frame: int
    scene: str
    entities: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ContradictionResult:
    contradiction_type: ContradictionType
    existing_edge: Optional[Edge] = None
    conflict_category: Optional[ConflictCategory] = None

@dataclass
class RevisionAction:
    action_type: RevisionActionType
    existing_edge_id: Optional[str] = None
    new_edge: Optional[Edge] = None
    revision_reason: str = ""

@dataclass
class UpdateReport:
    frame_id: int = 0
    expanded: int = 0
    corroborated: int = 0
    revised: int = 0
    rejected: int = 0
    revisions: List[RevisionAction] = field(default_factory=list)

@dataclass
class GraphStats:
    total_nodes: int = 0
    total_edges: int = 0
    active_edges: int = 0
    superseded_edges: int = 0
    storage_bytes: int = 0
    avg_confidence: float = 0.0
