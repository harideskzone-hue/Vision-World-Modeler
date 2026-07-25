# world_model/temporal_versioning.py
from typing import Optional
from shared.models import Edge
from shared.enums import EdgeStatus, ExtractionMethod

def close_validity(edge: Edge, frame_id: int, superseded_by_id: str, reason: str) -> Edge:
    edge.status = EdgeStatus.SUPERSEDED
    edge.t_valid_until = frame_id
    edge.superseded_by = superseded_by_id
    edge.revision_reason = reason
    return edge

def extend_validity(edge: Edge, frame_id: int, corroboration_boost: float, max_confidence: float) -> Edge:
    edge.corroboration_count += 1
    edge.t_valid_until = None
    edge.confidence = min(max_confidence, edge.confidence + corroboration_boost)
    return edge

def is_edge_active_at_frame(edge: Edge, frame_id: int) -> bool:
    if edge.t_valid_from > frame_id:
        return False
    if edge.t_valid_until is not None and frame_id >= edge.t_valid_until:
        return False
    return True

def create_replacement_edge(
    old_edge: Edge,
    new_object: str,
    new_confidence: float,
    frame_id: int,
    extraction_method: ExtractionMethod
) -> Edge:
    return Edge(
        subject=old_edge.subject,
        relation=old_edge.relation,
        object=new_object,
        confidence=new_confidence,
        extraction_method=extraction_method,
        source_frame_id=frame_id,
        t_observed=frame_id,
        t_valid_from=frame_id,
        t_valid_until=None,
        status=EdgeStatus.ACTIVE,
        corroboration_count=1
    )
