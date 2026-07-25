# updater/contradiction_detector.py
from typing import Optional
from shared.models import CandidateFact, ContradictionResult
from shared.enums import ContradictionType, ConflictCategory, RelationType
from world_model.graph_store import InMemoryGraphStore
from world_model.schema import are_states_conflicting, is_functional_relation

class ContradictionDetector:
    def __init__(self, graph: InMemoryGraphStore):
        self._graph = graph

    def detect(self, candidate: CandidateFact) -> ContradictionResult:
        existing_edges = self._graph.get_active_edges_by_slot(
            candidate.subject, candidate.relation
        )

        if not existing_edges:
            return ContradictionResult(contradiction_type=ContradictionType.EXPAND)

        for existing in existing_edges:
            if existing.object == candidate.object:
                return ContradictionResult(
                    contradiction_type=ContradictionType.CORROBORATE,
                    existing_edge=existing,
                )
            
            if candidate.relation == RelationType.HAS_STATE:
                if are_states_conflicting(existing.object, candidate.object):
                    return ContradictionResult(
                        contradiction_type=ContradictionType.REVISE,
                        existing_edge=existing,
                        conflict_category=ConflictCategory.STATE_MUTUAL_EXCLUSION,
                    )
            
            if is_functional_relation(candidate.relation):
                # Any difference in object for a functional relation is a violation
                return ContradictionResult(
                    contradiction_type=ContradictionType.REVISE,
                    existing_edge=existing,
                    conflict_category=ConflictCategory.FUNCTIONAL_EDGE_VIOLATION,
                )
        
        return ContradictionResult(contradiction_type=ContradictionType.EXPAND)
