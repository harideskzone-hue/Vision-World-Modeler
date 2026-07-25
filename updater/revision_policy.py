# updater/revision_policy.py
from shared.models import CandidateFact, Edge, ContradictionResult, RevisionAction
from shared.enums import ContradictionType, RevisionActionType, EdgeStatus
from world_model.temporal_versioning import create_replacement_edge
from shared.config import DEFAULT_CONFIG

class RevisionPolicy:
    def resolve(
        self,
        result: ContradictionResult,
        candidate: CandidateFact,
        frame_id: int,
    ) -> RevisionAction:
        if result.contradiction_type == ContradictionType.EXPAND:
            new_edge = self._candidate_to_edge(candidate, frame_id)
            return RevisionAction(
                action_type=RevisionActionType.EXPAND,
                new_edge=new_edge,
                revision_reason=f"new fact at frame {frame_id}",
            )

        if result.contradiction_type == ContradictionType.CORROBORATE:
            return RevisionAction(
                action_type=RevisionActionType.CORROBORATE,
                existing_edge_id=result.existing_edge.id if result.existing_edge else None,
                revision_reason=f"corroborated at frame {frame_id}",
            )

        existing = result.existing_edge
        if existing is None:
            new_edge = self._candidate_to_edge(candidate, frame_id)
            return RevisionAction(
                action_type=RevisionActionType.EXPAND,
                new_edge=new_edge,
                revision_reason=f"revision fallback at frame {frame_id}",
            )

        threshold = DEFAULT_CONFIG.updater.corroboration_threshold
        penalty = DEFAULT_CONFIG.updater.r2_confidence_penalty

        if existing.corroboration_count >= threshold:
            # R2: Corroboration-Override
            penalized_confidence = max(
                DEFAULT_CONFIG.updater.min_confidence,
                candidate.confidence + penalty,
            )
            new_edge = self._candidate_to_edge(
                candidate, frame_id, override_confidence=penalized_confidence,
            )
            reason = (
                f"R2: provisional supersede at frame {frame_id} — "
                f"old had {existing.corroboration_count} corroborations, "
                f"confidence penalized by {abs(penalty)}"
            )
            return RevisionAction(
                action_type=RevisionActionType.PROVISIONAL_SUPERSEDE,
                existing_edge_id=existing.id,
                new_edge=new_edge,
                revision_reason=reason,
            )
        else:
            # R1: Recency-Wins
            new_edge = self._candidate_to_edge(candidate, frame_id)
            reason = (
                f"R1: recency-wins supersede at frame {frame_id} — "
                f"old corroboration={existing.corroboration_count} < threshold={threshold}"
            )
            return RevisionAction(
                action_type=RevisionActionType.SUPERSEDE,
                existing_edge_id=existing.id,
                new_edge=new_edge,
                revision_reason=reason,
            )

    def _candidate_to_edge(
        self,
        candidate: CandidateFact,
        frame_id: int,
        override_confidence: float | None = None,
    ) -> Edge:
        return Edge(
            subject=candidate.subject,
            relation=candidate.relation,
            object=candidate.object,
            confidence=override_confidence if override_confidence is not None else candidate.confidence,
            source_frame_id=frame_id,
            extraction_method=candidate.extraction_method,
            t_observed=frame_id,
            t_valid_from=frame_id,
            t_valid_until=None,
            status=EdgeStatus.ACTIVE,
            corroboration_count=1
        )
