# updater/updater.py
from typing import List
from shared.models import CandidateFact, Node, UpdateReport, RevisionAction
from shared.enums import NodeType, NodeStatus, RevisionActionType
from world_model.graph_store import InMemoryGraphStore
from world_model.schema import normalize_entity_name
from updater.contradiction_detector import ContradictionDetector
from updater.revision_policy import RevisionPolicy
from updater.scene_change_detector import SceneChangeDetector

class Updater:
    def __init__(self, graph: InMemoryGraphStore):
        self._graph = graph
        self._detector = ContradictionDetector(graph)
        self._policy = RevisionPolicy()
        self._scene_detector = SceneChangeDetector(graph)

    def update(self, candidates: List[CandidateFact], frame_id: int) -> UpdateReport:
        report = UpdateReport(frame_id=frame_id)

        # Apply decay to occluded nodes before processing new facts
        self._graph.update_node_confidence_occlusion(frame_id)

        for candidate in candidates:
            try:
                action = self._process_candidate(candidate, frame_id)
                self._execute_action(action, frame_id)
                self._update_report(report, action)
            except Exception as e:
                print(f"Failed to process candidate {candidate}: {e}")
                report.rejected += 1

        return report

    def _process_candidate(self, candidate: CandidateFact, frame_id: int) -> RevisionAction:
        self._ensure_node(candidate.subject, frame_id)
        self._ensure_node(candidate.object, frame_id, is_state_value=(
            candidate.relation.value == "has_state"
        ))

        result = self._detector.detect(candidate)
        action = self._policy.resolve(result, candidate, frame_id)
        return action

    def _execute_action(self, action: RevisionAction, frame_id: int) -> None:
        if action.action_type == RevisionActionType.EXPAND:
            if action.new_edge:
                self._graph.add_edge(action.new_edge)

        elif action.action_type == RevisionActionType.CORROBORATE:
            if action.existing_edge_id:
                self._graph.corroborate_edge(action.existing_edge_id, frame_id)

        elif action.action_type in (
            RevisionActionType.SUPERSEDE,
            RevisionActionType.PROVISIONAL_SUPERSEDE,
        ):
            if action.existing_edge_id and action.new_edge:
                self._graph.supersede_edge(
                    action.existing_edge_id,
                    action.new_edge,
                    action.revision_reason,
                )
                self._scene_detector.handle_scene_change(action, frame_id)

    def _update_report(self, report: UpdateReport, action: RevisionAction) -> None:
        if action.action_type == RevisionActionType.EXPAND:
            report.expanded += 1
        elif action.action_type == RevisionActionType.CORROBORATE:
            report.corroborated += 1
        elif action.action_type in (
            RevisionActionType.SUPERSEDE,
            RevisionActionType.PROVISIONAL_SUPERSEDE,
        ):
            report.revised += 1
            report.revisions.append(action)

    def _ensure_node(self, name: str, frame_id: int, is_state_value: bool = False) -> None:
        if is_state_value:
            return

        normalized = normalize_entity_name(name)
        existing = self._graph.get_node(normalized)
        
        # Determine node type from context (hacky for now, Tracker will handle better later)
        if name == "camera":
            node_type = NodeType.CAMERA
        elif existing is None:
            # We assume it's an entity by default unless we know it's a scene
            # Real scene typing happens via vision extractor
            node_type = NodeType.ENTITY
        else:
            node_type = existing.node_type
            
        if existing is None:
            self._graph.add_node(Node(
                id=normalized,
                name=name,
                node_type=node_type,
                confidence=0.55,
                status=NodeStatus.ACTIVE,
                first_observed_frame=frame_id,
                last_observed_frame=frame_id,
                observation_count=1,
                category="unknown"
            ))
        else:
            existing.last_observed_frame = frame_id
            if existing.status == NodeStatus.ARCHIVED:
                existing.status = NodeStatus.ACTIVE
                existing.confidence = 0.50
