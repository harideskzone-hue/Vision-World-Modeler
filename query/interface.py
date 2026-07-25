# query/interface.py
from typing import List, Dict, Any, Optional
from world_model.graph_store import InMemoryGraphStore
from shared.enums import NodeStatus, RelationType
from object_tracker.entity_registry import EntityRegistry
import json

class QueryInterface:
    """
    Read-only query interface for the World Model.
    Supports structured queries and explanation mode.
    """
    def __init__(self, graph: InMemoryGraphStore, registry: EntityRegistry):
        self.graph = graph
        self.registry = registry

    def resolve_id(self, name: str) -> Optional[str]:
        # Tries to find the exact stable ID. If given base name, tries registry.
        if self.graph.get_node(name):
            return name
        return self.registry.get_id(name)

    def explain(self, name: str, frame_id: int = None) -> str:
        stable_id = self.resolve_id(name)
        if not stable_id:
            return f"Entity '{name}' not found."
            
        node = self.graph.get_node(stable_id)
        
        # Gather location
        loc_edges = [e for e in self.graph.get_active_edges_for_entity(stable_id) if e.relation == RelationType.LOCATED_IN]
        locations = ", ".join([e.object for e in loc_edges]) if loc_edges else "Unknown"
        
        # Gather state
        state_edges = [e for e in self.graph.get_active_edges_for_entity(stable_id) if e.relation == RelationType.HAS_STATE]
        states = ", ".join([e.object for e in state_edges]) if state_edges else "None"
        
        response = [
            f"{node.name.capitalize()} ({node.id})",
            "",
            f"Location:\n  {locations}",
            "",
            f"State:\n  {states}",
            "",
            f"Confidence:\n  {node.confidence:.2f}",
            "",
            f"Evidence:\n  Observed {node.observation_count} times. First seen: Frame {node.first_observed_frame}, Last seen: Frame {node.last_observed_frame}",
            "",
            f"Status:\n  {node.status.value.capitalize()}"
        ]
        return "\n".join(response)

    def get_objects_in_scene(self, scene_name: str) -> List[str]:
        scene_id = self.resolve_id(scene_name) or scene_name
        active_edges = self.graph.get_all_active_edges()
        
        objects = []
        for edge in active_edges:
            if edge.relation == RelationType.LOCATED_IN and edge.object == scene_id:
                if edge.subject != "camera":
                    objects.append(edge.subject)
        return objects

    def get_occluded_objects(self, current_frame: int) -> List[str]:
        occluded = []
        for node in self.graph.get_all_nodes():
            if node.status == NodeStatus.ACTIVE and node.last_observed_frame < current_frame:
                if node.id != "camera":
                    occluded.append(node.id)
        return occluded

    def get_archived_entities(self) -> List[str]:
        archived = []
        for node in self.graph.get_all_nodes():
            if node.status == NodeStatus.ARCHIVED:
                archived.append(node.id)
        return archived

    def what_changed_since(self, frame_id: int) -> Dict[str, Any]:
        new_nodes = [n.id for n in self.graph.get_all_nodes() if n.first_observed_frame >= frame_id]
        updated_nodes = [n.id for n in self.graph.get_all_nodes() if n.last_observed_frame >= frame_id and n.first_observed_frame < frame_id]
        
        new_edges = [e for e in self.graph.get_all_active_edges() if e.source_frame_id >= frame_id]
        
        return {
            "new_entities": new_nodes,
            "updated_entities": updated_nodes,
            "new_facts": [f"{e.subject} {e.relation.value} {e.object}" for e in new_edges]
        }
