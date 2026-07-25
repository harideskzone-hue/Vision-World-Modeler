# world_model/graph_store.py
import json
import sys
from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple

from shared.models import Node, Edge, GraphStats
from shared.enums import NodeStatus, EdgeStatus, RelationType
from world_model.schema import normalize_entity_name
from world_model.temporal_versioning import close_validity, extend_validity, is_edge_active_at_frame
from shared.config import DEFAULT_CONFIG

class InMemoryGraphStore:
    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        
        self._subject_index: Dict[str, Set[str]] = defaultdict(set)
        self._object_index: Dict[str, Set[str]] = defaultdict(set)
        self._slot_index: Dict[Tuple[str, RelationType], Set[str]] = defaultdict(set)

    def add_node(self, node: Node) -> str:
        node.id = normalize_entity_name(node.id)
        if node.id in self._nodes:
            existing = self._nodes[node.id]
            existing.last_observed_frame = max(existing.last_observed_frame, node.last_observed_frame)
            existing.observation_count += 1
            existing.confidence = min(
                DEFAULT_CONFIG.updater.max_confidence, 
                existing.confidence + DEFAULT_CONFIG.updater.corroboration_boost
            )
            if node.status == NodeStatus.ACTIVE:
                existing.status = NodeStatus.ACTIVE
            return existing.id
        else:
            self._nodes[node.id] = node
            return node.id

    def update_node_confidence_occlusion(self, frame_id: int):
        """Decay confidence of nodes that weren't seen this frame."""
        for node in self._nodes.values():
            if node.status == NodeStatus.ACTIVE and node.last_observed_frame < frame_id:
                if node.observation_count >= 1:
                    node.confidence = max(
                        DEFAULT_CONFIG.updater.min_confidence,
                        node.confidence + DEFAULT_CONFIG.updater.occlusion_decay
                    )
                if frame_id - node.last_observed_frame >= DEFAULT_CONFIG.graph.stale_threshold_frames:
                    node.status = NodeStatus.ARCHIVED

    def add_edge(self, edge: Edge) -> str:
        self._edges[edge.id] = edge
        self._subject_index[edge.subject].add(edge.id)
        self._object_index[edge.object].add(edge.id)
        self._slot_index[(edge.subject, edge.relation)].add(edge.id)
        return edge.id

    def supersede_edge(self, edge_id: str, new_edge: Edge, reason: str) -> str:
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id} not found in graph store")

        old_edge = self._edges[edge_id]
        close_validity(
            old_edge,
            frame_id=new_edge.t_observed,
            superseded_by_id=new_edge.id,
            reason=reason,
        )
        self.add_edge(new_edge)
        return new_edge.id

    def corroborate_edge(self, edge_id: str, frame_id: int) -> None:
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id} not found in graph store")
        extend_validity(
            self._edges[edge_id], 
            frame_id, 
            DEFAULT_CONFIG.updater.corroboration_boost,
            DEFAULT_CONFIG.updater.max_confidence
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(normalize_entity_name(node_id))

    def get_all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def get_active_edges_for_entity(self, entity_id: str) -> List[Edge]:
        entity_id = normalize_entity_name(entity_id)
        edge_ids = self._subject_index.get(entity_id, set()) | \
                   self._object_index.get(entity_id, set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and self._edges[eid].status == EdgeStatus.ACTIVE
        ]

    def get_active_edges_by_slot(self, subject: str, relation: RelationType) -> List[Edge]:
        subject = normalize_entity_name(subject)
        edge_ids = self._slot_index.get((subject, relation), set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and self._edges[eid].status == EdgeStatus.ACTIVE
        ]

    def get_edges_at_frame(self, entity_id: str, frame_id: int) -> List[Edge]:
        entity_id = normalize_entity_name(entity_id)
        edge_ids = self._subject_index.get(entity_id, set()) | \
                   self._object_index.get(entity_id, set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and is_edge_active_at_frame(self._edges[eid], frame_id)
        ]

    def get_all_active_edges(self) -> List[Edge]:
        return [e for e in self._edges.values() if e.status == EdgeStatus.ACTIVE]

    def serialize(self) -> str:
        # Simplistic serialization for human readability and inspection
        def node_to_dict(n: Node) -> dict:
            label = "CONFIRMED" if n.confidence >= DEFAULT_CONFIG.updater.confirmed_threshold else (
                    "PROBABLE" if n.confidence >= 0.50 else (
                    "UNCERTAIN" if n.confidence >= 0.25 else "UNLIKELY"))
            return {
                "id": n.id,
                "name": n.name,
                "type": n.node_type.value,
                "category": n.category,
                "confidence": round(n.confidence, 4),
                "label": label,
                "first_seen": n.first_observed_frame,
                "last_seen": n.last_observed_frame,
                "times_observed": n.observation_count
            }

        def edge_to_dict(e: Edge) -> dict:
            return {
                "id": e.id,
                "fact": f"{e.subject} {e.relation.value} {e.object}",
                "confidence": round(e.confidence, 4),
                "corroborations": e.corroboration_count,
                "valid_from": e.t_valid_from,
                "valid_until": e.t_valid_until,
                "status": e.status.value,
                "superseded_by": e.superseded_by,
                "reason": e.revision_reason
            }

        data = {
            "nodes": [node_to_dict(n) for n in self._nodes.values()],
            "edges": [edge_to_dict(e) for e in self._edges.values()],
        }
        return json.dumps(data, indent=2)

    def export_json(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.serialize())

    def export_graphml(self, path: str) -> None:
        # Simple GraphML representation
        with open(path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            f.write('  <graph id="G" edgedefault="directed">\n')
            
            # Nodes
            for n in self._nodes.values():
                if n.status == NodeStatus.ACTIVE:
                    f.write(f'    <node id="{n.id}"/>\n')
                    
            # Edges
            for e in self._edges.values():
                if e.status == EdgeStatus.ACTIVE:
                    f.write(f'    <edge id="{e.id}" source="{e.subject}" target="{e.object}">\n')
                    f.write(f'      <data key="relation">{e.relation.value}</data>\n')
                    f.write(f'    </edge>\n')
                    
            f.write('  </graph>\n')
            f.write('</graphml>\n')

    def export_dot(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write("digraph WorldModel {\n")
            f.write('  rankdir=LR;\n')
            f.write('  node [shape=box, style=filled, fillcolor=lightgray];\n')
            
            for n in self._nodes.values():
                if n.status == NodeStatus.ACTIVE:
                    f.write(f'  "{n.id}" [label="{n.name}\\n({n.confidence:.2f})"];\n')
                    
            for e in self._edges.values():
                if e.status == EdgeStatus.ACTIVE:
                    f.write(f'  "{e.subject}" -> "{e.object}" [label="{e.relation.value}\\n({e.confidence:.2f})"];\n')
                    
            f.write("}\n")

    def get_stats(self) -> GraphStats:
        active = sum(1 for e in self._edges.values() if e.status == EdgeStatus.ACTIVE)
        superseded = sum(1 for e in self._edges.values() if e.status == EdgeStatus.SUPERSEDED)
        confidences = [e.confidence for e in self._edges.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        storage = sys.getsizeof(self.serialize())

        return GraphStats(
            total_nodes=len(self._nodes),
            total_edges=len(self._edges),
            active_edges=active,
            superseded_edges=superseded,
            storage_bytes=storage,
            avg_confidence=round(avg_conf, 4),
        )
