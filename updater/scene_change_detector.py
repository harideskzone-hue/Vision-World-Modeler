# updater/scene_change_detector.py
from shared.models import CandidateFact, RevisionAction
from shared.enums import RelationType, RevisionActionType, NodeStatus
from world_model.graph_store import InMemoryGraphStore
import logging

logger = logging.getLogger(__name__)

class SceneChangeDetector:
    """
    Implements R5 Rule: Scene Change Detection.
    Instantly archives objects from the previous scene instead of gradually decaying them.
    """
    def __init__(self, graph: InMemoryGraphStore):
        self.graph = graph

    def handle_scene_change(self, action: RevisionAction, frame_id: int):
        if not action.new_edge:
            return
            
        edge = action.new_edge
        if edge.subject == "camera" and edge.relation == RelationType.LOCATED_IN:
            if action.action_type in (RevisionActionType.SUPERSEDE, RevisionActionType.PROVISIONAL_SUPERSEDE):
                # Scene has changed! 
                # action.existing_edge_id points to the old camera location
                old_scene = None
                if action.existing_edge_id:
                    old_edge = self.graph._edges.get(action.existing_edge_id) # Using internal access for speed, though get_node/edges is better
                    if old_edge:
                        old_scene = old_edge.object
                        
                new_scene = edge.object
                logger.info(f"Scene change detected at frame {frame_id}: {old_scene} -> {new_scene}")
                
                # Archive all active entities not observed in this frame
                for node in self.graph.get_all_nodes():
                    if node.status == NodeStatus.ACTIVE and node.id != "camera":
                        # Wait, we want to archive entities that were in the old scene.
                        # Or simply archive all active nodes that haven't been seen in this frame.
                        if node.last_observed_frame < frame_id:
                            node.status = NodeStatus.ARCHIVED
                            logger.info(f"R5: Instantly archived {node.id} due to scene change.")
