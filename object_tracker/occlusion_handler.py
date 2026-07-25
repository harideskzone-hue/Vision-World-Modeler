# object_tracker/occlusion_handler.py
from world_model.graph_store import InMemoryGraphStore
from object_tracker.entity_registry import EntityRegistry
from shared.enums import NodeStatus

class OcclusionHandler:
    """
    Synchronizes occlusion states between the Graph Store and Entity Registry.
    When an object is ARCHIVED in the graph (due to prolonged occlusion),
    it is also archived in the registry to prevent accidental rebinding.
    """
    def __init__(self, graph: InMemoryGraphStore, registry: EntityRegistry):
        self.graph = graph
        self.registry = registry

    def handle_occlusions(self):
        """Finds all archived nodes in the graph and mirrors the state in the registry."""
        nodes = self.graph.get_all_nodes()
        for node in nodes:
            if node.status == NodeStatus.ARCHIVED and not self.registry.is_archived(node.id):
                self.registry.archive(node.id)
            elif node.status == NodeStatus.ACTIVE and self.registry.is_archived(node.id):
                self.registry.reactivate(node.id)
