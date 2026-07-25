# tests/test_corrections.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.models import SceneObservation
from object_tracker.entity_registry import EntityRegistry
from object_tracker.entity_matcher import EntityMatcher
from object_tracker.candidate_converter import CandidateConverter
from object_tracker.occlusion_handler import OcclusionHandler
from world_model.graph_store import InMemoryGraphStore
from updater.updater import Updater
from shared.enums import NodeStatus

class MockPipeline:
    def __init__(self):
        self.registry = EntityRegistry()
        self.matcher = EntityMatcher(self.registry)
        self.converter = CandidateConverter()
        self.graph = InMemoryGraphStore()
        self.updater = Updater(self.graph)
        self.occlusion_handler = OcclusionHandler(self.graph, self.registry)

    def process(self, frame_id: int, scene: str, raw_entities: list):
        obs = SceneObservation(frame=frame_id, scene=scene, entities=raw_entities)
        matched_obs = self.matcher.match(obs)
        candidates = self.converter.convert(matched_obs)
        report = self.updater.update(candidates, frame_id)
        self.occlusion_handler.handle_occlusions()
        return report

def test_t_cor_01_contradictory_states():
    """T-COR-01: Contradictory state updates (R1/R2 Revision Policy)."""
    print("\n--- Running T-COR-01: Contradictory States ---")
    pipeline = MockPipeline()
    
    # Frame 1: Door is closed
    pipeline.process(1, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    door_id = pipeline.registry.get_id("door")
    edges_f1 = pipeline.graph.get_active_edges_for_entity(door_id)
    state_f1 = next((e.object for e in edges_f1 if e.relation.value == "has_state"), None)
    assert state_f1 == "closed"
    
    # Frame 2-4: Corroborate closed state (builds R2 threshold)
    pipeline.process(2, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    pipeline.process(3, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    pipeline.process(4, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    
    # Frame 5: VLM suddenly says 'open' (Contradiction!)
    report = pipeline.process(5, "hallway", [{"name": "door", "category": "architecture", "state": "open", "confidence": 0.85}])
    edges_f5 = pipeline.graph.get_active_edges_for_entity(door_id)
    state_f5 = next((e.object for e in edges_f5 if e.relation.value == "has_state"), None)
    
    assert state_f5 == "open" # Recency wins, but it should be provisional (R2)
    assert report.revised == 1
    
    # Check that the reason includes R2 provisional penalty because it had strong prior
    rev_action = report.revisions[0]
    assert "R2" in rev_action.revision_reason
    
    print("✔ T-COR-01 Passed: Conflicting state resolved correctly via R2.")

def test_t_cor_02_occlusion_reappearance():
    """T-COR-02: Occlusion followed by reappearance."""
    print("\n--- Running T-COR-02: Occlusion Reappearance ---")
    pipeline = MockPipeline()
    
    pipeline.process(1, "kitchen", [{"name": "mug", "category": "container", "confidence": 0.9}])
    mug_id = pipeline.registry.get_id("mug")
    
    # Frame 2-31: Camera looks away (mug occluded)
    for i in range(2, 35):
        pipeline.process(i, "kitchen", [{"name": "table", "category": "furniture", "confidence": 0.9}])
        
    # By frame 32, mug should be archived
    mug_node = pipeline.graph.get_node(mug_id)
    assert mug_node.status == NodeStatus.ARCHIVED
    assert pipeline.registry.is_archived(mug_id)
    
    # Frame 35: Mug reappears
    pipeline.process(35, "kitchen", [{"name": "mug", "category": "container", "confidence": 0.9}])
    
    # Check that ID was reactivated, not duplicated
    assert pipeline.registry.get_id("mug") == mug_id
    assert not pipeline.registry.is_archived(mug_id)
    assert pipeline.graph.get_node(mug_id).status == NodeStatus.ACTIVE
    
    print("✔ T-COR-02 Passed: Occluded object archived and reactivated successfully.")

def test_t_cor_03_scene_transition_r5():
    """T-COR-03: Scene transition triggers R5."""
    print("\n--- Running T-COR-03: Scene Transition (R5) ---")
    pipeline = MockPipeline()
    
    pipeline.process(1, "kitchen", [{"name": "mug", "category": "container", "confidence": 0.9}])
    mug_id = pipeline.registry.get_id("mug")
    
    # Scene change to hallway
    pipeline.process(2, "hallway", [{"name": "door", "category": "architecture", "confidence": 0.9}])
    
    # Verify R5 instantly archived the mug
    mug_node = pipeline.graph.get_node(mug_id)
    assert mug_node.status == NodeStatus.ARCHIVED
    
    print("✔ T-COR-03 Passed: R5 triggered instant archiving of previous scene objects.")

if __name__ == "__main__":
    print("==================================================")
    print("RUNNING CORRECTION TESTS (PHASE 6)")
    print("==================================================")
    test_t_cor_01_contradictory_states()
    test_t_cor_02_occlusion_reappearance()
    test_t_cor_03_scene_transition_r5()
    print("\nAll correction tests passed.")
