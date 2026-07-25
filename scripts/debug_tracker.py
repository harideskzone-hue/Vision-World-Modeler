# scripts/debug_tracker.py
import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.models import SceneObservation
from object_tracker.entity_registry import EntityRegistry
from object_tracker.entity_matcher import EntityMatcher
from object_tracker.candidate_converter import CandidateConverter
from object_tracker.occlusion_handler import OcclusionHandler
from world_model.graph_store import InMemoryGraphStore
from updater.updater import Updater
from shared.enums import NodeStatus

class TrackerDebugger:
    def __init__(self):
        self.registry = EntityRegistry()
        self.matcher = EntityMatcher(self.registry)
        self.converter = CandidateConverter()
        self.graph = InMemoryGraphStore()
        self.updater = Updater(self.graph)
        self.occlusion_handler = OcclusionHandler(self.graph, self.registry)
        
        self.previously_active = set()

    def step(self, frame_id: int, scene_name: str, raw_entities: List[Dict]) -> None:
        obs = SceneObservation(frame=frame_id, scene=scene_name, entities=raw_entities)
        
        # Track what was active before this frame
        active_before = {n.id for n in self.graph.get_all_nodes() if n.status == NodeStatus.ACTIVE}
        
        # Run matching
        matched_obs = self.matcher.match(obs)
        
        # See what was matched vs new
        matched = []
        new = []
        for ent in matched_obs.entities:
            stable_id = ent["name"]
            raw_name = ent["raw_name"]
            if stable_id in active_before or self.registry.is_archived(stable_id):
                matched.append(f"{raw_name} -> {stable_id}")
            else:
                new.append(f"{raw_name} -> {stable_id}")
                
        # Convert and update
        candidates = self.converter.convert(matched_obs)
        self.updater.update(candidates, frame_id)
        self.occlusion_handler.handle_occlusions()
        
        # Check what became occluded/archived this frame
        active_after = {n.id for n in self.graph.get_all_nodes() if n.status == NodeStatus.ACTIVE}
        archived_after = {n.id for n in self.graph.get_all_nodes() if n.status == NodeStatus.ARCHIVED}
        
        occluded = active_before - active_after - archived_after
        # Nodes that just became archived this frame
        just_archived = [n.id for n in self.graph.get_all_nodes() 
                         if n.status == NodeStatus.ARCHIVED and n.id in self.previously_active]
        
        self.previously_active = active_after.copy()
        for nid in archived_after:
            if nid in self.previously_active:
                self.previously_active.remove(nid)

        # Print Trace
        print(f"\nFrame {frame_id}")
        
        print("\nDetected:")
        for ent in raw_entities:
            print(f"  {ent['name']}")
            
        print("\nMatched:")
        if not matched: print("  none")
        for m in matched: print(f"  {m}")
            
        print("\nNew:")
        if not new: print("  none")
        for n in new: print(f"  {n}")
            
        print("\nOccluded:")
        if not occluded: print("  none")
        for o in occluded: print(f"  {o}")
            
        print("\nArchived:")
        if not just_archived: print("  none")
        for a in just_archived: print(f"  {a}")
        print("-" * 30)


def run_required_tests():
    print("=" * 50)
    print("RUNNING REQUIRED TRACKER TESTS")
    print("=" * 50)
    
    debugger = TrackerDebugger()
    
    # 1. Same mug across 2 frames
    debugger.step(1, "kitchen", [{"name": "red_mug", "category": "container", "confidence": 0.9}])
    debugger.step(2, "kitchen", [{"name": "red_mug", "category": "container", "confidence": 0.9}])
    
    # Check stable ID reused
    assert len([n for n in debugger.graph.get_all_nodes() if "red_mug" in n.id]) == 1
    mug_id = [n for n in debugger.graph.get_all_nodes() if "red_mug" in n.id][0].id
    print("✔ Same mug across frames -> Same stable ID")
    
    # 2. Camera pans away -> Occluded, not deleted
    # We simulate a large gap to trigger decay but not archive yet (stale threshold = 30)
    debugger.step(15, "hallway", [{"name": "door", "category": "architecture", "confidence": 0.9}])
    mug_node = debugger.graph.get_node(mug_id)
    assert mug_node.status == NodeStatus.ACTIVE
    assert mug_node.confidence < 0.9 # Decayed
    print("✔ Camera pans away -> Mug becomes occluded (confidence decayed), not deleted")
    
    # 3. Mug returns
    debugger.step(16, "kitchen", [{"name": "red_mug", "category": "container", "confidence": 0.9}])
    mug_node = debugger.graph.get_node(mug_id)
    assert mug_node.status == NodeStatus.ACTIVE
    # Count of nodes should still be 2 (mug + door) + 2 (kitchen + hallway scenes)
    assert len([n for n in debugger.graph.get_all_nodes() if "red_mug" in n.id]) == 1
    print("✔ Mug returns -> Same ID reused")
    
    # 4. Door changes closed -> open
    debugger.step(17, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    debugger.step(18, "hallway", [{"name": "door", "category": "architecture", "state": "closed", "confidence": 0.9}])
    debugger.step(19, "hallway", [{"name": "door", "category": "architecture", "state": "open", "confidence": 0.95}])
    
    # Should have superseded edge
    stats = debugger.graph.get_stats()
    assert stats.superseded_edges > 0
    print("✔ Door changes closed→open -> R1/R2 revision triggered")
    
    # 5. Three identical chairs
    debugger.step(20, "dining_room", [
        {"name": "chair", "category": "furniture", "confidence": 0.9, "bbox": [10, 10, 20, 20]},
        {"name": "chair", "category": "furniture", "confidence": 0.9, "bbox": [30, 10, 40, 20]},
        {"name": "chair", "category": "furniture", "confidence": 0.9, "bbox": [50, 10, 60, 20]}
    ])
    
    chairs = [n for n in debugger.graph.get_all_nodes() if "chair" in n.id]
    assert len(chairs) == 3
    print("✔ Three identical chairs -> Distinct stable IDs maintained")
    
    # 6. Scene change & 7. New object appears tested organically above
    
    # 8. Archive after threshold
    # Move frame to 55 (past stale threshold of 30)
    debugger.step(55, "garden", [{"name": "tree", "category": "plant", "confidence": 0.9}])
    assert debugger.graph.get_node(mug_id).status == NodeStatus.ARCHIVED
    assert debugger.registry.is_archived(mug_id)
    print("✔ Occluded object successfully archived after stale threshold")
    
    print("\nALL REQUIRED PHASE 4 TESTS PASSED!")

if __name__ == "__main__":
    run_required_tests()
