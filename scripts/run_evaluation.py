# scripts/run_evaluation.py
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
from evaluation.evaluator import Evaluator
from shared.enums import EdgeStatus, RelationType

def run_evaluation():
    print("Setting up pipeline...")
    registry = EntityRegistry()
    matcher = EntityMatcher(registry)
    converter = CandidateConverter()
    graph = InMemoryGraphStore()
    updater = Updater(graph)
    occlusion_handler = OcclusionHandler(graph, registry)
    
    evaluator = Evaluator("ground_truth/dataset.json")
    
    print("Simulating processing of GT frames...")
    
    for gt_frame in evaluator.gt_data["frames"]:
        frame_id = gt_frame["frame_id"]
        scene_name = gt_frame["scene_type"]
        
        # Simulate slight imperfections in VLM perception
        raw_entities = []
        for ent in gt_frame["entities"]:
            if ent["present"]:
                # We skip 'apple' to simulate a missed detection (false negative)
                if ent["name"] == "apple":
                    continue
                # Add it correctly
                raw_entities.append({
                    "name": ent["name"],
                    "category": ent["category"],
                    "state": ent.get("state"),
                    "confidence": 0.9
                })
        
        # Simulate a false positive
        if frame_id == 2:
            raw_entities.append({
                "name": "ghost_object",
                "category": "unknown",
                "confidence": 0.6
            })
            
        # Simulate false split (predicts red_mug_2 instead of matching)
        if frame_id == 4:
            for e in raw_entities:
                if e["name"] == "red_mug":
                    # This simulates the raw name being different enough that the matcher creates a new ID
                    e["name"] = "red_mug_other"
        
        obs = SceneObservation(frame=frame_id, scene=scene_name, entities=raw_entities)
        
        # Pipeline Steps
        matched_obs = matcher.match(obs)
        candidates = converter.convert(matched_obs)
        updater.update(candidates, frame_id)
        occlusion_handler.handle_occlusions()
        
        # Gather active edges for the Evaluator
        active_state_edges = []
        active_location_edges = []
        
        for edge in graph.get_all_active_edges():
            if edge.relation == RelationType.HAS_STATE:
                active_state_edges.append(edge)
            elif edge.relation == RelationType.LOCATED_IN:
                active_location_edges.append(edge)
                
        evaluator.record_frame(frame_id, matched_obs.entities, active_state_edges, active_location_edges)
        
    print("Running Evaluation Metrics...")
    evaluator.evaluate()

if __name__ == "__main__":
    run_evaluation()
