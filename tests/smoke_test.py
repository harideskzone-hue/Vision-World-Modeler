import sys
import os

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model.graph_store import InMemoryGraphStore
from updater.updater import Updater
from shared.models import CandidateFact
from shared.enums import RelationType, ExtractionMethod

def run_smoke_test():
    print("Initializing Graph Store...")
    graph = InMemoryGraphStore()
    
    print("Initializing Updater...")
    updater = Updater(graph)
    
    print("Creating Candidate Facts...")
    candidates = [
        CandidateFact(
            subject="camera",
            relation=RelationType.LOCATED_IN,
            object="kitchen",
            confidence=1.0,
            source_frame_id=0,
            extraction_method=ExtractionMethod.RULE
        ),
        CandidateFact(
            subject="kitchen",
            relation=RelationType.CONTAINS,
            object="wooden_table",
            confidence=0.85,
            source_frame_id=0,
            extraction_method=ExtractionMethod.VLM
        ),
        CandidateFact(
            subject="wooden_table",
            relation=RelationType.HAS_STATE,
            object="clean",
            confidence=0.85,
            source_frame_id=0,
            extraction_method=ExtractionMethod.VLM
        )
    ]
    
    print("Running Updater (Frame 0)...")
    report = updater.update(candidates, frame_id=0)
    print(f"Report: expanded={report.expanded}, corroborated={report.corroborated}, revised={report.revised}")
    
    stats = graph.get_stats()
    print(f"Graph Stats: nodes={stats.total_nodes}, active edges={stats.active_edges}")
    assert stats.active_edges == 3
    assert stats.total_nodes == 3 # camera, kitchen, wooden_table (states are not nodes)
    
    print("Creating Contradicting Facts (Frame 1)...")
    candidates_frame_1 = [
        CandidateFact(
            subject="wooden_table",
            relation=RelationType.HAS_STATE,
            object="dirty", # should override clean with R1
            confidence=0.80,
            source_frame_id=1,
            extraction_method=ExtractionMethod.VLM
        )
    ]
    
    report_1 = updater.update(candidates_frame_1, frame_id=1)
    print(f"Report 1: expanded={report_1.expanded}, corroborated={report_1.corroborated}, revised={report_1.revised}")
    
    stats = graph.get_stats()
    print(f"Graph Stats: nodes={stats.total_nodes}, active edges={stats.active_edges}, superseded edges={stats.superseded_edges}")
    assert stats.active_edges == 3
    assert stats.superseded_edges == 1
    
    print("Smoke test passed successfully!")

if __name__ == "__main__":
    run_smoke_test()
