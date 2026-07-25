# tests/test_query_interface.py
import pytest
from world_model.graph_store import InMemoryGraphStore
from object_tracker.entity_registry import EntityRegistry
from query.interface import QueryInterface
from tests.test_corrections import MockPipeline
from shared.enums import RelationType, NodeStatus

def test_query_interface_empty_graph():
    """Verify QueryInterface behaves gracefully on empty graphs without crashing."""
    graph = InMemoryGraphStore()
    registry = EntityRegistry()
    query = QueryInterface(graph, registry)
    
    # Query non-existent scene
    assert query.get_objects_in_scene("kitchen") == []
    
    # Explain non-existent object
    exp = query.explain("sofa")
    assert "not found" in exp.lower()
    
    # Check occluded objects
    assert query.get_occluded_objects(current_frame=1) == []

def test_query_interface_scene_and_history():
    """Verify scene queries, explanation reports, and occlusion calculations."""
    pipeline = MockPipeline()
    query = QueryInterface(pipeline.graph, pipeline.registry)
    
    # Populate graph via pipeline process
    pipeline.process(10, "living_room", [
        {"name": "sofa_1", "category": "furniture", "state": "clean", "confidence": 0.90}
    ])
    pipeline.process(15, "kitchen", [
        {"name": "coffee_maker_1", "category": "appliance", "confidence": 0.95}
    ])
    
    sofa_id = pipeline.registry.get_id("sofa_1") or "sofa_1"
    coffee_maker_id = pipeline.registry.get_id("coffee_maker_1") or "coffee_maker_1"
    
    # Verify location query for living_room (using case or resolved ID)
    living_room_objs = query.get_objects_in_scene("living_room")
    assert sofa_id in living_room_objs or any("sofa" in o for o in living_room_objs)
    
    kitchen_objs = query.get_objects_in_scene("kitchen")
    assert coffee_maker_id in kitchen_objs or any("coffee_maker" in o for o in kitchen_objs)
    
    # Verify explanation generation
    explanation = query.explain(sofa_id)
    assert "Location:" in explanation
    assert "living_room" in explanation
    assert "State:" in explanation
    assert "clean" in explanation

    # Verify occlusion list at frame 25 (sofa and coffee_maker not observed since 10 and 15)
    occluded = query.get_occluded_objects(current_frame=25)
    assert sofa_id in occluded or coffee_maker_id in occluded
