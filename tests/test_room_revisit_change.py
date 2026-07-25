# tests/test_room_revisit_change.py
"""
HackTronix Track 2 Evaluation Criterion C3 Test Suite:
Verifies that the architecture correctly updates world state when the exact same room
is revisited later in time after an entity's physical state has changed in the interim.
"""

import pytest
from tests.test_corrections import MockPipeline
from shared.enums import NodeStatus, EdgeStatus, RelationType

def test_room_revisit_with_state_change():
    pipeline = MockPipeline()
    
    # Phase 1: Initial visit to 'Library' (Frame 1)
    # Lamp is observed as OFF, Table is CLEAN
    obs_1 = [
        {"name": "reading_lamp", "category": "appliance", "state": "off", "confidence": 0.90},
        {"name": "study_table", "category": "furniture", "state": "clean", "confidence": 0.95}
    ]
    pipeline.process(1, "Library", obs_1)
    
    lamp_id = pipeline.registry.get_id("reading_lamp")
    lamp_node = pipeline.graph.get_node(lamp_id)
    assert lamp_node is not None, f"Lamp node with ID {lamp_id} must be instantiated on first visit"
    assert lamp_node.status == NodeStatus.ACTIVE
    
    # Phase 2: Leave Library, transition to 'Classroom' (Frame 5)
    obs_2 = [
        {"name": "projector", "category": "electronics", "state": "on", "confidence": 0.88}
    ]
    pipeline.process(5, "Classroom", obs_2)
    
    # Phase 3: Revisit 'Library' at Frame 15 with interim state changes!
    # Someone turned the reading lamp ON while the camera was away in the classroom.
    obs_3 = [
        {"name": "reading_lamp", "category": "appliance", "state": "on", "confidence": 0.92},
        {"name": "study_table", "category": "furniture", "state": "clean", "confidence": 0.95}
    ]
    pipeline.process(15, "Library", obs_3)
    
    # Verify C1 ID Consistency: No duplicate reading_lamp IDs should exist!
    lamp_id_after = pipeline.registry.get_id("reading_lamp")
    assert lamp_id == lamp_id_after, "Must maintain 100% C1 stable ID consistency upon room re-entry"
    assert pipeline.graph.get_node(lamp_id).status == NodeStatus.ACTIVE
    
    # Verify C3 State Reconciliation: Old 'off' state fact must be superseded by new 'on' fact
    state_edges = [
        e for e in pipeline.graph._edges.values()
        if e.subject == lamp_id and e.relation == RelationType.HAS_STATE
    ]
    
    active_states = [e for e in state_edges if e.status == EdgeStatus.ACTIVE]
    superseded_states = [e for e in state_edges if e.status == EdgeStatus.SUPERSEDED]
    
    assert len(active_states) == 1, "Must enforce architectural single-occupancy state invariant"
    assert active_states[0].object == "on", "New state 'on' must override previous belief"
    assert len(superseded_states) >= 1, "Previous state 'off' must be gracefully archived as SUPERSEDED"
    assert superseded_states[0].object == "off"
    assert superseded_states[0].superseded_by == active_states[0].id

    print("\n✅ C3 Room Revisit & State Reconciliation Verification Passed!")

if __name__ == "__main__":
    test_room_revisit_with_state_change()
