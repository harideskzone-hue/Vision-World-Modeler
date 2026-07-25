# evaluation/consistency_metrics.py
from typing import List, Dict, Tuple, Any
from shared.enums import EdgeStatus

def calculate_c1_id_stability(frame_entities_over_time: List[List[str]]) -> float:
    """
    C1: Stable ID consistency.
    Measures if an object maintains its ID across consecutive frames.
    Penalty for ID switches.
    """
    if len(frame_entities_over_time) < 2:
        return 1.0
        
    total_transitions = 0
    consistent_transitions = 0
    
    for i in range(len(frame_entities_over_time) - 1):
        prev = set(frame_entities_over_time[i])
        curr = set(frame_entities_over_time[i+1])
        
        # We only care about items that theoretically exist in both frames
        # For simplicity in this approximation, we just check intersection over union 
        # of base names vs stable IDs
        
        prev_bases = {p.rsplit('_', 1)[0] if len(p.rsplit('_', 1))>1 else p for p in prev}
        curr_bases = {c.rsplit('_', 1)[0] if len(c.rsplit('_', 1))>1 else c for c in curr}
        
        common_bases = prev_bases.intersection(curr_bases)
        
        for base in common_bases:
            total_transitions += 1
            # Check if the exact stable ID was maintained
            prev_id = next((p for p in prev if p.startswith(base)), None)
            curr_id = next((c for c in curr if c.startswith(base)), None)
            
            if prev_id and curr_id and prev_id == curr_id:
                consistent_transitions += 1
                
    return consistent_transitions / total_transitions if total_transitions > 0 else 1.0


def calculate_c2_temporal_consistency(state_edges_over_time: List[Dict[str, Any]]) -> float:
    """
    C2: Temporal belief consistency.
    Ensures that an object does not have contradictory states simultaneously.
    Our World Model uses superseding, so this should theoretically always be 1.0.
    """
    # state_edges_over_time is a list of active state edges per frame
    # We check if any subject has multiple active HAS_STATE edges pointing to conflicting values
    from world_model.schema import are_states_conflicting
    
    inconsistencies = 0
    total_checks = 0
    
    for frame_edges in state_edges_over_time:
        # Group by subject
        subject_states = {}
        for edge in frame_edges:
            subject_states.setdefault(edge.subject, []).append(edge.object)
            
        for subj, states in subject_states.items():
            total_checks += 1
            # Check pairwise conflicts
            conflict_found = False
            for i in range(len(states)):
                for j in range(i+1, len(states)):
                    if are_states_conflicting(states[i], states[j]):
                        conflict_found = True
                        break
                if conflict_found: break
            
            if conflict_found:
                inconsistencies += 1
                
    return 1.0 - (inconsistencies / total_checks) if total_checks > 0 else 1.0

def calculate_c3_spatial_consistency(location_edges_over_time: List[Dict[str, Any]]) -> float:
    """
    C3: Spatial consistency.
    Checks single-occupancy constraints (e.g. object can only be in one room at a time).
    """
    inconsistencies = 0
    total_checks = 0
    
    for frame_edges in location_edges_over_time:
        subject_locations = {}
        for edge in frame_edges:
            subject_locations.setdefault(edge.subject, []).append(edge.object)
            
        for subj, locs in subject_locations.items():
            total_checks += 1
            if len(set(locs)) > 1: # More than one distinct location actively believed!
                inconsistencies += 1
                
    return 1.0 - (inconsistencies / total_checks) if total_checks > 0 else 1.0
