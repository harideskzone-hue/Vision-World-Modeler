# tests/test_entity_matcher_and_metrics.py
import pytest
import numpy as np
from object_tracker.entity_registry import EntityRegistry
from object_tracker.entity_matcher import EntityMatcher
from shared.models import SceneObservation
from evaluation.metrics import (
    calculate_entity_precision_recall_f1,
    calculate_state_accuracy,
    calculate_relation_accuracy,
    calculate_false_merge_split_rates
)

def test_entity_matcher_iou_tracking():
    """Verify EntityMatcher assigns stable IDs and matches overlapping bounding boxes via IoU."""
    registry = EntityRegistry()
    matcher = EntityMatcher(registry, use_clip=False)  # Run without model weights for speed/determinism
    
    # Frame 1: Detect a sofa and a chair with explicit bboxes
    obs1 = SceneObservation(frame=1, scene="living_room", entities=[
        {"name": "sofa", "category": "furniture", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.90},
        {"name": "chair", "category": "furniture", "bbox": [0.6, 0.6, 0.9, 0.9], "confidence": 0.85}
    ])
    res1 = matcher.match(obs1)
    
    sofa_id_f1 = res1.entities[0]["name"]
    chair_id_f1 = res1.entities[1]["name"]
    assert sofa_id_f1 != chair_id_f1
    assert res1.entities[0]["match_confidence"] > 0
    
    # Frame 2: Slight motion in bounding box (sofa moved from 0.1 to 0.12)
    obs2 = SceneObservation(frame=2, scene="living_room", entities=[
        {"name": "sofa", "category": "furniture", "bbox": [0.12, 0.12, 0.52, 0.52], "confidence": 0.92}
    ])
    res2 = matcher.match(obs2)
    sofa_id_f2 = res2.entities[0]["name"]
    
    # Verify spatial continuity preserves exact ID
    assert sofa_id_f2 == sofa_id_f1
    assert "IoU" in res2.entities[0]["match_reason"] or "Spatial" in res2.entities[0]["match_reason"] or res2.entities[0]["match_confidence"] >= 0.5

def test_evaluation_metrics_precision_recall():
    """Verify mathematical correctness of entity F1 scoring."""
    pred = ["sofa_1", "tv_1", "table_1"]
    gt = ["sofa_1", "tv_1", "chair_1"]
    
    metrics = calculate_entity_precision_recall_f1(pred, gt)
    assert abs(metrics["precision"] - (2/3)) < 0.001
    assert abs(metrics["recall"] - (2/3)) < 0.001
    assert abs(metrics["f1"] - (2/3)) < 0.001

def test_state_and_relation_accuracy():
    """Verify state agreement accuracy calculation and false merge/split rates."""
    pred_states = {"door_1": "open", "window_1": "closed"}
    gt_states = {"door_1": "open", "window_1": "open"}
    
    acc = calculate_state_accuracy(pred_states, gt_states)
    assert abs(acc - 0.5) < 0.001

    pred_rels = [("sofa_1", "located_in", "living_room")]
    gt_rels = [("sofa_1", "located_in", "living_room")]
    assert calculate_relation_accuracy(pred_rels, gt_rels) == 1.0

    # Using 6-hex hash naming format expected by false merge rate heuristics
    rates = calculate_false_merge_split_rates(["chair_a1b2c3", "chair_d4e5f6"], ["chair", "chair"])
    assert rates["false_merge_rate"] == 0.0
    assert rates["false_split_rate"] == 0.0
