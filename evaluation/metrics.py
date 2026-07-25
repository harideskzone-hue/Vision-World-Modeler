# evaluation/metrics.py
from typing import List, Dict, Any, Tuple

def calculate_entity_precision_recall_f1(predicted_entities: List[str], gt_entities: List[str]) -> Dict[str, float]:
    pred_set = set(predicted_entities)
    gt_set = set(gt_entities)
    
    true_positives = len(pred_set.intersection(gt_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_state_accuracy(predicted_states: Dict[str, str], gt_states: Dict[str, str]) -> float:
    # Only evaluate entities present in GT that have state
    correct = 0
    total = 0
    for entity, gt_state in gt_states.items():
        if gt_state:
            total += 1
            if predicted_states.get(entity) == gt_state:
                correct += 1
    
    return correct / total if total > 0 else 1.0

def calculate_relation_accuracy(predicted_relations: List[Tuple[str, str, str]], gt_relations: List[Tuple[str, str, str]]) -> float:
    pred_set = set(predicted_relations)
    gt_set = set(gt_relations)
    if not gt_set:
        return 1.0
        
    correct = len(pred_set.intersection(gt_set))
    return correct / len(gt_set)

def calculate_false_merge_split_rates(pred_ids: List[str], gt_names: List[str]) -> Dict[str, float]:
    """
    Approximation for false merges and splits based on naming heuristics (since we don't have dense pixel tracking GT).
    If GT has multiple items (e.g. chair, chair, chair) and Pred has fewer, it's a False Merge.
    If Pred has more IDs mapped to the same base name, it's a False Split.
    """
    gt_counts = {}
    for n in gt_names:
        gt_counts[n] = gt_counts.get(n, 0) + 1
        
    pred_base_counts = {}
    for pid in pred_ids:
        # Assuming ID format: base_name_hash or base_name_left
        base = pid.rsplit("_", 1)[0]
        if len(pid.rsplit("_", 1)) > 1 and len(pid.rsplit("_", 1)[1]) == 6: # likely a hash
             # Check if it looks like a hash
             try:
                 int(pid.rsplit("_", 1)[1], 16)
             except ValueError:
                 base = pid # Not a hash
        elif pid.endswith("_left") or pid.endswith("_center") or pid.endswith("_right"):
             pass # base is correct
        else:
             base = pid
             
        pred_base_counts[base] = pred_base_counts.get(base, 0) + 1
        
    false_merges = 0
    false_splits = 0
    total_gt = max(1, len(gt_names))
    
    for base, gt_count in gt_counts.items():
        pred_count = pred_base_counts.get(base, 0)
        if pred_count < gt_count:
            false_merges += (gt_count - pred_count)
        elif pred_count > gt_count:
            false_splits += (pred_count - gt_count)
            
    return {
        "false_merge_rate": false_merges / total_gt,
        "false_split_rate": false_splits / total_gt
    }
