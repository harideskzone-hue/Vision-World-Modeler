# evaluation/evaluator.py
import json
import logging
from typing import List, Dict, Any

from evaluation.metrics import (
    calculate_entity_precision_recall_f1,
    calculate_state_accuracy,
    calculate_relation_accuracy,
    calculate_false_merge_split_rates
)
from evaluation.consistency_metrics import (
    calculate_c1_id_stability,
    calculate_c2_temporal_consistency,
    calculate_c3_spatial_consistency
)

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Computes official HackTronix metrics against a Ground Truth dataset.
    """
    def __init__(self, gt_path: str):
        self.gt_path = gt_path
        with open(gt_path, 'r') as f:
            self.gt_data = json.load(f)
            
        self.pred_history = []
        self.gt_history = []

    def record_frame(self, frame_id: int, predicted_entities: List[Dict], active_state_edges, active_location_edges):
        """Records pipeline output for a frame."""
        # Find corresponding GT frame
        gt_frame = next((f for f in self.gt_data["frames"] if f["frame_id"] == frame_id), None)
        if not gt_frame:
            return
            
        self.pred_history.append({
            "frame_id": frame_id,
            "entities": predicted_entities,
            "state_edges": active_state_edges,
            "location_edges": active_location_edges
        })
        self.gt_history.append(gt_frame)

    def evaluate(self):
        """Computes and prints the final evaluation report."""
        if not self.pred_history:
            print("No data to evaluate.")
            return

        print("="*50)
        print("HACKTRONIX TRACK 2 - FINAL EVALUATION REPORT")
        print("="*50)

        # 1. Detection Metrics
        all_pred_names = []
        all_gt_names = []
        all_pred_states = {}
        all_gt_states = {}
        
        for pred, gt in zip(self.pred_history, self.gt_history):
            p_names = [e["name"].rsplit("_", 1)[0] if len(e["name"].rsplit("_", 1))>1 else e["name"] for e in pred["entities"]]
            g_names = [e["name"] for e in gt["entities"] if e["present"]]
            all_pred_names.extend(p_names)
            all_gt_names.extend(g_names)
            
            for g_ent in gt["entities"]:
                if g_ent.get("state"):
                    # Find matching pred
                    p_match = next((p for p in pred["entities"] if p["name"].startswith(g_ent["name"])), None)
                    all_gt_states[f"{gt['frame_id']}_{g_ent['name']}"] = g_ent["state"]
                    if p_match:
                        # Find state edge for this pred
                        p_state = next((edge.object for edge in pred["state_edges"] if edge.subject == p_match["name"]), None)
                        all_pred_states[f"{gt['frame_id']}_{g_ent['name']}"] = p_state
                        
        det_metrics = calculate_entity_precision_recall_f1(all_pred_names, all_gt_names)
        state_acc = calculate_state_accuracy(all_pred_states, all_gt_states)
        merge_split = calculate_false_merge_split_rates([e["name"] for p in self.pred_history for e in p["entities"]], all_gt_names)
        
        # 2. Consistency Metrics
        frame_entities_over_time = [[e["name"] for e in p["entities"]] for p in self.pred_history]
        state_edges_over_time = [p["state_edges"] for p in self.pred_history]
        loc_edges_over_time = [p["location_edges"] for p in self.pred_history]
        
        c1 = calculate_c1_id_stability(frame_entities_over_time)
        c2 = calculate_c2_temporal_consistency(state_edges_over_time)
        c3 = calculate_c3_spatial_consistency(loc_edges_over_time)

        # Print Report
        print(f"{'Metric':<25} | {'Score':<10}")
        print("-" * 40)
        print(f"{'Entity Precision':<25} | {det_metrics['precision']:.4f}")
        print(f"{'Entity Recall':<25} | {det_metrics['recall']:.4f}")
        print(f"{'Entity F1':<25} | {det_metrics['f1']:.4f}")
        print(f"{'State Accuracy':<25} | {state_acc:.4f}")
        print(f"{'ID Consistency (C1)':<25} | {c1:.4f}")
        print(f"{'Temporal Consist. (C2)':<25} | {c2:.4f}")
        print(f"{'Spatial Consist. (C3)':<25} | {c3:.4f}")
        print(f"{'False Merge Rate':<25} | {merge_split['false_merge_rate']:.4f}")
        print(f"{'False Split Rate':<25} | {merge_split['false_split_rate']:.4f}")
        print("="*50)
