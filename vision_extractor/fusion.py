# vision_extractor/fusion.py
from typing import List, Dict, Any
from shared.models import SceneObservation
from shared.config import DEFAULT_CONFIG

class FusionModule:
    """
    Deterministically fuses VLM semantics and YOLO geometry.
    - Agreement boosts confidence.
    - Disagreement lowers confidence.
    """
    def fuse(self, vlm_obs: SceneObservation, yolo_detections: List[Dict[str, Any]]) -> SceneObservation:
        # Create a new observation to avoid mutating the original
        fused_entities = []
        
        # We process VLM entities and try to map them to YOLO detections
        yolo_used = set()
        
        for vlm_ent in vlm_obs.entities:
            vlm_name = vlm_ent["name"]
            vlm_cat = vlm_ent["category"]
            vlm_conf = vlm_ent["confidence"]
            
            # Find best YOLO match based on category or name string overlap
            best_yolo = None
            best_yolo_idx = -1
            best_score = 0.0
            
            for i, yolo_det in enumerate(yolo_detections):
                if i in yolo_used:
                    continue
                    
                yolo_name = yolo_det["name"]
                
                # Simple matching logic
                if vlm_name == yolo_name or vlm_cat == yolo_name:
                    best_yolo = yolo_det
                    best_yolo_idx = i
                    break
                    
                # Substring match
                if yolo_name in vlm_name or vlm_name in yolo_name:
                    best_yolo = yolo_det
                    best_yolo_idx = i
                    break
                    
            if best_yolo:
                # Agreement -> boost confidence
                fused_conf = min(
                    DEFAULT_CONFIG.updater.max_confidence,
                    (vlm_conf + best_yolo["confidence"]) / 2.0 + 0.10
                )
                
                fused_ent = vlm_ent.copy()
                fused_ent["confidence"] = fused_conf
                fused_ent["bbox"] = best_yolo["bbox"] # Add geometry
                
                fused_entities.append(fused_ent)
                yolo_used.add(best_yolo_idx)
            else:
                # Disagreement (VLM saw it, YOLO didn't) -> lower confidence
                fused_conf = max(
                    DEFAULT_CONFIG.updater.min_confidence,
                    vlm_conf - 0.15
                )
                fused_ent = vlm_ent.copy()
                fused_ent["confidence"] = fused_conf
                fused_entities.append(fused_ent)
                
        # Add YOLO detections that VLM missed, but with lower confidence and no state
        for i, yolo_det in enumerate(yolo_detections):
            if i not in yolo_used:
                fused_conf = max(
                    DEFAULT_CONFIG.updater.min_confidence,
                    yolo_det["confidence"] - 0.15
                )
                fused_entities.append({
                    "name": yolo_det["name"],
                    "category": yolo_det["category"],
                    "state": None,
                    "confidence": fused_conf,
                    "bbox": yolo_det["bbox"]
                })
                
        return SceneObservation(
            frame=vlm_obs.frame,
            scene=vlm_obs.scene,
            entities=fused_entities
        )
