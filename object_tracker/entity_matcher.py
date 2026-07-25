# object_tracker/entity_matcher.py
import math
from typing import Dict, List, Any, Optional
import numpy as np
import logging

from shared.models import SceneObservation
from object_tracker.entity_registry import EntityRegistry
from vision_extractor.clip_reidentifier import CLIPReidentifier

logger = logging.getLogger(__name__)

class EntityMatcher:
    """
    Matches raw parsed entities to stable world model IDs.
    Disambiguates identical objects using spatial heuristics if bboxes are present.
    """
    def __init__(self, registry: EntityRegistry, use_clip: bool = True, clip_threshold: float = 0.85, clip_reid: Optional[CLIPReidentifier] = None):
        self.registry = registry
        # Keep track of last known positions for spatial IoU tracking
        self._last_positions: Dict[str, List[float]] = {}
        self._last_bboxes: Dict[str, List[float]] = {}
        self._last_categories: Dict[str, str] = {}
        
        # CLIP re-identification
        self.use_clip = use_clip
        self.clip_threshold = clip_threshold
        if use_clip:
            if clip_reid:
                self.clip_reidentifier = clip_reid
                logger.info("✅ CLIP re-identification enabled (shared instance)")
            else:
                try:
                    self.clip_reidentifier = CLIPReidentifier(lazy_load=False)
                    logger.info("✅ CLIP re-identification enabled (new instance)")
                except Exception as e:
                    logger.warning(f"⚠️ CLIP unavailable, falling back to IoU tracking: {e}")
                    self.use_clip = False
                    self.clip_reidentifier = None
        else:
            self.clip_reidentifier = None
            
        self._embeddings: Dict[str, np.ndarray] = {}

    def match(self, observation: SceneObservation, current_frame: Optional[np.ndarray] = None) -> SceneObservation:
        """Assigns stable IDs to all entities in the observation."""
        disambiguated_entities = self._disambiguate(observation.entities)
        
        matched_entities = []
        for ent in disambiguated_entities:
            # Try CLIP-based re-identification first
            if self.use_clip and current_frame is not None and self.clip_reidentifier and "bbox" in ent:
                stable_id, match_details = self._find_match_with_clip(ent, current_frame)
            else:
                # Fall back to IoU-based matching
                stable_id, match_details = self._find_or_create_match(ent)
            
            # Create a copy and update name to stable ID
            matched_ent = ent.copy()
            matched_ent["raw_name"] = ent["name"]
            matched_ent["name"] = stable_id
            matched_ent["match_confidence"] = match_details["confidence"]
            matched_ent["match_reason"] = match_details["reason"]
            
            # Store position and bbox for future spatial tracking
            if "bbox" in matched_ent:
                self._last_positions[stable_id] = self._get_center(matched_ent["bbox"])
                self._last_bboxes[stable_id] = matched_ent["bbox"]
                self._last_categories[stable_id] = matched_ent.get("category", "unknown")
                
                # Extract and cache CLIP embedding
                if self.use_clip and current_frame is not None and self.clip_reidentifier:
                    embedding = self.clip_reidentifier.extract_embedding(current_frame, matched_ent["bbox"])
                    if embedding is not None:
                        self._embeddings[stable_id] = embedding
                
            matched_entities.append(matched_ent)

        return SceneObservation(
            frame=observation.frame,
            scene=observation.scene,
            entities=matched_entities
        )

    def _disambiguate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handles cases like [chair, chair, chair] -> [chair_left, chair_center, chair_right]."""
        # Group by name
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ent in entities:
            grouped.setdefault(ent["name"], []).append(ent)
            
        disambiguated = []
        for name, group in grouped.items():
            if len(group) == 1:
                disambiguated.append(group[0])
                continue
                
            # Multiple items with same name. Attempt spatial sort if bboxes exist.
            # Sort left to right based on center X
            group_with_pos = []
            group_without_pos = []
            
            for ent in group:
                if "bbox" in ent:
                    center_x = (ent["bbox"][0] + ent["bbox"][2]) / 2.0
                    group_with_pos.append((center_x, ent))
                else:
                    group_without_pos.append(ent)
                    
            group_with_pos.sort(key=lambda x: x[0])
            
            # If we have positions, label them left/center/right or by index
            if len(group_with_pos) == 2:
                labels = ["_left", "_right"]
            elif len(group_with_pos) == 3:
                labels = ["_left", "_center", "_right"]
            else:
                labels = [f"_{i+1}" for i in range(len(group_with_pos))]
                
            for i, (_, ent) in enumerate(group_with_pos):
                ent_copy = ent.copy()
                ent_copy["name"] = f"{name}{labels[i]}"
                disambiguated.append(ent_copy)
                
            # Handle those without positions
            for i, ent in enumerate(group_without_pos):
                ent_copy = ent.copy()
                ent_copy["name"] = f"{name}_unpos_{i+1}"
                disambiguated.append(ent_copy)
                
        return disambiguated

    def _find_match_with_clip(self, entity: Dict[str, Any], frame: np.ndarray) -> tuple[str, Dict[str, Any]]:
        """Find match using CLIP embeddings with similarity threshold."""
        base_name = entity["name"]
        category = entity.get("category", "unknown")
        
        # 1. Exact name match first (fastest)
        existing_id = self.registry.get_id(base_name)
        if existing_id:
            if self.registry.is_archived(existing_id):
                self.registry.reactivate(existing_id)
            return existing_id, {
                "confidence": 0.95,
                "reason": {"name": 0.95, "clip": 0.0}
            }
            
        # 2. CLIP embedding match
        best_clip_score = 0.0
        best_clip_id = None
        
        current_embedding = self.clip_reidentifier.extract_embedding(frame, entity["bbox"])
        
        if current_embedding is not None:
            # Compare against all archived embeddings
            for archived_id, archived_embedding in self._embeddings.items():
                if self._last_categories.get(archived_id) == category:
                    similarity = self.clip_reidentifier.compute_similarity(current_embedding, archived_embedding)
                    
                    if similarity > best_clip_score and similarity > self.clip_threshold:
                        best_clip_score = similarity
                        best_clip_id = archived_id
                        
        # If we found a strong CLIP match
        if best_clip_id and best_clip_score > self.clip_threshold:
            if self.registry.is_archived(best_clip_id):
                self.registry.reactivate(best_clip_id)
            
            self.registry.register_id(base_name, best_clip_id)
            
            return best_clip_id, {
                "confidence": 0.92 + (best_clip_score * 0.05),
                "reason": {
                    "name": 0.0, "clip": best_clip_score, "category": 1.0, "clip_reidentifier": True
                }
            }
            
        # 3. Fall back to IoU-based tracking
        return self._find_or_create_match(entity)

    def _find_or_create_match(self, entity: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        base_name = entity["name"]
        category = entity.get("category", "unknown")
        
        # 1. Exact Name Match (Primary)
        existing_id = self.registry.get_id(base_name)
        if existing_id:
            # Reactivate if archived
            if self.registry.is_archived(existing_id):
                self.registry.reactivate(existing_id)
                
            return existing_id, {
                "confidence": 0.95,
                "reason": {"name": 0.95, "location": 0.0, "temporal": 0.0, "category": 0.0}
            }

        # 2. Advanced Spatial Match (IoU Tracking Fallback)
        if "bbox" in entity:
            best_iou = 0.0
            best_match_id = None
            
            for archived_id, last_bbox in self._last_bboxes.items():
                # Only compare if categories match (e.g., don't map a 'chair' to a 'desk' just because they overlap)
                if self._last_categories.get(archived_id) == category:
                    iou = self._calculate_iou(entity["bbox"], last_bbox)
                    if iou > best_iou and iou > 0.4: # 40% overlap threshold
                        best_iou = iou
                        best_match_id = archived_id
                        
            if best_match_id:
                if self.registry.is_archived(best_match_id):
                    self.registry.reactivate(best_match_id)
                # Link the new name alias to the matched stable ID in the registry
                self.registry.register_id(base_name, best_match_id)
                return best_match_id, {
                    "confidence": 0.85 + (best_iou * 0.1), 
                    "reason": {"name": 0.0, "location": best_iou, "temporal": 0.8, "category": 1.0, "iou_tracker": True}
                }

        # Fallback: create new ID
        new_id = self.registry.generate_id(base_name)
        return new_id, {
            "confidence": 1.0, 
            "reason": {"name": 1.0, "location": 0.0, "temporal": 0.0, "category": 0.0, "new_entity": True}
        }

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculates Intersection over Union (IoU) for two bounding boxes [x1, y1, x2, y2]."""
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou

    def _get_center(self, bbox: List[float]) -> List[float]:
        return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
