import numpy as np
from typing import Optional, List, Dict
import logging
from shared.models import SceneObservation
from shared.config import DEFAULT_CONFIG
from vision_extractor.clip_reidentifier import CLIPReidentifier

logger = logging.getLogger(__name__)

class SemanticFusionModule:
    """Enhanced VLM + YOLO fusion with semantic similarity and spatial constraints."""
    
    def __init__(self, clip_reid: Optional[CLIPReidentifier] = None):
        self.max_fusion_distance = 50  # pixels
        self.clip_reid = clip_reid
        self.text_emb_cache = {}
    
    def compute_semantic_similarity(self, vlm_name: str, yolo_name: str) -> float:
        """
        Compute semantic similarity between VLM and YOLO category names.
        
        Returns:
            float in [0, 1]
        """
        vlm_name_lower = vlm_name.lower().replace("_", " ")
        yolo_name_lower = yolo_name.lower().replace("_", " ")
        
        # 1. Exact match (fastest)
        if vlm_name_lower == yolo_name_lower:
            return 1.0
            
        # 2. CLIP Zero-Shot Neural Similarity
        if self.clip_reid is not None:
            if vlm_name_lower not in self.text_emb_cache:
                self.text_emb_cache[vlm_name_lower] = self.clip_reid.extract_text_embedding(vlm_name_lower)
            if yolo_name_lower not in self.text_emb_cache:
                self.text_emb_cache[yolo_name_lower] = self.clip_reid.extract_text_embedding(yolo_name_lower)
                
            emb1 = self.text_emb_cache[vlm_name_lower]
            emb2 = self.text_emb_cache[yolo_name_lower]
            if emb1 is not None and emb2 is not None:
                sim = self.clip_reid.compute_similarity(emb1, emb2)
                # Boost if highly similar (synonyms) to exceed the threshold
                if sim > 0.88:
                    return 0.95
                # Do NOT return intermediate values like 0.70 for sim > 0.80
                # because CLIP text embeddings for unrelated nouns can be as high as 0.83!
                
        # 3. Levenshtein distance for typo tolerance (fallback)
        distance = self._levenshtein(vlm_name_lower, yolo_name_lower)
        max_len = max(len(vlm_name_lower), len(yolo_name_lower))
        if max_len == 0:
            return 0.0
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _get_center(self, bbox: List[float]) -> List[float]:
        return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
        
    def validate_spatial_alignment(self, vlm_bbox: Optional[List[float]],
                                   yolo_bbox: List[float]) -> float:
        """
        Validate spatial alignment between VLM-implied position and YOLO bbox.
        
        Returns:
            float in [0, 1] confidence in alignment
        """
        if vlm_bbox is None:
            return 1.0  # VLM didn't provide bbox, can't contradict
        
        # Calculate center distance
        vlm_center = self._get_center(vlm_bbox)
        yolo_center = self._get_center(yolo_bbox)
        
        distance = np.sqrt(
            (vlm_center[0] - yolo_center[0])**2 +
            (vlm_center[1] - yolo_center[1])**2
        )
        
        # Confidence decreases with distance
        if distance > self.max_fusion_distance:
            return 0.0
        
        return 1.0 - (distance / self.max_fusion_distance)
    
    def fuse(self, vlm_obs: SceneObservation,
                             yolo_detections: List[Dict]) -> SceneObservation:
        """
        Fuse VLM and YOLO with semantic and spatial constraints.
        Matches the interface of the original fusion.py
        """
        fused_entities = []
        yolo_used = set()
        
        for vlm_ent in vlm_obs.entities:
            vlm_name = vlm_ent["name"]
            vlm_cat = vlm_ent.get("category", vlm_name)
            vlm_conf = vlm_ent["confidence"]
            vlm_bbox = vlm_ent.get("bbox")
            
            # Find best YOLO match using semantic similarity + spatial alignment
            best_yolo = None
            best_yolo_idx = -1
            best_match_score = 0.0
            
            for i, yolo_det in enumerate(yolo_detections):
                if i in yolo_used:
                    continue
                
                yolo_name = yolo_det["name"]
                yolo_conf = yolo_det["confidence"]
                
                # Semantic similarity (check both name and category against YOLO)
                semantic_sim_name = self.compute_semantic_similarity(vlm_name, yolo_name)
                semantic_sim_cat = self.compute_semantic_similarity(vlm_cat, yolo_name)
                semantic_sim = max(semantic_sim_name, semantic_sim_cat)
                
                if semantic_sim < 0.3: # Skip bad semantic matches entirely
                    continue
                    
                # Spatial alignment
                spatial_align = self.validate_spatial_alignment(vlm_bbox, yolo_det["bbox"])
                
                # Combined score (weighted)
                match_score = (semantic_sim * 0.7) + (spatial_align * 0.3)
                
                if match_score > best_match_score and match_score > 0.5:
                    best_match_score = match_score
                    best_yolo = yolo_det
                    best_yolo_idx = i
            
            if best_yolo and best_match_score > 0.6:
                # High-confidence fusion
                fused_conf = min(
                    DEFAULT_CONFIG.updater.max_confidence,
                    (vlm_conf + best_yolo["confidence"]) / 2.0 * best_match_score + 0.10
                )
                
                fused_ent = vlm_ent.copy()
                fused_ent["confidence"] = fused_conf
                fused_ent["bbox"] = best_yolo["bbox"]
                fused_ent["fusion_score"] = best_match_score
                # Overwrite VLM hallucinated category with reliable YOLO category
                fused_ent["category"] = best_yolo.get("category", best_yolo["name"])
                
                fused_entities.append(fused_ent)
                yolo_used.add(best_yolo_idx)
            else:
                # Low confidence fusion, lower VLM confidence
                fused_conf = max(DEFAULT_CONFIG.updater.min_confidence, vlm_conf - 0.20)
                fused_ent = vlm_ent.copy()
                fused_ent["confidence"] = fused_conf
                # Orphans don't have a YOLO match to correct their category, so rely on VLM name
                fused_ent["category"] = vlm_name
                fused_entities.append(fused_ent)
        
        # Orphan YOLO detections (VLM missed)
        for i, yolo_det in enumerate(yolo_detections):
            if i not in yolo_used:
                fused_conf = max(DEFAULT_CONFIG.updater.min_confidence, yolo_det["confidence"] - 0.20)
                fused_entities.append({
                    "name": yolo_det["name"],
                    "category": yolo_det["category"],
                    "state": None,
                    "confidence": fused_conf,
                    "bbox": yolo_det["bbox"],
                    "source": "yolo_only"
                })
        
        return SceneObservation(
            frame=vlm_obs.frame,
            scene=vlm_obs.scene,
            entities=fused_entities
        )
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return SemanticFusionModule._levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
