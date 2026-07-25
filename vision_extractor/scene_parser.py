# vision_extractor/scene_parser.py
import json
import re
from typing import Dict, Any, Optional
import logging

from shared.models import SceneObservation
from shared.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class SceneParser:
    """
    Parses VLM text output into a structured SceneObservation using a 4-stage fallback.
    Includes schema validation to reject malformed data.
    """

    def parse(self, text: str, frame_id: int) -> Optional[SceneObservation]:
        """4-Stage parsing fallback."""
        if not text:
            return None
            
        parsed_dict = None
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed_dict = json.loads(text)
        except json.JSONDecodeError:
            # Strategy 2: Extract JSON block
            parsed_dict = self._extract_json_block(text)
            
            # Strategy 3: Regex recovery
            if parsed_dict is None:
                parsed_dict = self._regex_recovery(text)

        # Strategy 4: Return None
        if parsed_dict is None:
            logger.warning(f"Failed to parse VLM output at frame {frame_id}: {text[:100]}")
            return None
            
        return self._validate_and_build(parsed_dict, frame_id)

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Finds JSON within markdown fences or curly braces."""
        # Try markdown fences
        match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Try finding first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
                
        return None

    def _regex_recovery(self, text: str) -> Optional[Dict[str, Any]]:
        """Aggressive regex to salvage entity lists from completely broken JSON."""
        scene = "unknown"
        scene_match = re.search(r'"scene"\s*:\s*"([^"]+)"', text)
        if scene_match:
            scene = scene_match.group(1)
            
        entities = []
        # Look for entity blocks
        entity_blocks = re.findall(r'{[^{}]+name[^{}]+}', text)
        for block in entity_blocks:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            if not name_match:
                continue
                
            cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', block)
            state_match = re.search(r'"state"\s*:\s*("([^"]+)"|null)', block)
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', block)
            
            if name_match:
                state_val = None
                if state_match:
                    val = state_match.group(1)
                    if val != "null":
                        state_val = state_match.group(2)
                        
                entities.append({
                    "name": name_match.group(1),
                    "category": cat_match.group(1) if cat_match else "unknown",
                    "state": state_val,
                    "confidence": float(conf_match.group(1)) if conf_match else DEFAULT_CONFIG.vision_extractor.base_vlm_confidence
                })
                
        if entities or scene != "unknown":
            return {"scene": scene, "entities": entities}
            
        return None

    def _validate_and_build(self, data: Dict[str, Any], frame_id: int) -> Optional[SceneObservation]:
        """Validates schema and builds SceneObservation."""
        if not isinstance(data, dict):
            return None
            
        scene = data.get("scene", "unknown")
        if not isinstance(scene, str):
            scene = "unknown"
            
        raw_entities = data.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []
            
        validated_entities = []
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
                
            name = ent.get("name")
            if not name or not isinstance(name, str):
                continue
                
            # Clean up name (lowercase, replace spaces)
            name = name.lower().replace(" ", "_")
            
            # Confidence bounds
            conf = ent.get("confidence", DEFAULT_CONFIG.vision_extractor.base_vlm_confidence)
            try:
                conf = float(conf)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = DEFAULT_CONFIG.vision_extractor.base_vlm_confidence
                
            state = ent.get("state")
            if state is not None and not isinstance(state, str):
                state = None
                
            category = ent.get("category", "unknown")
            if not isinstance(category, str):
                category = "unknown"
                
            validated_entities.append({
                "name": name,
                "category": category,
                "state": state,
                "confidence": conf
            })
            
        return SceneObservation(
            frame=frame_id,
            scene=scene,
            entities=validated_entities
        )
