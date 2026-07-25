# vision_extractor/ollama_extractor.py
import numpy as np
import logging
import json
import base64
import cv2
import requests

from vision_extractor.base import VisionExtractor
from vision_extractor.prompt_templates import SCENE_EXTRACTION_PROMPT
from vision_extractor.scene_parser import SceneParser
from shared.models import SceneObservation

logger = logging.getLogger(__name__)

class OllamaExtractor(VisionExtractor):
    """
    Implementation of VisionExtractor using Ollama (moondream / llava GGUF with Metal GPU acceleration).
    Runs at ~0.3s per frame on Apple Silicon M1.
    """
    
    def __init__(self, model_name: str = "moondream", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self._parser = SceneParser()
        logger.info(f"OllamaExtractor initialized using model '{model_name}' at {self.endpoint}")

    def extract(self, image: np.ndarray, frame_id: int) -> SceneObservation:
        try:
            # Encode image to JPEG base64
            success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not success:
                logger.error(f"Failed to encode image to JPEG at frame {frame_id}")
                return SceneObservation(frame=frame_id, scene="unknown", entities=[])
            
            b64_image = base64.b64encode(buffer).decode("utf-8")
            
            payload = {
                "model": self.model_name,
                "prompt": SCENE_EXTRACTION_PROMPT,
                "images": [b64_image],
                "stream": False,
                "format": "json"
            }
            
            response = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=120)
            if response.status_code != 200:
                logger.error(f"Ollama API returned status {response.status_code}: {response.text}")
                return SceneObservation(frame=frame_id, scene="unknown", entities=[])
                
            result_json = response.json()
            raw_text = result_json.get("response", "")
            
            observation = self._parser.parse(raw_text, frame_id)
            if observation is None:
                return SceneObservation(frame=frame_id, scene="unknown", entities=[])
                
            return observation
            
        except Exception as e:
            logger.error(f"Error during Ollama inference at frame {frame_id}: {e}")
            return SceneObservation(frame=frame_id, scene="unknown", entities=[])
