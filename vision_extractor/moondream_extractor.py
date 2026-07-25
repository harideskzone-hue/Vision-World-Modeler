# vision_extractor/moondream_extractor.py
import numpy as np
import logging
from PIL import Image

from vision_extractor.base import VisionExtractor
from vision_extractor.prompt_templates import SCENE_EXTRACTION_PROMPT
from vision_extractor.scene_parser import SceneParser
from shared.models import SceneObservation

logger = logging.getLogger(__name__)

class MoondreamExtractor(VisionExtractor):
    """
    Implementation of VisionExtractor using Moondream2.
    """
    
    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: str = "2024-08-26"):
        self._parser = SceneParser()
        
        logger.info(f"Loading Moondream model {model_id}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("transformers or torch library not found. MoondreamExtractor requires both.")
            self.model = None
            self.tokenizer = None
            return

        try:
            # Detect M1 (MPS) or default to CPU, but force float16 to prevent memory swapping on 8GB Mac
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Loading Moondream model {model_id} on {device} (float16)...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                revision=revision,
                torch_dtype=torch.float16
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
            logger.info("Moondream model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Moondream model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            
    def extract(self, image: np.ndarray, frame_id: int) -> SceneObservation:
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded, returning empty observation.")
            return SceneObservation(frame=frame_id, scene="unknown", entities=[])
            
        try:
            pil_image = Image.fromarray(image)
            enc_image = self.model.encode_image(pil_image)
            
            raw_text = self.model.answer_question(enc_image, SCENE_EXTRACTION_PROMPT, self.tokenizer, max_new_tokens=64)
            
            observation = self._parser.parse(raw_text, frame_id)
            if observation is None:
                return SceneObservation(frame=frame_id, scene="unknown", entities=[])
                
            return observation
            
        except Exception as e:
            logger.error(f"Error during Moondream inference at frame {frame_id}: {e}", exc_info=True)
            return SceneObservation(frame=frame_id, scene="unknown", entities=[])
