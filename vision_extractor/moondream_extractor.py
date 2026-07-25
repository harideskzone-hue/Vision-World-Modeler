# vision_extractor/moondream_extractor.py
import numpy as np
import logging
from PIL import Image

from vision_extractor.base import VisionExtractor
from vision_extractor.prompt_templates import SCENE_EXTRACTION_PROMPT
from vision_extractor.scene_parser import SceneParser
from shared.models import SceneObservation
from shared.config import DEFAULT_CONFIG

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
            # Respect force_cpu compliance default, otherwise detect M1 (MPS) or CPU
            if DEFAULT_CONFIG.hardware.force_cpu:
                device = "cpu"
                dtype = torch.float32  # CPU inference on float32/float16
                logger.info("⚡ Moondream enforced to run on CPU (compliance default)")
            else:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                dtype = torch.float16
            
            logger.info(f"Loading Moondream model {model_id} on {device}...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                revision=revision,
                torch_dtype=dtype
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
