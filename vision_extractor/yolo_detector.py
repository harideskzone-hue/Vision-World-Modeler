# vision_extractor/yolo_detector.py
import numpy as np
from typing import List, Dict, Any
import logging

from shared.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class YOLODetector:
    """
    Lightweight object detection for geometric priors (bounding boxes, base categories).
    Does not infer complex semantic states (like "open" or "clean").
    """
    def __init__(self, model_name: str = "yolo26n.pt"):
        self.model_name = model_name
        self.confidence_threshold = DEFAULT_CONFIG.vision_extractor.base_yolo_confidence
        
        logger.info(f"Loading YOLO model {model_name}...")
        try:
            from ultralytics import YOLO
            import torch
            self.model = YOLO(model_name)
            
            # Respect compliance default (CPU-only unless explicitly allowed by config)
            if DEFAULT_CONFIG.hardware.force_cpu:
                self.device = 'cpu'
                logger.info("⚡ YOLO enforced to run on CPU (compliance default)")
            elif torch.backends.mps.is_available():
                self.device = 'mps'
                logger.info("🚀 YOLO accelerated via Apple Metal (MPS)")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("🚀 YOLO accelerated via CUDA")
            else:
                self.device = 'cpu'
                logger.info("YOLO running on CPU")
                
            logger.info("YOLO model loaded successfully.")
        except ImportError:
            logger.error("ultralytics library not found. YOLODetector requires it.")
            self.model = None

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO inference on the image.
        Returns a list of dicts with 'name', 'category' (same as name for yolo), 'bbox', 'confidence'.
        """
        if self.model is None:
            return []
            
        try:
            results = self.model(image, conf=self.confidence_threshold, device=self.device, verbose=False)
            
            detections = []
            if len(results) > 0:
                result = results[0]
                names = result.names
                boxes = result.boxes
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    
                    class_name = names[cls_id].lower().replace(" ", "_")
                    
                    detections.append({
                        "name": class_name,
                        "category": class_name,  # YOLO names usually serve as broad categories
                        "bbox": bbox,
                        "confidence": conf
                    })
                    
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")
            return []
