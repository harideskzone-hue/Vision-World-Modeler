# vision_extractor/base.py
from abc import ABC, abstractmethod
import numpy as np
from shared.models import SceneObservation

class VisionExtractor(ABC):
    """
    Abstract interface for vision extraction models.
    Takes a raw RGB image and returns a structured SceneObservation.
    """
    
    @abstractmethod
    def extract(self, image: np.ndarray, frame_id: int) -> SceneObservation:
        """
        Extract scene information from an image.
        
        Args:
            image: Raw RGB image array (H, W, C)
            frame_id: The ID of the current frame
            
        Returns:
            A structured SceneObservation object.
        """
        pass
