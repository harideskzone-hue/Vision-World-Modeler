# frame_pipeline/change_detector.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from frame_pipeline.video_source import FrameData
from shared.config import DEFAULT_CONFIG

class ChangeDetector:
    """Detects meaningful changes between frames to skip redundant processing."""
    
    def __init__(self, resize_width: int = 256):
        self.resize_width = resize_width
        self.threshold = DEFAULT_CONFIG.frame_pipeline.ssim_skip_threshold
        self.last_processed_gray: np.ndarray | None = None
        
    def preprocess_for_ssim(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize and convert to grayscale."""
        h, w = image_rgb.shape[:2]
        new_w = self.resize_width
        new_h = int((new_w / w) * h)
        
        resized = cv2.resize(image_rgb, (new_w, new_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return gray

    def should_skip(self, current_frame: FrameData) -> bool:
        """
        Returns True if the frame should be skipped (too similar to last processed),
        False if it contains new information and should be processed.
        """
        current_gray = self.preprocess_for_ssim(current_frame.image_rgb)
        
        if self.last_processed_gray is None:
            self.last_processed_gray = current_gray
            return False
            
        score, _ = ssim(self.last_processed_gray, current_gray, full=True)
        
        # If score is very high (close to 1.0), frames are nearly identical -> skip
        if score > self.threshold:
            return True
            
        # Significant change detected, keep this frame
        self.last_processed_gray = current_gray
        return False
