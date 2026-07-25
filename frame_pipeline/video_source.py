# frame_pipeline/video_source.py
import cv2
import time
from typing import Iterator, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameData:
    frame_id: int
    image_rgb: np.ndarray
    timestamp_sec: float

class FileVideoSource:
    def __init__(self, filepath: str, target_fps: float = 1.0):
        self.filepath = filepath
        self.target_fps = target_fps
        
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {filepath}")
            
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate how many source frames to skip to hit target FPS
        if self.source_fps <= 0:
            self.source_fps = 30.0  # Fallback
            
        self.frame_interval = max(1, int(self.source_fps / self.target_fps))
        self.current_frame_id = 0
        
    def stream(self) -> Iterator[FrameData]:
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_idx % self.frame_interval == 0:
                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_sec = frame_idx / self.source_fps
                
                yield FrameData(
                    frame_id=self.current_frame_id,
                    image_rgb=rgb_frame,
                    timestamp_sec=timestamp_sec
                )
                self.current_frame_id += 1
                
            frame_idx += 1
            
    def release(self):
        if self.cap:
            self.cap.release()
