# frame_pipeline/frame_buffer.py
from typing import Optional, List
from frame_pipeline.video_source import FrameData

class FrameBuffer:
    """Fixed-size ring buffer for storing recent frames."""
    
    def __init__(self, size: int):
        self.size = size
        self._buffer: List[Optional[FrameData]] = [None] * size
        self._head = 0
        self._count = 0
        
    def append(self, frame: FrameData) -> None:
        """O(1) append, automatically overwrites oldest when full."""
        self._buffer[self._head] = frame
        self._head = (self._head + 1) % self.size
        self._count = min(self.size, self._count + 1)
        
    def get_latest(self) -> Optional[FrameData]:
        """O(1) access to most recent frame."""
        if self._count == 0:
            return None
        idx = (self._head - 1) % self.size
        return self._buffer[idx]
        
    def get_all_ordered(self) -> List[FrameData]:
        """Returns valid frames ordered from oldest to newest."""
        if self._count == 0:
            return []
            
        if self._count < self.size:
            return [f for f in self._buffer[:self._count] if f is not None]
            
        # Buffer is full, ordered from head (oldest) to tail (newest)
        return [self._buffer[(self._head + i) % self.size] for i in range(self.size)] # type: ignore
        
    def clear(self) -> None:
        self._buffer = [None] * self.size
        self._head = 0
        self._count = 0
