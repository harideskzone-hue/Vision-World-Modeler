# evaluation/ground_truth.py
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class GTEntity:
    name: str
    category: str
    position: Optional[str]
    state: Optional[str]
    present: bool

@dataclass
class GTFrame:
    frame_id: int
    scene_type: str
    entities: List[GTEntity]

@dataclass
class GTAnnotation:
    video: str
    total_frames: int
    scene_types: List[str]
    frames: List[GTFrame]

def load_ground_truth(filepath: str) -> GTAnnotation:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    frames = []
    for frame_data in data.get("frames", []):
        entities = []
        for ent_data in frame_data.get("entities", []):
            entities.append(GTEntity(
                name=ent_data["name"],
                category=ent_data["category"],
                position=ent_data.get("position"),
                state=ent_data.get("state"),
                present=ent_data.get("present", True)
            ))
        frames.append(GTFrame(
            frame_id=frame_data["frame_id"],
            scene_type=frame_data["scene_type"],
            entities=entities
        ))
        
    return GTAnnotation(
        video=data["video"],
        total_frames=data["total_frames"],
        scene_types=data["scene_types"],
        frames=frames
    )
