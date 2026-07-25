# shared/config.py
import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class VisionExtractorConfig:
    base_vlm_confidence: float = 0.55
    base_yolo_confidence: float = 0.40
    base_fused_confidence: float = 0.65

@dataclass
class UpdaterConfig:
    corroboration_boost: float = 0.08
    occlusion_decay: float = -0.03
    r2_confidence_penalty: float = -0.20
    corroboration_threshold: int = 3
    confirmed_threshold: float = 0.75
    uncertain_threshold: float = 0.50
    max_confidence: float = 0.95
    min_confidence: float = 0.10

@dataclass
class GraphConfig:
    max_active_entities: int = 100
    max_active_edges: int = 300
    stale_threshold_frames: int = 30

@dataclass
class FramePipelineConfig:
    frame_ring_buffer_size: int = 30
    ssim_skip_threshold: float = 0.95

@dataclass
class HardwareConfig:
    force_cpu: bool = True   # compliance default; set to False only if organizers confirm MPS is allowed

@dataclass
class GlobalConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    vision_extractor: VisionExtractorConfig = field(default_factory=VisionExtractorConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    frame_pipeline: FramePipelineConfig = field(default_factory=FramePipelineConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> 'GlobalConfig':
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            if not data:
                return cls()
        
        return cls(
            hardware=HardwareConfig(**data.get('hardware', {})),
            vision_extractor=VisionExtractorConfig(**data.get('vision_extractor', {})),
            updater=UpdaterConfig(**data.get('updater', {})),
            graph=GraphConfig(**data.get('graph', {})),
            frame_pipeline=FramePipelineConfig(**data.get('frame_pipeline', {}))
        )

DEFAULT_CONFIG = GlobalConfig.load()
