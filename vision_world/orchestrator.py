# vision_world/orchestrator.py
import logging
from typing import Iterator

from frame_pipeline.video_source import FileVideoSource
from frame_pipeline.frame_buffer import FrameBuffer
from frame_pipeline.change_detector import ChangeDetector
from vision_extractor.moondream_extractor import MoondreamExtractor
from vision_extractor.ollama_extractor import OllamaExtractor
from vision_extractor.yolo_detector import YOLODetector
from vision_extractor.semantic_fusion import SemanticFusionModule
from object_tracker.entity_registry import EntityRegistry
from object_tracker.entity_matcher import EntityMatcher
from vision_extractor.clip_reidentifier import CLIPReidentifier
from object_tracker.candidate_converter import CandidateConverter
from object_tracker.occlusion_handler import OcclusionHandler
from world_model.graph_store import InMemoryGraphStore
from updater.updater import Updater

logger = logging.getLogger(__name__)

class VisionOrchestrator:
    """
    Coordinates the end-to-end vision world modeling pipeline.
    """
    def __init__(self, video_path: str, extractor_type: str = "ollama"):
        # 1. Frame Pipeline
        self.video_source = FileVideoSource(video_path, target_fps=1.0)
        self.frame_buffer = FrameBuffer(size=30)
        self.change_detector = ChangeDetector()
        
        # 2. Vision Extractor & Fusion
        if extractor_type == "ollama":
            self.vlm_extractor = OllamaExtractor(model_name="moondream")
        else:
            self.vlm_extractor = MoondreamExtractor()
        self.yolo_detector = YOLODetector()
        self.clip_reid = CLIPReidentifier(lazy_load=False)
        self.fusion = SemanticFusionModule(clip_reid=self.clip_reid)
        
        # 3. Object Tracker
        self.registry = EntityRegistry()
        self.matcher = EntityMatcher(self.registry, clip_reid=self.clip_reid)
        self.converter = CandidateConverter()
        
        # 4. World Model & Updater
        self.graph = InMemoryGraphStore()
        self.updater = Updater(self.graph)
        self.occlusion_handler = OcclusionHandler(self.graph, self.registry)
        
    def run(self):
        """Runs the pipeline over the entire video and outputs a performance timeline."""
        import time
        logger.info("Starting Vision Orchestrator pipeline...")
        
        for frame_data in self.video_source.stream():
            self.frame_buffer.append(frame_data)
            
            # SSIM Skip
            t0 = time.time()
            skip = self.change_detector.should_skip(frame_data)
            t_ssim = time.time() - t0
            
            if skip:
                logger.info(f"Frame {frame_data.frame_id} skipped (redundant) - SSIM {t_ssim*1000:.1f}ms")
                # We still decay occluded objects on skipped frames
                self.graph.update_node_confidence_occlusion(frame_data.frame_id)
                self.occlusion_handler.handle_occlusions()
                continue
                
            # Vision Extraction
            t0 = time.time()
            vlm_obs = self.vlm_extractor.extract(frame_data.image_rgb, frame_data.frame_id)
            t_vlm = time.time() - t0
            
            t0 = time.time()
            yolo_dets = self.yolo_detector.detect(frame_data.image_rgb)
            t_yolo = time.time() - t0
            
            # Fusion
            t0 = time.time()
            fused_obs = self.fusion.fuse(vlm_obs, yolo_dets)
            t_fusion = time.time() - t0
            
            # Tracking
            t0 = time.time()
            matched_obs = self.matcher.match(fused_obs, current_frame=frame_data.image_rgb)
            t_match = time.time() - t0
            
            # Candidate Conversion
            t0 = time.time()
            candidates = self.converter.convert(matched_obs)
            t_conv = time.time() - t0
            
            # World Model Update
            t0 = time.time()
            report = self.updater.update(candidates, frame_data.frame_id)
            self.occlusion_handler.handle_occlusions()
            t_update = time.time() - t0
            
            total_time = t_ssim + t_vlm + t_yolo + t_fusion + t_match + t_conv + t_update
            
            logger.info(
                f"\nFrame {frame_data.frame_id} processed: expanded={report.expanded}, revised={report.revised}\n"
                f"Timeline Breakdown:\n"
                f"  SSIM        {t_ssim*1000:6.1f} ms\n"
                f"  VLM         {t_vlm*1000:6.1f} ms\n"
                f"  YOLO        {t_yolo*1000:6.1f} ms\n"
                f"  Fusion      {t_fusion*1000:6.1f} ms\n"
                f"  Matching    {t_match*1000:6.1f} ms\n"
                f"  Conversion  {t_conv*1000:6.1f} ms\n"
                f"  Updater     {t_update*1000:6.1f} ms\n"
                f"  --------------------------\n"
                f"  Total       {total_time*1000:6.1f} ms\n"
            )
            
        self.video_source.release()
        logger.info("Pipeline finished.")
        return self.graph
