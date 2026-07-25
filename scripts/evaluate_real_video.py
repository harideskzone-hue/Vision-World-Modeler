import sys
import os
import argparse
import logging
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision_world.orchestrator import VisionOrchestrator
from evaluation.evaluator import Evaluator
from shared.enums import RelationType

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def evaluate_real_video(video_path: str, gt_path: str, extractor_type: str = "ollama"):
    print("=" * 50)
    print(f"EVALUATING REAL VIDEO: {video_path}")
    print(f"Ground Truth: {gt_path}")
    print(f"Extractor: {extractor_type}")
    print("=" * 50)
    
    orchestrator = VisionOrchestrator(video_path, extractor_type=extractor_type)
    evaluator = Evaluator(gt_path)
    
    # Run the pipeline and record results per frame
    for frame_data in orchestrator.video_source.stream():
        orchestrator.frame_buffer.append(frame_data)
        
        # SSIM Skip
        t0 = time.time()
        skip = orchestrator.change_detector.should_skip(frame_data)
        t_ssim = time.time() - t0
        
        if skip:
            logger.info(f"Frame {frame_data.frame_id} skipped (redundant) - SSIM {t_ssim*1000:.1f}ms")
            orchestrator.graph.update_node_confidence_occlusion(frame_data.frame_id)
            orchestrator.occlusion_handler.handle_occlusions()
            
            # Record current active state even if skipped
            active_state_edges = []
            active_location_edges = []
            for edge in orchestrator.graph.get_all_active_edges():
                if edge.relation == RelationType.HAS_STATE:
                    active_state_edges.append(edge)
                elif edge.relation == RelationType.LOCATED_IN:
                    active_location_edges.append(edge)
                    
            # We don't have new entity observations, but we could fetch current active nodes
            # For simplicity, pass empty list for entities on skipped frames or active nodes
            active_entities = [{"name": n.name, "category": n.category} for n in orchestrator.graph.get_all_active_nodes()]
            
            evaluator.record_frame(frame_data.frame_id, active_entities, active_state_edges, active_location_edges)
            continue
            
        # Vision Extraction
        vlm_obs = orchestrator.vlm_extractor.extract(frame_data.image_rgb, frame_data.frame_id)
        yolo_dets = orchestrator.yolo_detector.detect(frame_data.image_rgb)
        
        # Fusion
        fused_obs = orchestrator.fusion.fuse(vlm_obs, yolo_dets)
        
        # Tracking
        matched_obs = orchestrator.matcher.match(fused_obs)
        
        # Candidate Conversion
        candidates = orchestrator.converter.convert(matched_obs)
        
        # World Model Update
        report = orchestrator.updater.update(candidates, frame_data.frame_id)
        orchestrator.occlusion_handler.handle_occlusions()
        
        # Record for Evaluation
        active_state_edges = []
        active_location_edges = []
        
        for edge in orchestrator.graph.get_all_active_edges():
            if edge.relation == RelationType.HAS_STATE:
                active_state_edges.append(edge)
            elif edge.relation == RelationType.LOCATED_IN:
                active_location_edges.append(edge)
        
        # Evaluator expects entities as dicts with 'name' key, which is handled in Evaluator
        # matched_obs.entities is a list of Dict. We'll pass it.
        # Wait, matched_obs.entities might be a list of EntityObservation objects? Let's check. 
        # In EntityMatcher, it returns a SceneObservation with 'entities' which are dicts.
        
        evaluator.record_frame(frame_data.frame_id, matched_obs.entities, active_state_edges, active_location_edges)
        
        logger.info(f"Frame {frame_data.frame_id} evaluated.")
        
    orchestrator.video_source.release()
    print("\nRunning Evaluation Metrics...")
    evaluator.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on real video with GT")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--extractor", type=str, default="ollama")
    args = parser.parse_args()
    
    evaluate_real_video(args.video, args.ground_truth, extractor_type=args.extractor)
