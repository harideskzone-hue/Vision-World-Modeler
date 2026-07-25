# scripts/benchmark.py
import argparse
import sys
import os
import psutil
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision_world.orchestrator import VisionOrchestrator

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_benchmark(video_path: str, extractor_type: str = "ollama"):
    print("=" * 50)
    print(f"RUNNING BENCHMARK ON: {video_path} (Extractor: {extractor_type})")
    print("=" * 50)
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    peak_memory = 0
    
    # We will poll memory periodically if we were threaded, but for a single-threaded benchmark
    # we can just check it before and after, or rely on psutil's maxrss
    
    try:
        orchestrator = VisionOrchestrator(video_path, extractor_type=extractor_type)
        graph = orchestrator.run()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return
        
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # In a real environment, we'd poll this asynchronously
    memory_info = process.memory_info()
    peak_memory_mb = memory_info.rss / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Peak RAM Usage: {peak_memory_mb:.2f} MB")
    last_frame = orchestrator.frame_buffer.get_latest()
    processed_frames = last_frame.frame_id if last_frame else 0
    print(f"Processed Frames: {processed_frames}")
    print("=" * 50)
    
    # For a full evaluation, the user will pipe the orchestrator logs to a file 
    # and use a separate parser to compute average VLM/YOLO latency from the timelines.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline benchmark on a real video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--extractor", type=str, choices=["ollama", "moondream"], default="ollama", help="Vision extractor backend (default: ollama)")
    args = parser.parse_args()
    
    run_benchmark(args.video, extractor_type=args.extractor)
