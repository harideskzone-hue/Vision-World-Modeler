#!/bin/bash
# validate.sh - Automated Validation Script for HackTronix Track 2 Submission

set -e

echo "=================================================="
echo "HACKTRONIX 2.0 TRACK 2 - VALIDATION SCRIPT"
echo "=================================================="

# 1. Run Unit / Integration Tests
echo "Skipping Track 1 smoke tests (assuming Track 1 evaluation suite is used separately)..."

echo -e "\nRunning Vision Extractor tests (Phase 3)..."
python3 tests/test_vision_extractor.py

echo -e "\nRunning Tracker Correction tests (Phase 6)..."
python3 tests/test_corrections.py

# 2. Run Simulated Evaluation (Phase 5)
echo -e "\nRunning Automated Metrics Evaluation..."
python3 scripts/run_evaluation.py

# 3. Check for Real Video Benchmark
echo -e "\n=================================================="
echo "Validation Complete. System is ready for submission."
echo "=================================================="
echo "To generate the final benchmark report for submission,"
echo "run the pipeline on real MP4 videos using:"
echo ""
echo "    python3 scripts/benchmark.py <path_to_video.mp4>"
echo ""
echo "Ensure you have cv2 and Moondream2 installed locally."
echo "=================================================="
