#!/bin/bash
# run_demo.sh - Automated Demonstration Script for HackTronix Track 2

set -e

VIDEO_PATH=${1:-"demo.mp4"}

echo "=================================================="
echo "VISION WORLD MODELER - HACKTRONIX TRACK 2 DEMO"
echo "=================================================="

echo "[1/4] Verifying Environment..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found."
    exit 1
fi

echo "[2/4] Running End-to-End Pipeline on ${VIDEO_PATH}..."
# In a real run, this would be a real video. For demo purposes, we will mock it if the file doesn't exist.
if [ -f "$VIDEO_PATH" ]; then
    python3 scripts/benchmark.py "$VIDEO_PATH" > results/logs/benchmark.log 2>&1
    echo "Pipeline execution complete! Latency and timeline saved to results/logs/benchmark.log"
else
    echo "WARNING: Video ${VIDEO_PATH} not found. Running in MOCK mode to demonstrate architecture."
    python3 cli.py --mock --inspect history > results/logs/mock_run.log
    
    # We will trigger the graph export manually via python inline snippet for the demo
    python3 -c "
from tests.test_corrections import MockPipeline
p = MockPipeline()
p.process(1, 'kitchen', [{'name': 'mug', 'category': 'container', 'confidence': 0.9}])
p.graph.export_json('results/graphs/graph.json')
p.graph.export_graphml('results/graphs/graph.graphml')
p.graph.export_dot('results/graphs/graph.dot')
"
    
    echo "Mock pipeline execution complete."
fi

echo ""
echo "[3/4] Exporting Knowledge Graph State..."
echo "Knowledge graph exported to:"
echo " - results/graphs/graph.json"
echo " - results/graphs/graph.graphml"
echo " - results/graphs/graph.dot"

echo ""
echo "[4/4] Executing Sample Queries on World Model..."
echo "--------------------------------------------------"
echo "> Query: What are the active entities?"
python3 cli.py --mock --inspect "entities"

echo ""
echo "> Query: Show me the history of changes."
python3 cli.py --mock --inspect "history"

echo ""
echo "> Query: Where is the mug and what is the evidence? (Explanation Mode)"
python3 cli.py --mock --inspect "object mug"
echo "--------------------------------------------------"

echo ""
echo "=================================================="
echo "DEMO COMPLETE."
echo "Use 'python3 cli.py --help' to run custom queries."
echo "=================================================="
