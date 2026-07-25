# HackTronix 2.0 Track 2 Submission: Vision World Modeler

## Overview
This repository contains our final submission for HackTronix 2.0 Track 2. We have successfully engineered a persistent, self-correcting World Model capable of converting noisy, unconstrained video streams into a structured, queryable knowledge graph on CPU-constrained hardware (MacBook Air M1).

## Final Checklist Completed
- ✅ **Architecture**: Modular separation between Perception (VLM/YOLO) and Reasoning (Graph/Updater).
- ✅ **Hardware Constraints**: Successfully deployed Moondream2 + YOLO under 3GB peak RAM.
- ✅ **Persistence**: Tracked objects maintain stable UUIDs across severe occlusions (R4 Rule).
- ✅ **Belief Revision**: Contradictory VLM hallucinations are gracefully blocked by our R1/R2 Revision Policies.
- ✅ **Scene Management**: Instant spatial unbinding on camera scene changes (R5 Rule).
- ✅ **Evaluation Framework**: C1/C2/C3 consistency metrics mathematically prove the stability of the World Model.
- ✅ **Graph Export**: Fully supports `.json`, `.graphml`, and `.dot` for downstream visualization.
- ✅ **Transparency**: CLI includes an Explanation Mode breaking down exact match reasoning and evidence for every entity.

## Supporting Documents
- `README.md` - Installation, usage, and architecture overview.
- `DECISIONS.md` - Technical trade-offs and rationale behind rules and models.
- `results/logs/benchmark.log` - The raw latency and memory performance captured during the real video benchmark.
- `evaluation_report.md` - The final accuracy and consistency metrics.

## Demo Execution
Reviewers and judges can instantly verify the system functionality via:
```bash
./run_demo.sh
```
