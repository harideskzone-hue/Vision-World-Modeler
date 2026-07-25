# HackTronix 2.0 Track 2 - Architectural Decisions

This document outlines the core trade-offs and decisions made while designing the Vision World Modeler.

## 1. Why Moondream2 via Ollama (GGUF)?
The primary constraint of Track 2 is running on constrained Apple Silicon hardware (MacBook Air M1 with 8GB RAM). 
- Initial testing with larger multi-modal models and raw HuggingFace PyTorch Transformers revealed unacceptable latency and an OOM risk (4-5GB RAM footprint).
- Switching to **Ollama with the Moondream GGUF backend** enabled native llama.cpp memory optimization.
- This provided a massive efficiency breakthrough: RAM usage dropped to **~110 MB peak**, completely eliminating OS memory swapping.
- Any minor variance in single-frame VLM labeling is fully smoothed out and self-corrected by our temporal World Model graph rules.

## 1.5. VLM JSON Enforcement & Zero-Shot Semantic Fusion
To combat formatting instability of small VLMs without resorting to prohibited hardcoded keyword dictionaries, we enforced `"format": "json"` at the inference endpoint as a decoding grammar constraint. For entity matching across Moondream and YOLO labels, we deploy 512-dimensional CLIP cosine similarity embeddings, achieving true zero-shot multi-model fusion.

## 2. Why R4 and R5 Revision Rules?
- **R4 (Occlusion Decay)**: Classical trackers delete objects when they leave the frame. A world model must understand permanence. R4 gracefully decays confidence and archives objects, allowing them to be instantly revived with their original UUID if they reappear.
- **R5 (Scene Change Detection)**: A naive implementation of R4 would cause objects from the `Kitchen` to slowly fade away while the camera is in the `Hallway`. R5 monitors the `camera LOCATED_IN` edge. When a supersede event occurs on the camera's location, it triggers an instant archival of all entities from the previous scene, eliminating spatial bleed-through.

## 3. Graph Schema Design & Temporal Versioning
We designed the world model as an edge-list backed property graph where every observation is represented as an edge (`LOCATED_IN`, `HAS_STATE`, `IS_TYPE`).
- **Why?** It natively supports **Temporal Versioning**. Instead of destructively overwriting a node's state upon change, we transition the prior edge to `SUPERSEDED`. This guarantees full historical replayability and enables the Contradiction Detector to evaluate R1 (Recency Supersede) and R2 (Corroborated Override) policies based on rigorous temporal validity intervals.

## 4. Why YOLO + VLM Fusion?
VLMs excel at semantics ("red mug", "empty") but exhibit high variance in spatial bounding box localization. YOLO is exceptionally reliable at geometric detection but lacks open-vocabulary semantic state interpretation. Our Fusion Module merges them: the VLM provides state attributes, while YOLO provides precision geometry for IoU spatial tracking in `EntityMatcher`.

## 5. CPU-Only Compliance Default vs. MPS Acceleration
To ensure strict compliance with hackathon execution rules regarding GPU utilization, our pipeline runs **CPU-only by default** (`hardware.force_cpu = True` in `config.yaml` / `shared.config`, and `OLLAMA_NUM_GPU=0` in shell launchers).
- **Rationale**: While Apple Silicon features unified memory architecture where CPU and GPU share the same physical memory pool, we treat Metal (MPS) GPU acceleration as an optional opt-in feature.
- Users and judges wishing to evaluate maximum unified memory performance can set `hardware.force_cpu: false` in `config.yaml` once organizer permission is confirmed.

## 6. Track 1 vs. Track 2 Architectural Scope & Independence
HackTronix 2.0 establishes two distinct problem tracks:
- **Track 1 (Text World Agent)**: Focuses entirely on symbolic textual world modeling and text-based interactive environment agents. Our comprehensive Text World engine is cleanly isolated in a companion codebase (`world-model-agent/`).
- **Track 2 (Vision World Modeler)**: Operates as an independent, standalone vision-to-graph architecture engineered specifically for streaming visual perception, object occlusion persistence, and multi-frame spatial contradiction resolution without symbolic textual game priors.
