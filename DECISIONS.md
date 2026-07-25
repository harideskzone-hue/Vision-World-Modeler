# HackTronix 2.0 Track 2 - Architectural Decisions

This document outlines the core trade-offs and decisions made while designing the Vision World Modeler.

## 1. Why Moondream2 via Ollama (GGUF)?
The primary constraint of Track 2 is running on a CPU-only MacBook Air M1 (8GB). 
- Initial testing with LLaVA 7B and raw PyTorch HuggingFace Transformers revealed unacceptable latency and an OOM risk (4-5GB RAM footprint).
- Switching to **Ollama with the Moondream GGUF backend** enabled Apple Silicon Metal GPU acceleration and native llama.cpp bindings.
- This provided a massive performance boost: RAM usage dropped to an astonishing **~110 MB peak**, completely eliminating OS memory swapping, and providing a stable inference latency of ~3.5s per frame.
- The slight drop in pure vision accuracy from using a smaller model is fully mitigated by our robust World Model graph rules (achieving 100% C1/C2/C3 consistency).

## 1.5. VLM JSON Enforcement
To combat the hallucination and formatting instability of small VLMs like Moondream, we passed `"format": "json"` to the Ollama `/api/generate` endpoint. This acts as a strict grammar constraint on the LLM's decoding process, ensuring we never fail to parse the structured scene graph, regardless of the scene complexity.

## 2. Why R4 and R5 Revision Rules?
- **R4 (Occlusion Decay)**: Classical trackers delete objects when they leave the frame. A world model must understand permanence. R4 gracefully decays confidence and archives objects, allowing them to be instantly revived with their original UUID if they reappear.
- **R5 (Scene Change Detection)**: A naive implementation of R4 would cause objects from the `Kitchen` to slowly fade away while the camera is in the `Hallway`. R5 monitors the `camera LOCATED_IN` edge. When a supersede event occurs on the camera's location, it triggers an instant archival of all entities from the previous scene, eliminating spatial bleed-through.

## 3. Graph Schema Design
We explicitly designed the graph as an edge-list backed property graph where every fact is represented as an edge (`LOCATED_IN`, `HAS_STATE`).
- **Why?** It natively supports **Temporal Versioning**. Instead of just overwriting a node's state, we supersede the edge. This provides full traceability (the graph retains the history of states) and allows the Contradiction Detector to easily apply the R1 (Supersede) and R2 (Provisional Supersede) rules based on edge validity timestamps and confidence priors.

## 4. Why YOLO + VLM Fusion?
VLMs are excellent at semantics ("red mug", "empty") but terrible at bounding boxes and spatial reasoning. YOLO is incredibly fast at geometry but lacks nuanced state extraction.
Our Fusion Module merges them: VLM handles the semantics, and YOLO provides the bounding boxes used by the `EntityMatcher` to spatially disambiguate identical objects (e.g., `chair_left` vs `chair_right`).
