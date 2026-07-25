# Architecture Deep-Dive: Vision World Modeler

This document provides a technical overview of the four decoupled processing layers comprising our Vision World Modeler architecture, engineered specifically to satisfy HackTronix 2.0 Track 2 deliverables on Apple Silicon (MacBook Air M1, 8GB Unified Memory).

---

## 1. Modular Layer Overview

```
 [Video Input Stream @ 1FPS]
             │
             ▼
 ┌──────────────────────┐
 │ 1. Frame Pipeline    │ ──(SSIM Structural Gating >= 0.95 skip)
 └───────────┬──────────┘
             │ Active Frames (~20%)
             ▼
 ┌──────────────────────┐
 │ 2. Vision Extractor  │ ──(Moondream2 Semantics + YOLO Bboxes + CLIP Zero-Shot)
 └───────────┬──────────┘
             │ Unlabeled Spatial & Semantic Observations
             ▼
 ┌──────────────────────┐
 │ 3. Object Tracker    │ ──(Entity Registry + IoU Bounding + Stable ID Allocation)
 └───────────┬──────────┘
             │ Structured Candidate Facts (Subject, Relation, Object, Confidence)
             ▼
 ┌──────────────────────┐
 │ 4. Belief Updater    │ ──(R1-R5 Revision Rules + Contradiction Superseding)
 └───────────┬──────────┘
             │ Verified Temporal Graph Updates
             ▼
 ┌──────────────────────┐
 │ 5. World Graph Store │ ──(Property Edge Database + O1 Bounded Pruning + REPL)
 └──────────────────────┘
```

---

## 2. Module Specifications

### `vision_extractor/` (Deliverable D3)
Deception-free zero-shot visual comprehension engine operating without hardcoded semantic mapping tables:
* **`moondream_extractor.py`**: Interacts with local Ollama GGUF backend (`moondream`) to extract natural language environmental scene descriptions and conceptual state annotations (e.g., *open*, *empty*, *clean*).
* **`yolo_detector.py`**: Executes `yolo26n.pt` accelerated via Apple Silicon Metal Performance Shaders (MPS) to isolate realtime spatial geometric coordinates and bounding box metrics.
* **`clip_reidentifier.py` / `semantic_fusion.py`**: Compares semantic text tokens and observed image patches using 512-dimensional CLIP vector embeddings (`clip-vit-base-patch32`), aligning disparate object references purely through neural cosine similarity without keyword lookup dicts.

---

### `object_tracker/` (Deliverable D1 / C1 Invariance)
Maintains continuous identity mapping across prolonged occlusions and shifts in viewing angles:
* **`entity_registry.py`**: Serves as the single source of truth for stable ID allocation (`desk_chair_1`, `coffee_maker_1`). Prevents duplicate phantom identity split generation.
* **`entity_matcher.py`**: Fuses intersection-over-union (IoU) bounding spatial continuity with semantic confidence weights to match live observations against historical graph nodes.
* **`occlusion_handler.py`**: Monitors temporal observation freshness (`stale_threshold_frames: 30`). When objects drop out of camera view, it gracefully transitions them from `ACTIVE` to `OCCLUDED` and finally to `ARCHIVED`, syncing graph status with the entity registry to prevent erroneous identity rebinding upon room transitions.

---

### `updater/` & State Reconciliation (Deliverable D2 / D4)
Treats VLM detections as *Candidate Facts*, processing them through deterministic Belief Revision Rules (R1-R5):
* **Rule R1 (Temporal Recency & Dominance):** Newer observations with equal or higher confidence override superseded historical states upon verified physical changes.
* **Rule R2 (Corroboration Threshold):** Transient visual hallucinations must outscore historical priors (+0.08 multi-frame corroboration boost) before overturning confirmed properties.
* **Rule R3 (Semantic Contradiction Defusal):** Explicit physical antonyms (e.g., *open* vs *closed*) immediately terminate the temporal validity window (`t_valid_until`) of the preceding state edge, setting its status to `SUPERSEDED`.
* **Rule R4 (Occlusion Decay & Reactivation):** Temporarily occluded entities retain stable graph positioning with gradual confidence decay (-0.03/frame), instantly restoring to full active confidence upon visual reappearance.
* **Rule R5 (Scene Transition Archival):** When Moondream detects a fundamental change in environmental classification (e.g., migrating from *Library* to *Classroom*), all preceding local entities are instantly transitioned to `ARCHIVED` to preserve strict single-occupancy spatial consistency (C3).

---

### `world_model/` & Bounded Pruning (Objective O1)
An efficient in-memory property graph store optimized for zero-swap RAM execution on constrained Apple Silicon hardware:
* **`graph_store.py`**: Indexes directed relationship edges (`subject --[relation]--> object`) with strict UNIX creation timestamps and frame validity indexes.
* **`self.prune()` Automatic Eviction:** Monitors graph density against configured capacity ceilings (`max_active_entities: 100`, `max_active_edges: 300`). When thresholds are reached, the oldest permanently archived historical nodes and superseded edges are systematically freed from process memory, guaranteeing flat ~110MB RAM usage across indefinite video streaming durations.
* **`temporal_versioning.py`**: Supports historical replay and time-travel querying, allowing evaluators to reconstruct exact graph states at any prior video frame index.
