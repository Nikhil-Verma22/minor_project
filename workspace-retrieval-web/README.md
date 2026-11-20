# Object Retrieval in Dynamic Workspace Environments

A browser-first, local-first research prototype for RGB workspace understanding, combining **REACT++** Scene Graph Generation (SGG), persistent **IndexedDB** temporal memory, deterministic **NLQ graph reasoning**, and **SmolVLM-500M WebGPU** visual fallback.

---

## 1. System Pipeline Overview

```
Upload RGB Image
       ↓
REACT++ SGG Engine (Web Worker: WebGPU or WASM SIMD CPU)
       ↓
Vocabulary Adapter (OpenPSG 133 classes → 39 Workspace Categories, 6 Canonical Predicates)
       ↓
Identity Association (Re-identifies persistent physical objects; detects movement events)
       ↓
2D Scene Graph Builder (Constructs typed directed graph: <Subject, Predicate, Object>)
       ↓
Workspace Memory Store (IndexedDB via idb: frames, observations, relationships, events)
       ↓
Natural-Language Query (NLQ) Engine (Rule-based slot parser + Fuse.js fuzzy entity resolver)
       ↓
Graph & Memory Reasoning Engine
   ├── [Sufficient Evidence] ───────────────→ Direct Answer with Provenance
   └── [Unresolved / Ambiguous Relation] ───→ SmolVLM-500M ExecuTorch WebGPU Fallback
                                                    ↓
                                             Provisional Fact Quarantine & User Review
```

---

## 2. Key Features

- **Strictly Local-First & Private:** Default execution is entirely inside the user's browser. Zero image uploads to external cloud APIs or third-party servers.
- **REACT++ Browser Inference:** Pretrained PSG checkpoint (`yolo12n`, 45.4 MB) executes directly via `onnxruntime-web` with multi-threaded WASM SIMD CPU execution (~200ms latency) and WebGPU acceleration where supported.
- **Fixed Project Ontology:**
  - **5 Domains:** `student_workspace`, `home_office_workspace`, `art_craft_workspace`, `electronics_maker_workspace`, `repair_maintenance_workspace`.
  - **39 Object Categories:** Full taxonomy spanning common stationary, office gear, electronics prototyping boards, and maintenance tools.
  - **Canonical Workspace Entity:** `workspace_surface_001` (category: `workspace_surface`).
  - **6 Canonical Directed Predicates:** `on`, `under`, `next_to`, `in_front_of`, `behind`, `connected_to`.
- **Temporal Memory & State Tracking:** Tracks object movement events across sequential observations from fixed camera viewpoints, maintaining persistent `instanceId` across time.
- **Selective VLM Fallback:** SmolVLM-500M (518.2 MB package, strictly below the 2 GB budget) is lazy-loaded and called **only** when structured scene graph facts cannot answer the query. VLM-derived facts are quarantined as provisional.
- **Telemetry & Diagnostics:** Real-time latency timers for preprocess, inference, postprocess, memory writes, and active execution providers.

---

## 3. Quick Start

### Prerequisites
- Node.js `v18+` (Tested on Node.js `v22.23.4`)
- Modern WebGPU or WASM-capable browser (Chrome / Edge 113+, Safari 18+)

### Development Server
```bash
# Navigate to web application directory
cd workspace-retrieval-web

# Install dependencies (already pre-installed)
npm install

# Start Vite dev server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### Production Build & Preview
```bash
# Build production bundle
npm run build

# Preview production build locally
npm run preview
```

### Running Automated Test Suite
```bash
npx tsx tests/runAllTests.ts
```
The test suite validates:
1. `VocabularyAdapter`: PSG 133 classes to 39 workspace classes, 57 predicates to 6 canonical predicates.
2. `Postprocessing`: Coordinate un-letterboxing, score filtering, and triplet unpacking.
3. `IdentityAssociation`: Spatial IoU tracking, moved event detection, and persistent ID assignment.
4. `QueryParser`: Intent classification, entity slot extraction, and predicate synonym resolution.

---

## 4. Supported Natural Language Queries

| Query Example | Intent | Expected Reasoning Source |
|---|---|---|
| *"Where is the laptop?"* | `locate_current_object` | **Scene Graph** (Current frame detection & surface relations) |
| *"What is beside the laptop?"* | `find_current_relationship` | **Scene Graph** (Triplets matching `next_to`) |
| *"Where was my phone last seen?"* | `locate_last_seen_object` | **Workspace Memory** (Latest historical observation) |
| *"What was next to the notebook in the previous observation?"* | `find_historical_relationship` | **Workspace Memory** (Prior frame observation) |
| *"Did the screwdriver move?"* | `compare_object_state` | **Workspace Memory** (Centroid displacement across frames) |
| *"Which object is on the workspace surface?"* | `find_current_relationship` | **Scene Graph** (Triplets with `workspace_surface_001`) |
| *"Is the charger behind the monitor?"* (if unmapped by SGG) | `find_current_relationship` | **SmolVLM-500M Fallback** (Visual semantic reasoning) |

---

## 5. IndexedDB Storage Schema

Database Name: `workspace_retrieval_db` (Version 1)

1. `workspaces`: Physical workspace records and domain IDs.
2. `sequences`: Video/photograph sequences with camera viewpoint metadata.
3. `frames`: Chronological observation frames with dimensions and timestamps.
4. `objectInstances`: Persistent physical object identities (`instanceId`, `categoryId`, first seen, last seen, status).
5. `objectObservations`: Frame-level bounding boxes and confidence scores.
6. `relationships`: Directed relationship facts (`<subjectId, predicate, objectId>`).
7. `graphEvents`: Lifecycle and movement events (`first_seen`, `moved`, `became_occluded`).
8. `vlmCandidates`: Quarantined provisional facts suggested by SmolVLM.
9. `queryHistory`: Past NLQ queries and responses.
10. `appSettings`: User preferences and model configurations.

---

## 6. Model Attribution & Licenses

- **REACT++**: Maelic et al., *REACT++: Faster and Better Scene Graph Generation*, arXiv:2603.06386. Checkpoint: `maelic/REACTPlusPlus_PSG` (Apache 2.0).
- **SmolVLM-500M-Instruct**: Hugging Face TB & ExecuTorch Community (`executorch-community/SmolVLM-500M-Instruct-ExecuTorch-WebGPU`, Apache 2.0).
- **ONNX Runtime Web**: Microsoft (`onnxruntime-web`, MIT License).
- **idb**: Jake Archibald (`idb`, ISC License).
- **Fuse.js**: Krisk (`fuse.js`, Apache 2.0).
