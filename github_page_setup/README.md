# Object Retrieval in Dynamic Workspace Environments — GitHub Pages Live Demo (`github_page_setup`)

> **Live GitHub Pages Build:** Configured with `base: '/minor_project/'`, flattened ONNX Runtime Web WASM/WebGPU binaries (`ort-dist/`), and **REACT++ YOLOv12n** (`45.4 MB` ONNX) running directly in the visitor's browser.  
> **VLM Note:** `SmolVLM-500M` (`518.2 MB`) is intentionally skipped (`VLM_CONFIG.enabled = false`) to keep the live web demo lightweight; visual/VLM fallback queries report *"VLM unavailable right now"*, while Scene Graph Generation, Multi-Frame Tracking, IndexedDB Memory, NLQ Reasoning, and Session Logs run in full.

---

A browser-first, local-first visual workspace intelligence prototype built with **React 19**, **TypeScript**, **Vite**, **ONNX Runtime Web**, **IndexedDB**, and **REACT++**.

The application converts RGB workspace photographs into structured 2D scene graphs, maintains persistent physical object identities across multiple observations, supports natural-language queries with verifiable frame provenance, and logs structured session telemetry locally in the browser.

---

## 1. End-to-End System Pipeline

```text
Global Viewport Drag-and-Drop / "+ Upload Images" (Single or Batch)
       ↓
REACT++ SGG Engine (Dedicated Web Worker: WebGPU or Multi-Threaded WASM SIMD)
       ↓
Vocabulary Adapter (OpenPSG 133 classes → 39 Workspace Categories, 57 → 6 Canonical Predicates)
       ↓
Visual Crop Descriptor + Identity Association (Global 1-to-1 matching via IoU, distance, & RGB histograms)
       ↓
2D Scene Graph Builder (<Subject, Predicate, Object> directed triplets + workspace_surface_001)
       ↓
IndexedDB Workspace Memory (workspace_retrieval_db v2 — 11 object stores via idb)
       ↓
Multi-Frame Timeline & Observation Grid (Interactive frame selection & strict frame isolation)
       ↓
Workspace Assistant NLQ Engine (Deterministic Slot Parser + Fuse.js Entity Resolver)
   ├── [Confirmed SGG / Memory Evidence] ──→ Verifiable Answer + "View Frame" Provenance Navigation
   └── [Visual Gap / Attribute / /visual] ─→ Lazy-Loaded SmolVLM-500M WebGPU Worker (Provisional, Zero-BBox)
```

---

## 2. Implemented Capabilities

### 2.1 Local-First Scene Graph Generation (REACT++)
- **In-Browser ONNX Inference:** Runs `react_pp_yolo12n.onnx` (45.4 MB, OpenPSG checkpoint) inside `src/workers/reactpp.worker.ts` using `onnxruntime-web` (`wasm` SIMD multi-threaded fallback or `webgpu` acceleration).
- **Pre/Postprocessing:** Aspect-ratio preserving $640\times 640$ letterbox padding (`reactppPreprocess.ts`) and exact inverse coordinate un-letterboxing (`reactppPostprocess.ts`).
- **Ontology Integrity (`vocabularyAdapter.ts`):** Maps overlapping OpenPSG classes into 39 workspace categories across 5 domains (`student_workspace`, `home_office_workspace`, `art_craft_workspace`, `electronics_maker_workspace`, `repair_maintenance_workspace`) and 57 predicates into 6 canonical directed predicates (`on`, `under`, `next_to`, `in_front_of`, `behind`, `connected_to`). Out-of-vocabulary detections retain their raw model label with `mapped: false` (`unmapped`) rather than coercing false categories.

### 2.2 Multi-Frame Ingestion, Timeline & Comparative Grid
- **Global Drag-and-Drop & Unified Upload (`App.tsx`, `FrameSelector.tsx`):** Drop single or multiple images anywhere on the application viewport, or use the `+ Upload Images` button.
- **One-Click Timeline Selection:** Click any frame card in `FrameSelector` to toggle it on/off, or use `Select All` / `Clear Selection`. Selection changes filter UI panels and NLQ context immediately without re-running SGG inference.
- **Responsive Multi-Frame Observation Canvas (`WorkspaceObservation.tsx`):** Automatically switches between single-frame inspection, 2-column side-by-side comparison, and 3+-column responsive grids. Each tile uses an isolated SVG `viewBox` (`BoundingBoxOverlay.tsx`) supporting 4 overlay modes (`focus`, `objects`, `relationships`, `clean`) and an overlay prominence slider.
- **Safe Atomic Frame Deletion:** Deleting a frame cascades cleanly across observations, relationships, and orphaned instances while preserving unrelated frames.

### 2.3 Cross-Frame Physical Identity & Visual Re-ID
- **Visual Crop Descriptors (`visualDescriptor.ts`):** Computes lightweight scale-invariant appearance descriptors (mean/std RGB, 48-bin channel histograms, and aspect ratio) from bounding box crops in both browser Canvas and Node.js test environments.
- **Global 1-to-1 Identity Matching (`identityAssociation.ts`):** Associates detections with existing `PersistentObjectInstance` IDs (e.g., `cup_001`, `laptop_001`) using IoU, normalized center displacement, category/synonym compatibility, and visual appearance similarity, preventing identity swaps when objects move across observations (`relocated_visual_match` vs `stationary_high_iou`) and recording `first_seen` and `moved` events.

### 2.4 Workspace Assistant (Deterministic NLQ + Conditional VLM)
- **Deterministic Query Pipeline (`src/nlq/`):** Regex/grammar slot extraction (`parser.ts`) + fuzzy entity resolution via `Fuse.js` (`entityResolver.ts`) + graph/memory reasoning (`queryExecutor.ts`).
- **Supported Intents & Scopes:** Supports `object_location`, `last_seen_location`, `spatial_relationship`, `historical_relationship`, `object_attribute`, `object_change`, `describe_scene`, `ambiguous_visual`, and `unsupported_occluded`, scoped across `current`, `history`, `all`, `/sgg`, and `/visual`.
- **Interactive Disambiguation & Provenance:** Prompts the user with clickable candidate cards when multiple instances match (e.g. `person_001` vs `person_002`). Clicking **View Frame** jumps directly to the supporting frame, highlights the target bounding box, and scrolls it into view.
- **Conditional SmolVLM-500M Fallback (`src/models/vlm/`, `smolvlm.worker.ts`):** Lazy-loaded via WebGPU/Cache API only when structured scene graph facts cannot answer visual attribute queries or when `/visual` is specified. Enforces a **Zero-Coordinate Safeguard** (`targetBbox = null`) so VLM answers remain provisional and never alter calculated SGG bounding boxes.

### 2.5 Telemetry & Persistent Session Audit Logging
- **Diagnostics & Session Logs Drawer (`DiagnosticsPanel.tsx`, `src/logging/`):**
  - **Telemetry Tab:** Displays active execution provider, WebGPU hardware detection, checkpoint size, step-by-step latency (`preprocessMs`, `inferenceMs`, `postprocessMs`, `totalMs`), and IndexedDB instance/frame counts.
  - **Session Logs Tab:** Captures non-blocking buffered structured events (`upload`, `frame`, `inference`, `detection`, `identity`, `scene_graph`, `memory`, `query`, `vlm`, `deletion`, `error`) into the `sessionLogs` IndexedDB store, with level/category filtering, keyword search, modal JSON inspection, and **JSON / CSV export**.

---

## 3. IndexedDB Storage Schema (`workspace_retrieval_db` v2)

Defined in [`src/memory/dbSchema.ts`](src/memory/dbSchema.ts) and managed via [`src/memory/db.ts`](src/memory/db.ts), [`src/memory/memoryRepository.ts`](src/memory/memoryRepository.ts), and [`src/logging/logRepository.ts`](src/logging/logRepository.ts):

| Store Name | Key Path | Purpose |
| :--- | :--- | :--- |
| `workspaces` | `workspaceId` | Workspace metadata and active domain profile |
| `sequences` | `sequenceId` | Observation sequence grouping |
| `frames` | `frameId` | Frame timestamps, image dimensions, thumbnails, and image blobs |
| `objectInstances` | `instanceId` | Persistent physical entity identities and visual descriptors |
| `objectObservations` | `annotationId` | Per-frame bounding boxes, confidence scores, and crop descriptors |
| `relationships` | `relationId` | Directed `<subject, predicate, object>` triplets per frame |
| `graphEvents` | `eventId` | Lifecycle events (`first_seen`, `moved`, etc.) |
| `vlmCandidates` | `candidateId` | Provisional visual facts returned by SmolVLM fallback |
| `queryHistory` | `queryId` | Executed NLQ queries, plans, and provenance answers |
| `appSettings` | `key` | Local client configuration preferences |
| `sessionLogs` | `logId` | Structured session audit events indexed by `sessionId`, `category`, and `timestamp` |

---

## 4. Directory Structure

```text
workspace-retrieval-web/
├── docs/
│   ├── ARCHITECTURE.md                         # High-level subsystem diagram & contracts
│   ├── DATA_SCHEMA_MAPPING.md                  # Vocabulary mapping & IndexedDB stores
│   ├── MODEL_COMPATIBILITY.md                  # ONNX / WASM / SmolVLM compatibility audit
│   ├── DEMO_TECHNICAL_AND_FUNCTIONAL_DOSSIER.md# Full engineering dossier
│   ├── DEMO_MANIFEST.json                      # Machine-readable feature & model manifest
│   └── demo_assets/                            # UI screenshots for documentation
├── public/
│   ├── favicon.svg
│   ├── models/react_pp_yolo12n.onnx            # 45.4 MB quantized/exported REACT++ ONNX model
│   └── samples/sample_workspace_1.jpg          # Sample workspace test image
├── src/
│   ├── app/                                    # App shell, multi-frame state, and dark theme CSS
│   ├── components/                             # UI workstation panels & SVG overlays
│   ├── config/                                 # Workspace domains, 39 categories, 6 predicates, OpenPSG vocab
│   ├── graph/                                  # SceneGraph builder & reasoner
│   ├── logging/                                # Buffered session logger, IndexedDB repository, JSON/CSV export
│   ├── memory/                                 # IndexedDB schema, repositories, & identity association
│   ├── models/
│   │   ├── sgg/                                # REACT++ ONNX/local engines, preprocess, postprocess, vocab adapter
│   │   └── vlm/                                # SmolVLM-500M ExecuTorch engine, prompts, response schema
│   ├── nlq/                                    # Deterministic NLQ parser, Fuse.js resolver, & query executor
│   ├── pipeline/                               # PipelineOrchestrator coordinating inference & queries
│   ├── utils/                                  # Frame selection isolation, visual descriptors, colors, IDs, time
│   └── workers/                                # Background Web Workers (reactpp.worker.ts, smolvlm.worker.ts)
├── tests/
│   ├── runAllTests.ts                          # Master runner executing all 8 suites
│   ├── sgg/vocabularyAdapter.test.ts           # OpenPSG → workspace mapping tests
│   ├── sgg/postprocess.test.ts                 # Un-letterboxing & triplet parsing tests
│   ├── memory/identityAssociation.test.ts      # Spatial tracking & event tests
│   ├── nlq/parser.test.ts                      # NLQ slot parsing & slash-command tests
│   ├── workspaceAssistant.test.ts              # 16 Workspace Assistant acceptance tests
│   ├── multiFrame.test.ts                      # 10 Multi-frame comparative reasoning tests (A–J)
│   ├── sessionAndIsolation.test.ts             # 46 Upload, selection, deletion, isolation & logging tests
│   ├── crossFrameReidRegression.test.ts        # Cross-frame visual Re-ID regression test
│   └── fixtures/workspace-memory-fixture.json  # Captured multi-frame regression fixture
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts                              # Serves COOP/COEP headers & copies ORT WASM/MJS binaries
```

---

## 5. Setup, Build & Verification

### Prerequisites
- **Node.js** `v18+` (verified on `v22.23.4`)
- Modern browser with WebAssembly SIMD and optional WebGPU support (Chrome / Edge / Brave)

### Commands
```bash
cd workspace-retrieval-web

# Install dependencies
npm install

# Start Vite development server (http://127.0.0.1:5173/)
npm run dev

# Type-check and build production bundle into dist/
npm run build

# Run linter (oxlint)
npm run lint

# Run all 8 automated unit, integration, & regression test suites
npx tsx tests/runAllTests.ts
```

### Additional Documentation
- [System Architecture](docs/ARCHITECTURE.md)
- [Data Schema & Vocabulary Mapping](docs/DATA_SCHEMA_MAPPING.md)
- [Model Compatibility Report](docs/MODEL_COMPATIBILITY.md)
- [Complete Technical & Functional Dossier](docs/DEMO_TECHNICAL_AND_FUNCTIONAL_DOSSIER.md)
