# Data Schema & Vocabulary Mapping

**Project:** Object Retrieval in Dynamic Workspace Environments  
**Target:** 39 Workspace Object Classes, 6 Canonical Predicates, 5 Workspace Domains

---

## 1. Domain & Entity Specification

### The 5 Workspace Domains
1. `student_workspace`
2. `home_office_workspace`
3. `art_craft_workspace`
4. `electronics_maker_workspace`
5. `repair_maintenance_workspace`

### The Canonical Workspace Entity
- `workspace_surface_001` (category: `workspace_surface`)

---

## 2. Object Vocabulary Mapping (OpenPSG 133 $\rightarrow$ Workspace 39)

The pretrained REACT++ model was trained on the OpenPSG dataset (133 classes). The table below specifies the direct mapping into the project's 39 classes:

| Project Category (39) | Domain / Group | Mapped OpenPSG Class(es) | Mapping Confidence |
|---|---|---|---|
| `laptop` | Common Workspace | `laptop` | Direct (1.0) |
| `smartphone` | Common Workspace | `cell phone` | Direct (1.0) |
| `keyboard` | Common Workspace | `keyboard` | Direct (1.0) |
| `mouse` | Common Workspace | `mouse` | Direct (1.0) |
| `book` | Common Workspace | `book` | Direct (1.0) |
| `loose_paper` | Common Workspace | `paper` | Direct (1.0) |
| `water_bottle` | Common Workspace | `bottle` | Direct (0.9) |
| `scissors` | Common Workspace | `scissors` | Direct (1.0) |
| `monitor` | Common Workspace | `tv` | Direct (0.85) |
| `headphones` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `tablet` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `notebook` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `pen` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `pencil` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `marker` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `calculator` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `device_cable` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `charger` | Common Workspace | None (Pretrained OOV) | Fallback / VLM |
| `ruler` | Student | None (Pretrained OOV) | Fallback / VLM |
| `eraser` | Student | None (Pretrained OOV) | Fallback / VLM |
| `pencil_sharpener` | Student | None (Pretrained OOV) | Fallback / VLM |
| `printer` | Home/Office | None (Pretrained OOV) | Fallback / VLM |
| `document_folder` | Home/Office | None (Pretrained OOV) | Fallback / VLM |
| `stapler` | Home/Office | None (Pretrained OOV) | Fallback / VLM |
| `paintbrush` | Art/Craft | None (Pretrained OOV) | Fallback / VLM |
| `paint_tube` | Art/Craft | None (Pretrained OOV) | Fallback / VLM |
| `paint_palette` | Art/Craft | None (Pretrained OOV) | Fallback / VLM |
| `canvas` | Art/Craft | None (Pretrained OOV) | Fallback / VLM |
| `glue_bottle` | Art/Craft | None (Pretrained OOV) | Fallback / VLM |
| `arduino_board` | Electronics/Maker | None (Pretrained OOV) | Fallback / VLM |
| `breadboard` | Electronics/Maker | None (Pretrained OOV) | Fallback / VLM |
| `sensor_module` | Electronics/Maker | None (Pretrained OOV) | Fallback / VLM |
| `multimeter` | Electronics/Maker | None (Pretrained OOV) | Fallback / VLM |
| `jumper_wire` | Electronics/Maker | None (Pretrained OOV) | Fallback / VLM |
| `screwdriver` | Repair/Maintenance | None (Pretrained OOV) | Fallback / VLM |
| `pliers` | Repair/Maintenance | None (Pretrained OOV) | Fallback / VLM |
| `wrench` | Repair/Maintenance | None (Pretrained OOV) | Fallback / VLM |
| `hammer` | Repair/Maintenance | None (Pretrained OOV) | Fallback / VLM |
| `measuring_tape` | Repair/Maintenance | None (Pretrained OOV) | Fallback / VLM |

*Note: For out-of-vocabulary (OOV) classes, the raw OpenPSG model label is preserved in the detection record, with `projectCategory: null` and `mapped: false`.*

---

## 3. Predicate Mapping (OpenPSG 57 $\rightarrow$ Canonical 6)

| Canonical Predicate (6) | Mapped OpenPSG Predicate(s) | NLQ Natural Language Synonyms |
|---|---|---|
| `on` | `on`, `resting on`, `placed on`, `sitting on`, `standing on` | "on", "resting on", "placed on", "sitting on", "atop" |
| `under` | `under`, `beneath` | "under", "below", "beneath", "underneath" |
| `next_to` | `beside`, `standing next to`, `sitting next to`, `lying next to` | "next to", "beside", "adjacent to", "by", "near" |
| `in_front_of` | `in front of` | "in front of", "ahead of", "before" |
| `behind` | `behind` | "behind", "in back of" |
| `connected_to` | `connected to`, `attached to` | "connected to", "plugged into", "attached to", "wired to" |

---

## 4. IndexedDB Memory Stores Specification

Database Name: `workspace_retrieval_db` (Version 1)

1. `workspaces`: `[key: workspaceId]`
2. `sequences`: `[key: sequenceId, index: workspaceId]`
3. `frames`: `[key: frameId, index: sequenceId, timestamp]`
4. `objectInstances`: `[key: instanceId, index: workspaceId, categoryId]`
5. `objectObservations`: `[key: annotationId, index: frameId, instanceId]`
6. `relationships`: `[key: relationshipId, index: frameId, subjectId, objectId, canonicalPredicate]`
7. `graphEvents`: `[key: eventId, index: workspaceId, instanceId, timestamp]`
8. `vlmCandidates`: `[key: candidateId, index: frameId, status]`
9. `queryHistory`: `[key: queryId, index: timestamp]`
10. `appSettings`: `[key: settingKey]`
