# SGG Benchmark — Relationship & Predicate Vocabularies

> [!NOTE]
> This report contains the exact, unmapped original relationship/predicate vocabularies extracted from authoritative metadata of all four Scene Graph Generation datasets.
> No cross-dataset semantic mapping or equivalence merging has been performed at this stage.

## Executive Summary

| Dataset | Role | Predicate Count | Authoritative Source |
|---|---|---|---|
| **VG150** | **Canonical Vocabulary** | **50** | `datasets/VG150/VG150_coco_format/test/_annotations.coco.json` / `datasets/vg/VG-SGG-dicts-with-attri.json` |
| **PSG** | Source to be aligned | **56** | `datasets/PSG/coco_format/test/_annotations.coco.json` (`maelic/PSG-coco-format`) |
| **IndoorVG** | Source to be aligned | **37** | `datasets/IndoorVG/IndoorVG_coco_format/categories.json` (`maelic/IndoorVG-coco-format`) |
| **Open Images V6** | Source to be aligned | **31** | `oidv6-relationships-description.csv` (`storage.googleapis.com/openimages/v6/`) |

---
## 1. VG150 (Canonical Vocabulary)
- **Role**: Canonical target ontology for semantic alignment.
- **Source File**: `datasets/VG150/VG150_coco_format/test/_annotations.coco.json` and `datasets/vg/VG-SGG-dicts-with-attri.json`.
- **Total Predicates**: **50** (1-indexed, IDs 1 to 50).
- **Important Notes**:
  - Predicates in Visual Genome 150 represent high-frequency relationships selected by Xu et al. (2017) and widely adopted as the de facto standard benchmark for SGG.
  - The predicate mapping in `VG-SGG-dicts-with-attri.json` (`predicate_to_idx` / `idx_to_predicate`) is 100% identical to `rel_categories` in the COCO format annotation file.

| Predicate ID | Predicate Name |
|---|---|
| 1 | `above` |
| 2 | `across` |
| 3 | `against` |
| 4 | `along` |
| 5 | `and` |
| 6 | `at` |
| 7 | `attached to` |
| 8 | `behind` |
| 9 | `belonging to` |
| 10 | `between` |
| 11 | `carrying` |
| 12 | `covered in` |
| 13 | `covering` |
| 14 | `eating` |
| 15 | `flying in` |
| 16 | `for` |
| 17 | `from` |
| 18 | `growing on` |
| 19 | `hanging from` |
| 20 | `has` |
| 21 | `holding` |
| 22 | `in` |
| 23 | `in front of` |
| 24 | `laying on` |
| 25 | `looking at` |
| 26 | `lying on` |
| 27 | `made of` |
| 28 | `mounted on` |
| 29 | `near` |
| 30 | `of` |
| 31 | `on` |
| 32 | `on back of` |
| 33 | `over` |
| 34 | `painted on` |
| 35 | `parked on` |
| 36 | `part of` |
| 37 | `playing` |
| 38 | `riding` |
| 39 | `says` |
| 40 | `sitting on` |
| 41 | `standing on` |
| 42 | `to` |
| 43 | `under` |
| 44 | `using` |
| 45 | `walking in` |
| 46 | `walking on` |
| 47 | `watching` |
| 48 | `wearing` |
| 49 | `wears` |
| 50 | `with` |

---
## 2. PSG (Panoptic Scene Graph)
- **Role**: Source vocabulary to be aligned into VG150.
- **Source File**: `datasets/PSG/coco_format/test/_annotations.coco.json` (upstream: `maelic/PSG-coco-format`).
- **Total Predicates**: **56** (1-indexed, IDs 1 to 56).
- **Important Notes**:
  - Originates from the Panoptic Scene Graph benchmark (Yang et al., NeurIPS 2022).
  - PSG defines relationships connecting 'things' and 'stuff' (e.g., spatial relations like `beside`, `over`, `leaning on`, as well as actions like `playing with`, `riding`).
  - All 56 predicate categories are active and defined in the COCO format metadata.

| Predicate ID | Predicate Name |
|---|---|
| 1 | `over` |
| 2 | `in front of` |
| 3 | `beside` |
| 4 | `on` |
| 5 | `in` |
| 6 | `attached to` |
| 7 | `hanging from` |
| 8 | `on back of` |
| 9 | `falling off` |
| 10 | `going down` |
| 11 | `painted on` |
| 12 | `walking on` |
| 13 | `running on` |
| 14 | `crossing` |
| 15 | `standing on` |
| 16 | `lying on` |
| 17 | `sitting on` |
| 18 | `flying over` |
| 19 | `jumping over` |
| 20 | `jumping from` |
| 21 | `wearing` |
| 22 | `holding` |
| 23 | `carrying` |
| 24 | `looking at` |
| 25 | `guiding` |
| 26 | `kissing` |
| 27 | `eating` |
| 28 | `drinking` |
| 29 | `feeding` |
| 30 | `biting` |
| 31 | `catching` |
| 32 | `picking` |
| 33 | `playing with` |
| 34 | `chasing` |
| 35 | `climbing` |
| 36 | `cleaning` |
| 37 | `playing` |
| 38 | `touching` |
| 39 | `pushing` |
| 40 | `pulling` |
| 41 | `opening` |
| 42 | `cooking` |
| 43 | `talking to` |
| 44 | `throwing` |
| 45 | `slicing` |
| 46 | `driving` |
| 47 | `riding` |
| 48 | `parked on` |
| 49 | `driving on` |
| 50 | `about to hit` |
| 51 | `kicking` |
| 52 | `swinging` |
| 53 | `entering` |
| 54 | `exiting` |
| 55 | `enclosing` |
| 56 | `leaning on` |

---
## 3. IndoorVG
- **Role**: Source vocabulary to be aligned into VG150.
- **Source File**: `datasets/IndoorVG/IndoorVG_coco_format/categories.json` (upstream: `maelic/IndoorVG-coco-format/raw/main/categories.json`, Neau et al., BMVC 2025).
- **Total Predicates**: **37** (1-indexed, IDs 1 to 37).
- **Important Notes**:
  - Curated indoor subset of Visual Genome proposed by Neau et al. (2024, BMVC 2025).
  - The 37 predicates are a curated, disambiguated subset derived directly from the broader Visual Genome predicate distribution, specifically tailored for indoor scenes (rooms, furniture, domestic interactions).
  - Because IndoorVG is an indoor-focused derivation of Visual Genome, many predicates share exact lexical matches with VG150.

| Predicate ID | Predicate Name |
|---|---|
| 1 | `above` |
| 2 | `against` |
| 3 | `at` |
| 4 | `attached to` |
| 5 | `behind` |
| 6 | `between` |
| 7 | `carrying` |
| 8 | `covering` |
| 9 | `cutting` |
| 10 | `drinking` |
| 11 | `eating` |
| 12 | `filled with` |
| 13 | `for` |
| 14 | `hanging from` |
| 15 | `has` |
| 16 | `holding` |
| 17 | `in` |
| 18 | `in front of` |
| 19 | `laying on` |
| 20 | `looking at` |
| 21 | `lying on` |
| 22 | `mounted on` |
| 23 | `near` |
| 24 | `of` |
| 25 | `on` |
| 26 | `playing with` |
| 27 | `reading` |
| 28 | `sitting at` |
| 29 | `sitting on` |
| 30 | `standing on` |
| 31 | `taking` |
| 32 | `talking on` |
| 33 | `under` |
| 34 | `using` |
| 35 | `watching` |
| 36 | `wearing` |
| 37 | `with` |

---
## 4. Open Images V6
- **Role**: Source vocabulary to be aligned into VG150.
- **Source File**: `oidv6-relationships-description.csv` (official Google Open Images V6 metadata: `https://storage.googleapis.com/openimages/v6/oidv6-relationships-description.csv`).
- **Total Predicates**: **31** unique relationship types.
- **Crucial Distinctions**:
  - **Predicate Vocabulary (31)**: The 31 distinct relation types connecting subjects to objects/attributes (e.g., `at`, `holds`, `wears`, `inside_of`, `interacts_with`, `is`).
  - **Relationship Triplets (1,466)**: Documented in `oidv6-relationship-triplets.csv`. These are the 1,466 unique `(LabelName1, LabelName2, RelationshipLabel)` combinations annotated in the dataset.
  - **Attribute Classes (15)**: Documented in `oidv6-attributes-description.csv` (e.g., `Wooden`, `Transparent`, `Stand`, `Walk`), which appear as `LabelName2` when `RelationshipLabel == 'is'`.
  - **Object Classes (600)**: Documented in `class-descriptions-boxable.csv`.

| # | Relationship Identifier | Human-Readable Name |
|---|---|---|
| 1 | `at` | at |
| 2 | `holds` | holds |
| 3 | `wears` | wears |
| 4 | `surf` | surf |
| 5 | `hang` | hang |
| 6 | `drink` | drink |
| 7 | `holding_hands` | holding hands |
| 8 | `on` | on |
| 9 | `ride` | ride |
| 10 | `dance` | dance |
| 11 | `skateboard` | skateboard |
| 12 | `catch` | catch |
| 13 | `highfive` | highfive |
| 14 | `inside_of` | inside of |
| 15 | `eat` | eat |
| 16 | `cut` | cut |
| 17 | `contain` | contain |
| 18 | `handshake` | handshake |
| 19 | `kiss` | kiss |
| 20 | `talk_on_phone` | talk on phone |
| 21 | `interacts_with` | interacts with |
| 22 | `under` | under |
| 23 | `hug` | hug |
| 24 | `throw` | throw |
| 25 | `hits` | hits |
| 26 | `snowboard` | snowboard |
| 27 | `is` | is |
| 28 | `kick` | kick |
| 29 | `ski` | ski |
| 30 | `plays` | plays |
| 31 | `read` | read |

