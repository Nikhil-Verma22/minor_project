# SGG Benchmark — Relationship Vocabulary Alignment Audit

> [!IMPORTANT]
> **Analysis-Only Audit**: This report audits the existing candidate and reviewed relationship alignments across **VG150**, **PSG**, **IndoorVG**, and **Open Images V6**.
> In accordance with project instructions, **no existing alignment, vocabulary, annotation, or image files were modified** during this audit.

## 1. Executive Summary

This audit scrutinized the proposed four-dataset semantic relationship alignment against authoritative dataset annotation files, schema definitions, and relationship triplet instances.

### Key Findings:
1. **Critical Vocabulary Discrepancy in PSG Alignment**: The reviewed alignment file (`relationship_alignment_reviewed.json`) evaluated a **phantom 56-predicate list for PSG** that diverged from the authoritative test annotation file (`datasets/PSG/coco_format/test/_annotations.coco.json`) by **15 predicates**. Specifically, `under`, `wears`, `and`, `between`, and others were hallucinated into the PSG candidate list, while real PSG predicates (`painted on`, `lying on`, `parked on`, `crossing`, etc.) were omitted.
2. **`under` Must Be Excluded from 4-Dataset Intersection**: Because official PSG annotations **do not contain `under`** (PSG annotates `over`, but not `under`), `under` cannot be part of the four-dataset common vocabulary. It is supported across 3 datasets (VG150, IndoorVG, Open Images), but not all 4.
3. **Recomputed 4-Dataset Common Vocabularies**:
   - **Strict 4-Dataset Intersection**: **6 predicates** (`eating, hanging from, holding, in, on, wearing`).
   - **Extended 4-Dataset Intersection**: **8 predicates** (`eating, hanging from, holding, in, looking at, on, playing, wearing`).
4. **Attire Discrepancy (`wearing` / `wears`) Resolved**: VG150 contains two redundant entries (`wearing` ID 48 and `wears` ID 49). PSG and IndoorVG only use `wearing`. Open Images only uses `wears`. By establishing `wearing` (ID 48) as the normalized canonical target, attire is universally supported across all 4 datasets with zero directionality or semantic distortion.
5. **Open Images Directionality Confirmed**: Directionality was verified using 1,466 ground-truth triplet schemas. `inside_of`, `hang`, `holds`, `eat`, `ride`, and `wears` strictly preserve subject/object orientation. `contain` must map to `has` (never `in`) to prevent relation inversion.

---
## 2. Validation Results (Audit 1)

| Metric | VG150 | PSG | IndoorVG | Open Images V6 |
|---|:---:|:---:|:---:|:---:|
| Authoritative Vocabulary Size | 50 | 56 | 37 | 31 |
| Mappings Evaluated in Reviewed File | — | 56 | 37 | 31 |
| Duplicate Source Predicates | 0 | 0 | 0 | 0 |
| Phantom Predicates (Not in True Vocab) | 0 | **15** | 0 | 0 |
| Omitted Real Predicates | 0 | **15** | 0 | 0 |
| Invalid Canonical Target Names | 0 | 0 | 0 | 0 |
| Canonical ID Inconsistencies | 0 | 0 | 0 | 0 |

### Detailed PSG Discrepancy Breakdown:
- **15 Phantom Predicates in Reviewed PSG List**: `and`, `between`, `hugging`, `jumping on`, `jumping to`, `pointing to`, `sitting behind`, `standing behind`, `stopping on`, `teaching`, `under`, `waiting for`, `walking with`, `watching`, `wears`.
- **15 Real Predicates Omitted from Reviewed PSG List**: `about to hit`, `biting`, `crossing`, `driving on`, `flying over`, `guiding`, `jumping from`, `jumping over`, `kicking`, `lying on`, `opening`, `painted on`, `parked on`, `picking`, `slicing`.
- **Impact**: Real PSG predicates `painted on`, `lying on`, and `parked on` are **exact lexical and semantic matches** with VG150 that were previously overlooked. Conversely, `under` was erroneously claimed to exist in PSG.

---
## 3. Nine-Predicate Proposed Intersection Audit (Audit 2)

| Canonical VG150 | VG150 Source | PSG Source | IndoorVG Source | Open Images Source | Mapping Statuses (PSG / Indoor / OI) | Directionality | Semantic Justification & Information Lost | Verdict |
|---|---|---|---|---|---|:---:|---|:---:|
| `eating` | `eating (ID 14)` | `eating (ID 27 in true PSG)` | `eating (ID 11)` | `eat (identifier 'eat')` | PSG: EXACT<br>Ind: EXACT<br>OI: SEMANTIC_EQUIVALENT | **PRESERVED (Subject: Ingestor -> Object: Food across all 4 datasets)** | **Justification**: Oral food ingestion; only grammatical inflection differs between Open Images base verb 'eat' and gerund 'eating'.<br>**Lost**: None. | **RETAIN in both Strict and Extended common vocabulary.** |
| `hanging from` | `hanging from (ID 19)` | `hanging from (ID 7 in true PSG)` | `hanging from (ID 14)` | `hang (identifier 'hang')` | PSG: EXACT<br>Ind: EXACT<br>OI: SEMANTIC_EQUIVALENT | **PRESERVED (Subject: Suspended item -> Object: Anchor/Support across all 4 datasets; e.g. fruit hang tree)** | **Justification**: Physical suspension of subject from an anchor/support. Only grammatical prepositional variation between 'hang' and 'hanging from'.<br>**Lost**: None. | **RETAIN in both Strict and Extended common vocabulary.** |
| `holding` | `holding (ID 21)` | `holding (ID 22 in true PSG)` | `holding (ID 16)` | `holds (identifier 'holds') [also holding_hands ~]` | PSG: EXACT<br>Ind: EXACT<br>OI: SEMANTIC_EQUIVALENT (holds), ACCEPTABLE_SEMANTIC_GENERALIZATION (holding_hands) | **PRESERVED (Subject: Grasper -> Object: Grasped object in 'holds'; mutual in 'holding_hands')** | **Justification**: Physical manual grasp/support of an object. 'holds' is 3rd person singular present vs gerund 'holding'.<br>**Lost**: In 'holding_hands', interpersonal mutual reciprocity is abstracted into physical grasp. | **RETAIN in both Strict and Extended common vocabulary (via primary mapping 'holds' -> 'holding').** |
| `in` | `in (ID 22)` | `in (ID 5 in true PSG)` | `in (ID 17)` | `inside_of (identifier 'inside_of')` | PSG: EXACT<br>Ind: EXACT<br>OI: SEMANTIC_EQUIVALENT | **PRESERVED (Subject: Contained entity -> Object: Enclosing boundary/container across all 4 datasets; e.g. Man inside_of Car)** | **Justification**: Topological 2D/3D interior boundary containment. 'inside_of' and 'in' are complete semantic equivalents.<br>**Lost**: None. | **RETAIN in both Strict and Extended common vocabulary.** |
| `looking at` | `looking at (ID 25)` | `looking at (ID 24 in true PSG)` | `looking at (ID 20) [also reading ~]` | `read (identifier 'read')` | PSG: EXACT<br>Ind: EXACT (looking at), ACCEPTABLE_SEMANTIC_GENERALIZATION (reading)<br>OI: ACCEPTABLE_SEMANTIC_GENERALIZATION | **PRESERVED (Subject: Observer -> Object: Target of gaze / Reading material across all 4 datasets)** | **Justification**: Directing visual gaze at reading material is the primary visual relational act. In image-based scene graphs, reading is an observational specialization of looking at.<br>**Lost**: Specific cognitive textual comprehension is abstracted to visual gaze engagement. | **RETAIN in Extended common vocabulary; EXCLUDE from Strict common vocabulary.** |
| `on` | `on (ID 31)` | `on (ID 4 in true PSG)` | `on (ID 25)` | `on (identifier 'on')` | PSG: EXACT<br>Ind: EXACT<br>OI: EXACT | **PRESERVED (Subject: Supported item -> Object: Supporting surface across all 4 datasets)** | **Justification**: Identical surface contact and support relationship across all 4 datasets.<br>**Lost**: None. | **RETAIN in both Strict and Extended common vocabulary.** |
| `playing` | `playing (ID 37)` | `playing (ID 37 in true PSG) [also playing with ~]` | `playing with (ID 26)` | `plays (identifier 'plays')` | PSG: EXACT (playing), ACCEPTABLE_SEMANTIC_GENERALIZATION (playing with)<br>Ind: ACCEPTABLE_SEMANTIC_GENERALIZATION<br>OI: SEMANTIC_EQUIVALENT | **PRESERVED (Subject: Player -> Object: Game/instrument/toy across all 4 datasets)** | **Justification**: Recreational game/sport or musical instrument performance. 'plays' is inflectional equivalent. IndoorVG lacks base 'playing' but annotates 'playing with'.<br>**Lost**: In IndoorVG, specific toy/object accompaniment is generalized to broader play engagement. | **RETAIN in Extended common vocabulary; EXCLUDE from Strict common vocabulary.** |
| `under` | `under (ID 43)` | `MISSING in true PSG (PSG lacks 'under'; phantom in reviewed alignment)` | `under (ID 33)` | `under (identifier 'under')` | PSG: NO_MATCH (Absent from true PSG vocabulary)<br>Ind: EXACT<br>OI: EXACT | **PRESERVED in VG150, IndoorVG, and Open Images (Subject: Lower entity -> Object: Upper entity)** | **Justification**: Identical spatial relation beneath/underneath in VG150, IndoorVG, and Open Images. However, PSG does not annotate 'under' (only 'over').<br>**Lost**: N/A for PSG (predicate does not exist). | **REVISE: MUST BE EXCLUDED from the 4-dataset intersection because PSG lacks 'under'. Retained as a valid 3-dataset common predicate.** |
| `wearing` | `wearing (ID 48) [also duplicate wears ID 49]` | `wearing (ID 21 in true PSG)` | `wearing (ID 36)` | `wears (identifier 'wears')` | PSG: EXACT<br>Ind: EXACT<br>OI: SEMANTIC_EQUIVALENT (to 'wearing'), EXACT (to duplicate 'wears') | **PRESERVED (Subject: Person/Wearer -> Object: Clothing/Accessory across all 4 datasets)** | **Justification**: Physical attire/adornment relationship. 'wears' in Open Images is purely an inflectional variant of 'wearing'.<br>**Lost**: None. | **RETAIN in both Strict and Extended common vocabulary under normalized canonical label 'wearing'.** |

---
## 4. `wearing` / `wears` Discrepancy Investigation (Audit 3)

### Empirical Findings from Source Annotations:
1. **VG150 Contains Both Labels**:
   - `wearing` (Predicate ID 48): 136,099 total dataset instances; 20,148 test set annotations.
   - `wears` (Predicate ID 49): 15,457 total dataset instances; 2,324 test set annotations.
   - **Semantic Relationship**: Pure unlemmatized crowd-sourced duplicates. Both connect `Person -> Clothing/Accessory` with identical semantics and direction.
2. **PSG Uses Only `wearing`**: `wearing` is Category ID 21. `wears` does not exist in PSG.
3. **IndoorVG Uses Only `wearing`**: `wearing` is Category ID 36. `wears` does not exist in IndoorVG.
4. **Open Images V6 Uses Only `wears`**: Relationship identifier is `wears` (101 triplet schemas, e.g. `Man -> wears -> Glasses`). `wearing` does not exist in Open Images.

### Evaluation Recommendations:
- **Do NOT modify original dataset files**: VG150 vocabulary files must preserve both ID 48 and ID 49.
- **Evaluation Normalization**: Designate `wearing` (ID 48) as the single normalized canonical attire predicate.
- **Normalization Rule**: Map both VG150 `wears` (ID 49) and Open Images `wears` to normalized `wearing` (ID 48) as `SEMANTIC_EQUIVALENT`.
- **Consequence**: This promotes attire into both the Strict and Extended 4-way common vocabularies without loss of meaning.

---
## 5. Open Images V6 Semantic & Directionality Audit (Audit 4)

| Predicate | Proposed VG150 Target | Triplet Count | Sample Triplet (OIDv6 Ground Truth) | Directionality & Role Check | Semantic Scope Shift | Evidence & Verdict |
|---|---|:---:|---|---|---|---|
| `inside_of` | `in (ID 22)` | 48 | Man -> inside_of -> Car<br>Dog -> inside_of -> Car<br>Girl -> inside_of -> Bus | **Subject: Contained entity; Object: Enclosing vehicle/boundary. Roles are strictly preserved (same as VG150 'Man in Car').** | None. 3D interior enclosure is preserved. | ACCEPT as SEMANTIC_EQUIVALENT. |
| `contain` | `has (ID 20)` | 124 | Coffee cup -> contain -> Coffee<br>Platter -> contain -> Tart<br>Wine glass -> contain -> Cocktail | **Subject: Container; Object: Contained substance/item. Mapping to 'in' ('Coffee in cup') would INVERT subject and object roles (DIRECTIONALITY CONFLICT). Mapping to 'has' ('Cup has Coffee') preserves the Subject->Object orientation.** | Broadens from physical volumetric containment to general possession/meronymy. The container is affirmed to hold the content. | ACCEPT as ACCEPTABLE_SEMANTIC_GENERALIZATION to 'has' only; strictly reject mapping to 'in' due to directionality conflict. |
| `hang` | `hanging from (ID 19)` | 24 | Pear -> hang -> Tree<br>Grape -> hang -> Tree<br>Grapefruit -> hang -> Houseplant | **Subject: Suspended item; Object: Anchor/support. Roles are strictly preserved (identical to VG150 'Pear hanging from Tree').** | None. Pure grammatical prepositional variation. | ACCEPT as SEMANTIC_EQUIVALENT. |
| `holding_hands` | `holding (ID 21)` | 12 | Man -> holding_hands -> Man<br>Girl -> holding_hands -> Girl<br>Man -> holding_hands -> Boy | **Subject: Person; Object: Person. In VG150, 'holding' predominantly connects Person to an object/item. Here it connects Person to Person via mutual manual grasp.** | Loss of interpersonal mutual reciprocity. The relationship is generalized to physical manual grasp. | ACCEPT under Extended generalizations only; primary OI mapping to 'holding' is already provided by 'holds' (457 triplets). |
| `wears` | `wearing (ID 48)` | 101 | Man -> wears -> Glasses<br>Man -> wears -> Fedora<br>Boy -> wears -> Baseball glove | **Subject: Person; Object: Clothing/accessory. Roles are strictly preserved (identical to VG150 'Man wearing Glasses').** | None. Pure grammatical inflection (3rd person singular present vs participle). | ACCEPT as SEMANTIC_EQUIVALENT to 'wearing' (and EXACT to VG150 duplicate 'wears'). |
| `read` | `looking at (ID 25)` | 4 | Man -> read -> Book<br>Woman -> read -> Book<br>Girl -> read -> Book | **Subject: Person; Object: Book. Roles are strictly preserved.** | Generalizes cognitive textual reading to the required visual relational engagement ('looking at'). | ACCEPT under Extended generalizations only. |
| `plays` | `playing (ID 37)` | 65 | Man -> plays -> Violin<br>Man -> plays -> Piano<br>Woman -> plays -> Cello | **Subject: Person; Object: Musical instrument. Roles are strictly preserved (identical to VG150 'Man playing Violin').** | None. Pure grammatical inflection (3rd person singular present 'plays' vs participle 'playing'). | ACCEPT as SEMANTIC_EQUIVALENT. |
| `holds` | `holding (ID 21)` | 457 | Man -> holds -> Tennis racket<br>Woman -> holds -> Racket<br>Girl -> holds -> Racket | **Subject: Person; Object: Item/Tool. Roles are strictly preserved.** | None. Pure grammatical inflection (holds vs holding). | ACCEPT as SEMANTIC_EQUIVALENT. |
| `eat` | `eating (ID 14)` | 46 | Boy -> eat -> Watermelon<br>Woman -> eat -> Watermelon<br>Man -> eat -> Pizza | **Subject: Person; Object: Food. Roles are strictly preserved.** | None. Pure grammatical inflection (eat vs eating). | ACCEPT as SEMANTIC_EQUIVALENT. |
| `ride` | `riding (ID 38)` | 76 | Man -> ride -> Horse<br>Woman -> ride -> Cart<br>Man -> ride -> Canoe | **Subject: Person; Object: Mount/vehicle. Roles are strictly preserved.** | None. Pure grammatical inflection (ride vs riding). | ACCEPT as SEMANTIC_EQUIVALENT. |
| `at` | `at (ID 6)` | 13 | Man -> at -> Table<br>Woman -> at -> Table | **Subject: Person; Object: Furniture/location. Roles are strictly preserved.** | None. | ACCEPT as EXACT. |
| `on` | `on (ID 31)` | 272 | Cat -> on -> Bed<br>Bowl -> on -> Table | **Subject: Supported item; Object: Supporting surface. Roles are strictly preserved.** | None. | ACCEPT as EXACT. |
| `under` | `under (ID 43)` | 42 | Dog -> under -> Table<br>Cat -> under -> Car | **Subject: Lower item; Object: Upper structure. Roles are strictly preserved.** | None. | ACCEPT as EXACT. |

### Critical Directionality Warning on `contain`:
- In Open Images, `contain` strictly connects **Container -> Contents** (e.g. `Coffee cup -> contain -> Tea`).
- VG150 `in` strictly connects **Contents -> Container** (e.g. `Tea in Coffee cup`).
- **Mapping `contain` to `in` reverses the relationship direction (Subject/Object swap)** and is strictly invalid.
- Mapping `contain` to `has` (`Coffee cup has Tea`) preserves directionality and is accepted as an Extended generalization.

---
## 6. Additional Candidates to Expand Common Vocabulary

While no additional predicate can achieve unanimous 4-dataset coverage beyond the 8 extended predicates, several high-value candidates achieve **robust 3-dataset coverage**:
1. **`under` (3 Datasets: VG150, IndoorVG, Open Images)**: Exact lexical match in all three. Excluded only from PSG.
2. **`at` (3 Datasets: VG150, IndoorVG, Open Images)**: Exact spatial collocation in all three. Excluded only from PSG.
3. **`riding` (3 Datasets: VG150, PSG, Open Images)**: Exact active transit locomotion in VG150 (`riding`), PSG (`riding`), and Open Images (`ride`). Excluded only from IndoorVG.
4. **`has` (3 Datasets: VG150, IndoorVG, Open Images)**: Exact in VG150/IndoorVG; Open Images `contain` generalizable to `has`. Excluded only from PSG.
5. **`attached to`, `carrying`, `in front of`, `sitting on`, `standing on`, `lying on`, `near`, `against`, `over` (3 Datasets: VG150, PSG, IndoorVG)**: Supported across all three benchmark subsets; excluded only from Open Images.

---
## 7. Recomputed Vocabulary Coverage (Audit 5)

### Four-Dataset Common Intersections:
- **A. Strict 4-Dataset Common Vocabulary**: **6 predicates**
  > `eating, hanging from, holding, in, on, wearing`
  *(Criteria: Strictly EXACT or proven SEMANTIC_EQUIVALENT across all 4 datasets).*

- **B. Extended 4-Dataset Common Vocabulary**: **8 predicates**
  > `eating, hanging from, holding, in, looking at, on, playing, wearing`
  *(Criteria: EXACT, SEMANTIC_EQUIVALENT, or ACCEPTABLE_SEMANTIC_GENERALIZATION across all 4 datasets).*

### Multi-Dataset Support Breakdown:
- **Tri-Dataset Common Vocabulary (Supported by >= 3 datasets)**: **20 predicates**
  > `against, at, attached to, carrying, eating, hanging from, has, holding, in, in front of, looking at, lying on, near, on, playing, riding, sitting on, standing on, under, wearing`
- **Union Coverage (Supported by VG150 and at least 1 other dataset)**: **36 / 50 VG150 predicates (72.0%)**

### Per-Dataset Mapping Statistics:
| Dataset | Total Predicates | Defensible Mapped | Unmapped / Rejected | EXACT | SEMANTIC_EQUIVALENT | ACCEPTABLE_GENERALIZATION | REJECTED / NO_MATCH |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PSG (True Vocab)** | 56 | 24 | 32 | 20 | 1 | 3 | 32 |
| **IndoorVG** | 37 | 32 | 5 | 29 | 0 | 3 | 5 |
| **Open Images V6** | 31 | 13 | 18 | 3 | 7 | 3 | 18 |

---
## 8. Recommended Corrections

1. **Update PSG Vocabulary in Alignment Files**: Correct the 15 phantom PSG predicates in `relationship_alignment_candidates.json` and `relationship_alignment_reviewed.json` to reflect the official 56 categories in `datasets/PSG/coco_format/test/_annotations.coco.json`.
2. **Revise Extended 4-Way Vocabulary from 9 to 8**: Formally document that `under` is a 3-way predicate (VG150 + IndoorVG + Open Images), leaving 8 unanimously supported predicates in the 4-way extended intersection.
3. **Adopt Normalized `wearing` Label**: Formally designate `wearing` (ID 48) as the canonical attire evaluation label, documenting that VG150 `wears` (ID 49) and Open Images `wears` map into it as semantic equivalents.
4. **Document Directionality Preservation on Container Relations**: Explicitly mandate that Open Images `contain` maps only to `has` under generalization, preventing automated reverse mapping to `in`.

---
## 9. Unresolved Questions & Confidence Levels

- **`holding_hands` Generalization (Confidence: MEDIUM)**: Does treating person-to-person hand-holding as general `holding` alter evaluation for interpersonal actions? Recommended: Accept under Extended generalizations, but evaluate model sensitivity during inference.
- **3-Way vs. 4-Way Benchmark Structuring (Confidence: HIGH)**: Since Open Images has a smaller predicate set (31 predicates, primarily human-centric actions) while VG150, PSG, and IndoorVG share extensive physical/spatial relations, we strongly recommend evaluating models on **both** the 4-way core (8 predicates) and the 3-way core (23 predicates).

---
## 10. Explicit Preservation Confirmation

In accordance with project instructions, **no modifications were made to any existing files**:
- `evaluation_sets/VG150_200.txt` (Untouched, exactly 200 IDs)
- `evaluation_sets/VG150_200_metadata.json` (Untouched)
- `evaluation_sets/VG150_200_annotations.coco.json` (Untouched)
- `evaluation_sets/PSG_200.txt` (Untouched, exactly 200 IDs)
- `evaluation_sets/PSG_200_metadata.json` (Untouched)
- `evaluation_sets/PSG_200_annotations.coco.json` (Untouched)
- `evaluation_sets/images/VG150/` (Untouched, 200 image files)
- `evaluation_sets/images/PSG/` (Untouched, 200 image files)
- `evaluation_sets/relationship_vocabularies.*` (Untouched)
- `evaluation_sets/relationship_alignment_candidates.*` (Untouched)
- `evaluation_sets/relationship_alignment_reviewed.*` (Untouched)
- No new dataset test sets (`IndoorVG_200`, `OpenImages_200`) were created.
- No model inference, training, or code modification was performed.
