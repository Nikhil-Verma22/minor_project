# SGG Benchmark — Finalized Relationship Alignment & Evaluation Ontology

> [!IMPORTANT]
> **FROZEN EVALUATION ONTOLOGY**: This document formally freezes the relationship alignment for cross-dataset evaluation of pretrained Scene Graph Generation models (REACT, REACT++, FCSGG, and Fast Contextual SGG).
> It incorporates the audited authoritative PSG vocabulary from `_annotations.coco.json` and establishes universal attire normalization under canonical `wearing`.

## 1. Frozen Common Evaluation Vocabularies

### A. Strict Four-Dataset Core (6 Predicates)
- **Size**: **6 predicates**
- **Composition**: `eating`, `hanging from`, `holding`, `in`, `on`, `wearing`
- **Criteria**: Strictly EXACT or proven SEMANTIC_EQUIVALENT across all four datasets without relying on broader generalization.

### B. Extended Four-Dataset Core (8 Predicates) — *Target Benchmark Vocabulary*
- **Size**: **8 predicates**
- **Composition**: `eating`, `hanging from`, `holding`, `in`, `looking at`, `on`, `playing`, `wearing`
- **Criteria**: EXACT, SEMANTIC_EQUIVALENT, or documented ACCEPTABLE_SEMANTIC_GENERALIZATION across all four datasets.
- *Explicit Exclusion*: `under` is strictly excluded from both four-dataset cores because it does not exist in authoritative PSG annotations.

### C. Tri-Dataset Common Vocabulary (20 Predicates)
- **Size**: **20 predicates**
- **Composition**: `against`, `at`, `attached to`, `carrying`, `eating`, `hanging from`, `has`, `holding`, `in`, `in front of`, `looking at`, `lying on`, `near`, `on`, `playing`, `riding`, `sitting on`, `standing on`, `under`, `wearing`
- **Support**: Supported across VG150 and at least two other datasets (17 predicates unanimously supported across VG150, PSG, and IndoorVG).

---
## 2. Three-Dataset Predicate Support Matrix

| Predicate | Canonical VG150 ID | Tri-Dataset Eval ID | VG150 | PSG | IndoorVG | Open Images V6 | Dataset Support Count | Support Assessment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `against` | 3 | 1 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `at` | 6 | 2 | ✓ | ✗ | ✓ | ✓ | **3 / 4** | VG150 + IndoorVG + OIDv6 (PSG lacks) |
| `attached to` | 7 | 3 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `carrying` | 11 | 4 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `eating` | 14 | 5 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `hanging from` | 19 | 6 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `has` | 20 | 7 | ✓ | ✗ | ✓ | ✓ | **3 / 4** | VG150 + IndoorVG + OIDv6 (PSG lacks) |
| `holding` | 21 | 8 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `in` | 22 | 9 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `in front of` | 23 | 10 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `looking at` | 25 | 11 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `lying on` | 26 | 12 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `near` | 29 | 13 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `on` | 31 | 14 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `playing` | 37 | 15 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `riding` | 38 | 16 | ✓ | ✓ | ✗ | ✓ | **3 / 4** | VG150 + PSG + OIDv6 (IndoorVG lacks) |
| `sitting on` | 40 | 17 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `standing on` | 41 | 18 | ✓ | ✓ | ✓ | ✗ | **3 / 4** | Unanimous VG150 + PSG + IndoorVG |
| `under` | 43 | 19 | ✓ | ✗ | ✓ | ✓ | **3 / 4** | VG150 + IndoorVG + OIDv6 (PSG lacks) |
| `wearing` | 48 | 20 | ✓ | ✓ | ✓ | ✓ | **4 / 4** | Unanimous VG150 + PSG + IndoorVG |

---
## 3. PSG Final Mappings (56 Authoritative Predicates)

| Source Predicate | Original ID | Normalized Canonical Target | Canonical ID | Mapping Type | Confidence | Justification & Rule |
|---|:---:|---|:---:|:---:|:---:|---|
| `over` | 1 | `over` | 33 | **EXACT** | HIGH | Exact lexical and spatial match (vertically above). |
| `in front of` | 2 | `in front of` | 23 | **EXACT** | HIGH | Exact lexical and anterior spatial match. |
| `beside` | 3 | `near` | 29 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Lateral adjacency ('beside') is a specific topological instantiation of general spatial proximity ('near'). |
| `on` | 4 | `on` | 31 | **EXACT** | HIGH | Exact physical surface contact and support match. |
| `in` | 5 | `in` | 22 | **EXACT** | HIGH | Exact interior boundary containment match. |
| `attached to` | 6 | `attached to` | 7 | **EXACT** | HIGH | Exact physical/mechanical attachment match. |
| `hanging from` | 7 | `hanging from` | 19 | **EXACT** | HIGH | Exact suspension from support anchor match. |
| `on back of` | 8 | `on back of` | 32 | **EXACT** | HIGH | Exact posterior surface support match. |
| `falling off` | 9 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'falling off' absent in canonical VG150. |
| `going down` | 10 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'going down' absent in canonical VG150. |
| `painted on` | 11 | `painted on` | 34 | **EXACT** | HIGH | Exact surface graphic rendering match. |
| `walking on` | 12 | `walking on` | 46 | **EXACT** | HIGH | Exact pedestrian locomotion on surface match. |
| `running on` | 13 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'running on' absent in canonical VG150. |
| `crossing` | 14 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'crossing' absent in canonical VG150. |
| `standing on` | 15 | `standing on` | 41 | **EXACT** | HIGH | Exact upright bodily posture supported on surface match. |
| `lying on` | 16 | `lying on` | 26 | **EXACT** | HIGH | Exact recumbent bodily posture upon surface match. |
| `sitting on` | 17 | `sitting on` | 40 | **EXACT** | HIGH | Exact seated bodily posture supported on surface match. |
| `flying over` | 18 | `over` | 33 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Airborne locomotion spatially elevated over target; core spatial relation is 'over'. |
| `jumping over` | 19 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'jumping over' absent in canonical VG150. |
| `jumping from` | 20 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'jumping from' absent in canonical VG150. |
| `wearing` | 21 | `wearing` | 48 | **EXACT** | HIGH | Exact attire/garment adornment match. |
| `holding` | 22 | `holding` | 21 | **EXACT** | HIGH | Exact manual physical grasp match. |
| `carrying` | 23 | `carrying` | 11 | **EXACT** | HIGH | Exact bearing and transporting load match. |
| `looking at` | 24 | `looking at` | 25 | **EXACT** | HIGH | Exact directed visual gaze match. |
| `guiding` | 25 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'guiding' absent in canonical VG150. |
| `kissing` | 26 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'kissing' absent in canonical VG150. |
| `eating` | 27 | `eating` | 14 | **EXACT** | HIGH | Exact oral food consumption match. |
| `drinking` | 28 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'drinking' absent in canonical VG150. |
| `feeding` | 29 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'feeding' absent in canonical VG150. |
| `biting` | 30 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'biting' absent in canonical VG150. |
| `catching` | 31 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'catching' absent in canonical VG150. |
| `picking` | 32 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'picking' absent in canonical VG150. |
| `playing with` | 33 | `playing` | 37 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Recreational engagement with toy/object or peer; core interaction is 'playing'. |
| `chasing` | 34 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'chasing' absent in canonical VG150. |
| `climbing` | 35 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'climbing' absent in canonical VG150. |
| `cleaning` | 36 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'cleaning' absent in canonical VG150. |
| `playing` | 37 | `playing` | 37 | **EXACT** | HIGH | Exact recreational sport/game engagement match. |
| `touching` | 38 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'touching' absent in canonical VG150. |
| `pushing` | 39 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'pushing' absent in canonical VG150. |
| `pulling` | 40 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'pulling' absent in canonical VG150. |
| `opening` | 41 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'opening' absent in canonical VG150. |
| `cooking` | 42 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'cooking' absent in canonical VG150. |
| `talking to` | 43 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'talking to' absent in canonical VG150. |
| `throwing` | 44 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'throwing' absent in canonical VG150. |
| `slicing` | 45 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'slicing' absent in canonical VG150. |
| `driving` | 46 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'driving' absent in canonical VG150. |
| `riding` | 47 | `riding` | 38 | **EXACT** | HIGH | Exact vehicle/mount transit operation match. |
| `parked on` | 48 | `parked on` | 35 | **EXACT** | HIGH | Exact stationary vehicle resting on surface match. |
| `driving on` | 49 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'driving on' absent in canonical VG150. |
| `about to hit` | 50 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'about to hit' absent in canonical VG150. |
| `kicking` | 51 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'kicking' absent in canonical VG150. |
| `swinging` | 52 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'swinging' absent in canonical VG150. |
| `entering` | 53 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'entering' absent in canonical VG150. |
| `exiting` | 54 | *Unmapped* | — | **NO_MATCH** | HIGH | Domain/action-specific predicate 'exiting' absent in canonical VG150. |
| `enclosing` | 55 | *Unmapped* | — | **AMBIGUOUS_REJECTED** | HIGH | 3D boundary framing/containment differs topologically from 2D surface covering. |
| `leaning on` | 56 | `against` | 3 | **SEMANTIC_EQUIVALENT** | HIGH | Physical inclined contact resting against an upright support. |

---
## 4. IndoorVG Final Mappings (37 Predicates)

| Source Predicate | Original ID | Normalized Canonical Target | Canonical ID | Mapping Type | Confidence | Justification & Rule |
|---|:---:|---|:---:|:---:|:---:|---|
| `above` | 1 | `above` | 1 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `against` | 2 | `against` | 3 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `at` | 3 | `at` | 6 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `attached to` | 4 | `attached to` | 7 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `behind` | 5 | `behind` | 8 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `between` | 6 | `between` | 10 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `carrying` | 7 | `carrying` | 11 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `covering` | 8 | `covering` | 13 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `cutting` | 9 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific action predicate 'cutting' absent in canonical VG150. |
| `drinking` | 10 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific action predicate 'drinking' absent in canonical VG150. |
| `eating` | 11 | `eating` | 14 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `filled with` | 12 | *Unmapped* | — | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'in' causes DIRECTIONALITY_CONFLICT; mapping to 'has' loses volumetric containment. |
| `for` | 13 | `for` | 16 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `hanging from` | 14 | `hanging from` | 19 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `has` | 15 | `has` | 20 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `holding` | 16 | `holding` | 21 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `in` | 17 | `in` | 22 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `in front of` | 18 | `in front of` | 23 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `laying on` | 19 | `laying on` | 24 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `looking at` | 20 | `looking at` | 25 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `lying on` | 21 | `lying on` | 26 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `mounted on` | 22 | `mounted on` | 28 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `near` | 23 | `near` | 29 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `of` | 24 | `of` | 30 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `on` | 25 | `on` | 31 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `playing with` | 26 | `playing` | 37 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Recreational play interaction with toy/peer; core interaction is 'playing'. |
| `reading` | 27 | `looking at` | 25 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Observational engagement with reading material; core visual relation is 'looking at'. |
| `sitting at` | 28 | `at` | 6 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Spatial collocation relative to desk/table without asserting physical support ON desk. |
| `sitting on` | 29 | `sitting on` | 40 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `standing on` | 30 | `standing on` | 41 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `taking` | 31 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific action predicate 'taking' absent in canonical VG150. |
| `talking on` | 32 | *Unmapped* | — | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'using' is overly broad and discards conversational telecommunication semantic. |
| `under` | 33 | `under` | 43 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `using` | 34 | `using` | 44 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `watching` | 35 | `watching` | 47 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `wearing` | 36 | `wearing` | 48 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |
| `with` | 37 | `with` | 50 | **EXACT** | HIGH | Exact lexical and semantic match from Visual Genome indoor curation. |

---
## 5. Open Images V6 Final Mappings (31 Predicates)

| Source Predicate | Original ID | Normalized Canonical Target | Canonical ID | Mapping Type | Confidence | Justification & Rule |
|---|:---:|---|:---:|:---:|:---:|---|
| `at` | 1 | `at` | 6 | **EXACT** | HIGH | Exact spatial collocation match. |
| `holds` | 2 | `holding` | 21 | **SEMANTIC_EQUIVALENT** | HIGH | Physical manual grasp; 3rd person singular present inflection of 'holding'. |
| `wears` | 3 | `wearing` | 48 | **SEMANTIC_EQUIVALENT** | HIGH | Attire/garment adornment; inflectional equivalent of normalized canonical 'wearing' (ID 48). |
| `surf` | 4 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'surf' absent in canonical VG150. |
| `hang` | 5 | `hanging from` | 19 | **SEMANTIC_EQUIVALENT** | HIGH | Physical suspension from anchor; prepositional variation of 'hanging from'. |
| `drink` | 6 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'drink' absent in canonical VG150. |
| `holding_hands` | 7 | `holding` | 21 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | MEDIUM | Mutual manual grasp between persons; generalized to manual contact 'holding'. |
| `on` | 8 | `on` | 31 | **EXACT** | HIGH | Exact physical surface contact and support match. |
| `ride` | 9 | `riding` | 38 | **SEMANTIC_EQUIVALENT** | HIGH | Active locomotion/transit operating vehicle or mount; base form of 'riding'. |
| `dance` | 10 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'dance' absent in canonical VG150. |
| `skateboard` | 11 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'skateboard' absent in canonical VG150. |
| `catch` | 12 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'catch' absent in canonical VG150. |
| `highfive` | 13 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'highfive' absent in canonical VG150. |
| `inside_of` | 14 | `in` | 22 | **SEMANTIC_EQUIVALENT** | HIGH | Interior spatial containment; full semantic equivalent of 'in'. |
| `eat` | 15 | `eating` | 14 | **SEMANTIC_EQUIVALENT** | HIGH | Oral food ingestion; base form inflection of 'eating'. |
| `cut` | 16 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'cut' absent in canonical VG150. |
| `contain` | 17 | `has` | 20 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Container possesses/includes contents (Container -> Contents). Preserves directionality. |
| `handshake` | 18 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'handshake' absent in canonical VG150. |
| `kiss` | 19 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'kiss' absent in canonical VG150. |
| `talk_on_phone` | 20 | *Unmapped* | — | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'using' loses verbal telecommunication semantic. |
| `interacts_with` | 21 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'interacts_with' absent in canonical VG150. |
| `under` | 22 | `under` | 43 | **EXACT** | HIGH | Exact spatial beneath/underneath match. |
| `hug` | 23 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'hug' absent in canonical VG150. |
| `throw` | 24 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'throw' absent in canonical VG150. |
| `hits` | 25 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'hits' absent in canonical VG150. |
| `snowboard` | 26 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'snowboard' absent in canonical VG150. |
| `is` | 27 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'is' absent in canonical VG150. |
| `kick` | 28 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'kick' absent in canonical VG150. |
| `ski` | 29 | *Unmapped* | — | **NO_MATCH** | HIGH | Specific interaction/action predicate 'ski' absent in canonical VG150. |
| `plays` | 30 | `playing` | 37 | **SEMANTIC_EQUIVALENT** | HIGH | Musical instrument performance or sport game; inflectional form of 'playing'. |
| `read` | 31 | `looking at` | 25 | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Directing visual attention at reading material; visual relational core is 'looking at'. |

---
## 6. VG150 Intra-Dataset Normalization Mappings (50 Predicates)

| Source Predicate | Original ID | Normalized Canonical Target | Canonical ID | Mapping Type | Confidence | Justification & Rule |
|---|:---:|---|:---:|:---:|:---:|---|
| `above` | 1 | `above` | 1 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `across` | 2 | `across` | 2 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `against` | 3 | `against` | 3 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `along` | 4 | `along` | 4 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `and` | 5 | `and` | 5 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `at` | 6 | `at` | 6 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `attached to` | 7 | `attached to` | 7 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `behind` | 8 | `behind` | 8 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `belonging to` | 9 | `belonging to` | 9 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `between` | 10 | `between` | 10 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `carrying` | 11 | `carrying` | 11 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `covered in` | 12 | `covered in` | 12 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `covering` | 13 | `covering` | 13 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `eating` | 14 | `eating` | 14 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `flying in` | 15 | `flying in` | 15 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `for` | 16 | `for` | 16 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `from` | 17 | `from` | 17 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `growing on` | 18 | `growing on` | 18 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `hanging from` | 19 | `hanging from` | 19 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `has` | 20 | `has` | 20 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `holding` | 21 | `holding` | 21 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `in` | 22 | `in` | 22 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `in front of` | 23 | `in front of` | 23 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `laying on` | 24 | `laying on` | 24 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `looking at` | 25 | `looking at` | 25 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `lying on` | 26 | `lying on` | 26 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `made of` | 27 | `made of` | 27 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `mounted on` | 28 | `mounted on` | 28 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `near` | 29 | `near` | 29 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `of` | 30 | `of` | 30 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `on` | 31 | `on` | 31 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `on back of` | 32 | `on back of` | 32 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `over` | 33 | `over` | 33 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `painted on` | 34 | `painted on` | 34 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `parked on` | 35 | `parked on` | 35 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `part of` | 36 | `part of` | 36 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `playing` | 37 | `playing` | 37 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `riding` | 38 | `riding` | 38 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `says` | 39 | `says` | 39 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `sitting on` | 40 | `sitting on` | 40 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `standing on` | 41 | `standing on` | 41 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `to` | 42 | `to` | 42 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `under` | 43 | `under` | 43 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `using` | 44 | `using` | 44 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `walking in` | 45 | `walking in` | 45 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `walking on` | 46 | `walking on` | 46 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `watching` | 47 | `watching` | 47 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `wearing` | 48 | `wearing` | 48 | **EXACT** | HIGH | Canonical baseline identity mapping. |
| `wears` | 49 | `wearing` | 48 | **SEMANTIC_EQUIVALENT** | HIGH | Crowd-sourced inflectional duplicate (ID 49) normalized onto canonical 'wearing' (ID 48) for shared evaluation ontology. |
| `with` | 50 | `with` | 50 | **EXACT** | HIGH | Canonical baseline identity mapping. |

