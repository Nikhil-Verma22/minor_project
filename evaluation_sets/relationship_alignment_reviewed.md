# SGG Benchmark — Reviewed Semantic Relationship Alignment (Second-Pass)

> [!IMPORTANT]
> **Second-Pass Review Summary**: This document re-evaluates all candidate relationships with the objective of constructing the **largest defensible common relationship vocabulary** across VG150, PSG, IndoorVG, and Open Images V6.
> Broadened mappings are approved under **ACCEPTABLE_SEMANTIC_GENERALIZATION** when the canonical VG150 relation represents the core semantic relationship and does not create a materially misleading evaluation.

## 1. Executive Summary Statistics

| Source Dataset | Total Predicates | EXACT (✓) | SEMANTIC_EQUIVALENT (✓) | ACCEPTABLE_GENERALIZATION (~) | REJECTED / NO_MATCH (✗) | Total Defensible Mappings |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PSG** | 56 | 21 | 2 | 5 | 28 | **28** |
| **IndoorVG** | 37 | 29 | 0 | 3 | 5 | **32** |
| **Open Images V6** | 31 | 3 | 7 | 3 | 18 | **13** |

---
## 2. Common Vocabulary Subsets

### A. Strict Common Vocabulary (4-Way Intersection)
- **Size**: **7 predicates**
- **Criteria**: Strictly EXACT or SEMANTIC_EQUIVALENT across all 4 datasets.
- **Predicate List**: `eating, hanging from, holding, in, on, under, wearing`

### B. Extended Common Vocabulary (4-Way Intersection) — *Target Research Vocabulary*
- **Size**: **9 predicates**
- **Criteria**: EXACT, SEMANTIC_EQUIVALENT, or ACCEPTABLE_SEMANTIC_GENERALIZATION across all 4 datasets.
- **Predicate List**: `eating, hanging from, holding, in, looking at, on, playing, under, wearing`

### C. Tri-Dataset Common Vocabulary (3-Way Intersection)
- **Size**: **23 predicates**
- **Criteria**: Supported in VG150 and at least two other datasets.
- **Predicate List**: `against, at, attached to, behind, between, carrying, eating, hanging from, has, holding, in, in front of, looking at, near, on, playing, riding, sitting on, standing on, under, watching, wearing, with`

### D. Union Supported Canonical Predicates
- **Size**: **35 / 50 VG150 predicates** supported by at least one other dataset.

---
## 3. Canonical VG150 Coverage Matrix (All 50 Predicates)

Legend: `✓` = Reliable Exact/Equivalent mapping | `~` = Acceptable Semantic Generalization | `✗` = No Defensible Mapping

| ID | Canonical VG150 Predicate | VG150 | PSG | IndoorVG | Open Images V6 | Source Predicates Mapped |
|:---:|---|:---:|:---:|:---:|:---:|---|
| 1 | `above` | ✓ | ✗ | ✓ | ✗ | IndoorVG: above (✓) |
| 2 | `across` | ✓ | ✗ | ✗ | ✗ | *None* |
| 3 | `against` | ✓ | ✓ | ✓ | ✗ | PSG: leaning on (✓)<br>IndoorVG: against (✓) |
| 4 | `along` | ✓ | ✗ | ✗ | ✗ | *None* |
| 5 | `and` | ✓ | ✓ | ✗ | ✗ | PSG: and (✓) |
| 6 | `at` | ✓ | ✗ | ✓ | ✓ | IndoorVG: at (✓), sitting at (~)<br>OIDv6: at (✓) |
| 7 | `attached to` | ✓ | ✓ | ✓ | ✗ | PSG: attached to (✓)<br>IndoorVG: attached to (✓) |
| 8 | `behind` | ✓ | ~ | ✓ | ✗ | PSG: standing behind (~), sitting behind (~)<br>IndoorVG: behind (✓) |
| 9 | `belonging to` | ✓ | ✗ | ✗ | ✗ | *None* |
| 10 | `between` | ✓ | ✓ | ✓ | ✗ | PSG: between (✓)<br>IndoorVG: between (✓) |
| 11 | `carrying` | ✓ | ✓ | ✓ | ✗ | PSG: carrying (✓)<br>IndoorVG: carrying (✓) |
| 12 | `covered in` | ✓ | ✗ | ✗ | ✗ | *None* |
| 13 | `covering` | ✓ | ✗ | ✓ | ✗ | IndoorVG: covering (✓) |
| 14 | `eating` | ✓ | ✓ | ✓ | ✓ | PSG: eating (✓)<br>IndoorVG: eating (✓)<br>OIDv6: eat (✓) |
| 15 | `flying in` | ✓ | ✗ | ✗ | ✗ | *None* |
| 16 | `for` | ✓ | ✗ | ✓ | ✗ | IndoorVG: for (✓) |
| 17 | `from` | ✓ | ✗ | ✗ | ✗ | *None* |
| 18 | `growing on` | ✓ | ✗ | ✗ | ✗ | *None* |
| 19 | `hanging from` | ✓ | ✓ | ✓ | ✓ | PSG: hanging from (✓)<br>IndoorVG: hanging from (✓)<br>OIDv6: hang (✓) |
| 20 | `has` | ✓ | ✗ | ✓ | ~ | IndoorVG: has (✓)<br>OIDv6: contain (~) |
| 21 | `holding` | ✓ | ✓ | ✓ | ✓ | PSG: holding (✓)<br>IndoorVG: holding (✓)<br>OIDv6: holds (✓), holding_hands (~) |
| 22 | `in` | ✓ | ✓ | ✓ | ✓ | PSG: in (✓)<br>IndoorVG: in (✓)<br>OIDv6: inside_of (✓) |
| 23 | `in front of` | ✓ | ✓ | ✓ | ✗ | PSG: in front of (✓)<br>IndoorVG: in front of (✓) |
| 24 | `laying on` | ✓ | ✗ | ✓ | ✗ | IndoorVG: laying on (✓) |
| 25 | `looking at` | ✓ | ✓ | ✓ | ~ | PSG: looking at (✓)<br>IndoorVG: looking at (✓), reading (~)<br>OIDv6: read (~) |
| 26 | `lying on` | ✓ | ✗ | ✓ | ✗ | IndoorVG: lying on (✓) |
| 27 | `made of` | ✓ | ✗ | ✗ | ✗ | *None* |
| 28 | `mounted on` | ✓ | ✗ | ✓ | ✗ | IndoorVG: mounted on (✓) |
| 29 | `near` | ✓ | ~ | ✓ | ✗ | PSG: beside (~)<br>IndoorVG: near (✓) |
| 30 | `of` | ✓ | ✗ | ✓ | ✗ | IndoorVG: of (✓) |
| 31 | `on` | ✓ | ✓ | ✓ | ✓ | PSG: on (✓)<br>IndoorVG: on (✓)<br>OIDv6: on (✓) |
| 32 | `on back of` | ✓ | ✓ | ✗ | ✗ | PSG: on back of (✓) |
| 33 | `over` | ✓ | ✓ | ✗ | ✗ | PSG: over (✓) |
| 34 | `painted on` | ✓ | ✗ | ✗ | ✗ | *None* |
| 35 | `parked on` | ✓ | ✗ | ✗ | ✗ | *None* |
| 36 | `part of` | ✓ | ✗ | ✗ | ✗ | *None* |
| 37 | `playing` | ✓ | ✓ | ~ | ✓ | PSG: playing with (~), playing (✓)<br>IndoorVG: playing with (~)<br>OIDv6: plays (✓) |
| 38 | `riding` | ✓ | ✓ | ✗ | ✓ | PSG: riding (✓)<br>OIDv6: ride (✓) |
| 39 | `says` | ✓ | ✗ | ✗ | ✗ | *None* |
| 40 | `sitting on` | ✓ | ✓ | ✓ | ✗ | PSG: sitting on (✓)<br>IndoorVG: sitting on (✓) |
| 41 | `standing on` | ✓ | ✓ | ✓ | ✗ | PSG: standing on (✓)<br>IndoorVG: standing on (✓) |
| 42 | `to` | ✓ | ✗ | ✗ | ✗ | *None* |
| 43 | `under` | ✓ | ✓ | ✓ | ✓ | PSG: under (✓)<br>IndoorVG: under (✓)<br>OIDv6: under (✓) |
| 44 | `using` | ✓ | ✗ | ✓ | ✗ | IndoorVG: using (✓) |
| 45 | `walking in` | ✓ | ✗ | ✗ | ✗ | *None* |
| 46 | `walking on` | ✓ | ✓ | ✗ | ✗ | PSG: walking on (✓) |
| 47 | `watching` | ✓ | ✓ | ✓ | ✗ | PSG: watching (✓)<br>IndoorVG: watching (✓) |
| 48 | `wearing` | ✓ | ✓ | ✓ | ✓ | PSG: wearing (✓), wears (✓)<br>IndoorVG: wearing (✓)<br>OIDv6: wears (✓) |
| 49 | `wears` | ✓ | ✗ | ✗ | ✗ | *None* |
| 50 | `with` | ✓ | ~ | ✓ | ✗ | PSG: walking with (~)<br>IndoorVG: with (✓) |

---
## 4. PSG Second-Pass Reviewed Mappings (56 Predicates)

| Source Predicate | Canonical VG150 Target | Original Status | Revised Decision | Confidence | Rationale & Justification |
|---|---|:---:|:---:|:---:|---|
| `over` | `over` (ID 33) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; expresses spatial position higher than/vertically above the object in the subject→object direction. |
| `in front of` | `in front of` (ID 23) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; expresses anterior spatial positioning relative to the reference object. |
| `beside` | `near` (ID 29) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Lateral adjacency ('beside') is a specific topological instantiation of general spatial proximity ('near'). Every instance of being beside an object is visually and physically near it; does not create a misleading scene graph evaluation. |
| `on` | `on` (ID 31) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; expresses physical contact and surface support. |
| `in` | `in` (ID 22) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; expresses spatial containment or interior placement. |
| `attached to` | `attached to` (ID 7) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes physical fastening or mechanical connection. |
| `hanging from` | `hanging from` (ID 19) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes suspension from a support anchor. |
| `on back of` | `on back of` (ID 32) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes surface support upon the posterior or dorsal region of an entity. |
| `falling off` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic downward detachment action with no semantic counterpart in VG150. |
| `going down` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Directional descending movement; no equivalent predicate in VG150. |
| `stopping on` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic deceleration/transition to rest on a surface; no general counterpart in VG150. |
| `between` | `between` (ID 10) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes spatial position intermediate to reference bounds. |
| `and` | `and` (ID 5) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; symmetrical coordination / co-presence relation. |
| `under` | `under` (ID 43) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes spatial position beneath/underneath. |
| `riding` | `riding` (ID 38) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; denotes active operation/transit upon a mount or vehicle in the subject→object direction. |
| `playing with` | `playing` (ID 37) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core relationship is recreational activity/play engagement with a peer or toy. VG150 'playing' captures this core interaction sufficiently for evaluation. |
| `sitting on` | `sitting on` (ID 40) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; seated bodily posture supported by a surface. |
| `standing on` | `standing on` (ID 41) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; upright bodily posture supported by a surface. |
| `jumping on` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic vertical propulsion action onto a surface; no VG150 equivalent. |
| `walking on` | `walking on` (ID 46) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; pedestrian locomotion supported by a surface. |
| `running on` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Rapid gaited locomotion on a surface; distinct from walking in gait mechanics and speed; absent in VG150. |
| `climbing` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Ascending locomotion using limbs; no counterpart in VG150. |
| `jumping to` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Directed ballistic airborne movement; no counterpart in VG150. |
| `holding` | `holding` (ID 21) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; manual grasp/support of an object. |
| `touching` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Physical surface contact without load-bearing support or containment; absent in VG150. |
| `carrying` | `carrying` (ID 11) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; bearing and transporting an object. |
| `eating` | `eating` (ID 14) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; consumption/oral ingestion of food. |
| `drinking` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Liquid ingestion action; absent in VG150 (VG150 only contains 'eating'). |
| `pulling` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Exerting tensile force to draw an object toward the subject; absent in VG150. |
| `pushing` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Exerting compressive force to propel an object away; absent in VG150. |
| `throwing` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Ballistic propulsion/release of a projectile; absent in VG150. |
| `catching` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Manual interception/capture of a projectile; absent in VG150. |
| `wearing` | `wearing` (ID 48) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; wearing attire or accessory. |
| `wears` | `wearing` (ID 48) | EXACT | **SEMANTIC_EQUIVALENT** | HIGH | Both express the subject wearing attire/garment in the subject→object direction; only grammatical inflection differs (3rd person singular present 'wears' vs participle 'wearing'). Consolidates onto canonical 'wearing' (ID 48); also exact match to duplicate VG150 'wears' (ID 49). |
| `pointing to` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Deictic gestural indication; absent in VG150. |
| `watching` | `watching` (ID 47) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; sustained visual monitoring of a target. |
| `driving` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Vehicular control and operation; absent in VG150. |
| `cooking` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Culinary food preparation; absent in VG150. |
| `looking at` | `looking at` (ID 25) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; directing gaze toward a target. |
| `kissing` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Affectionate oral contact; absent in VG150. |
| `hugging` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Affectionate bodily embrace; absent in VG150. |
| `teaching` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Instructional interpersonal social interaction; absent in VG150. |
| `talking to` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Verbal interpersonal communication; absent in VG150 (VG150 'says' represents displayed written text, e.g. 'sign says stop'). |
| `waiting for` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Temporal expectation/delay; absent in VG150. |
| `cleaning` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Sanitation/maintenance action on an object; absent in VG150. |
| `playing` | `playing` (ID 37) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; recreational activity/sport/game engagement. |
| `chasing` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Pursuit locomotion; absent in VG150. |
| `feeding` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Providing nourishment to another entity; absent in VG150. |
| `standing behind` | `behind` (ID 8) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core relational semantic is posterior spatial positioning ('behind'); upright posture ('standing') is secondary bodily state information that does not contradict the core spatial relation. |
| `sitting behind` | `behind` (ID 8) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core relational semantic is posterior spatial positioning ('behind'); seated posture is secondary bodily state that does not contradict the core spatial relation. |
| `walking with` | `with` (ID 50) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core relational semantic between the two entities is social accompaniment / companionship ('with'); concurrent walking locomotion is secondary activity context. |
| `swinging` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Periodic oscillatory movement; absent in VG150. |
| `entering` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic spatial ingress transition; absent in VG150. |
| `exiting` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic spatial egress transition; absent in VG150. |
| `enclosing` | *None* | AMBIGUOUS | **AMBIGUOUS_REJECTED** | HIGH | 'enclosing' denotes 3D boundary containment or framing, whereas VG150 'covering' denotes 2D surface overlay. Mapping 3D volumetric framing to surface coating alters the fundamental topological physics and would create a materially misleading evaluation. |
| `leaning on` | `against` (ID 3) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both describe oblique physical contact where subject rests inclined against an upright support in the subject→object direction (e.g. bicycle leaning on wall vs bicycle against wall). |

---
## 5. IndoorVG Second-Pass Reviewed Mappings (37 Predicates)

| Source Predicate | Canonical VG150 Target | Original Status | Revised Decision | Confidence | Rationale & Justification |
|---|---|:---:|:---:|:---:|---|
| `above` | `above` (ID 1) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; vertical elevation without contact. |
| `against` | `against` (ID 3) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; physical contact/bearing against an upright surface. |
| `at` | `at` (ID 6) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; spatial collocation. |
| `attached to` | `attached to` (ID 7) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; physical or mechanical connection. |
| `behind` | `behind` (ID 8) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; posterior spatial positioning. |
| `between` | `between` (ID 10) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; intermediate spatial positioning. |
| `carrying` | `carrying` (ID 11) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; bearing and transporting an object. |
| `covering` | `covering` (ID 13) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; overlaying surface contact. |
| `cutting` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Instrumental separation of material; absent in VG150. |
| `drinking` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Liquid ingestion action; absent in VG150. |
| `eating` | `eating` (ID 14) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; food consumption. |
| `filled with` | *None* | AMBIGUOUS | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'in' creates a DIRECTIONALITY_CONFLICT (container is subject, contents are object); mapping to 'has' loses physical containment volume semantics, and mapping to 'covered in' conflates interior volume filling with exterior surface coating. |
| `for` | `for` (ID 16) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; functional purpose/destination. |
| `hanging from` | `hanging from` (ID 19) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; suspension from support anchor. |
| `has` | `has` (ID 20) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; meronymic or possession relationship. |
| `holding` | `holding` (ID 21) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; manual grasp/support of an object. |
| `in` | `in` (ID 22) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; spatial containment/interior placement. |
| `in front of` | `in front of` (ID 23) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; anterior spatial positioning. |
| `laying on` | `laying on` (ID 24) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; recumbent posture supported by a surface. |
| `looking at` | `looking at` (ID 25) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; directing visual gaze. |
| `lying on` | `lying on` (ID 26) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; recumbent bodily posture upon a surface. |
| `mounted on` | `mounted on` (ID 28) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; fixed fixture attachment on a surface. |
| `near` | `near` (ID 29) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; general spatial proximity. |
| `of` | `of` (ID 30) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; associative or possessive connection. |
| `on` | `on` (ID 31) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; surface contact and support. |
| `playing with` | `playing` (ID 37) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core interaction is recreational engagement with an object/peer; accurately captured by canonical 'playing'. |
| `reading` | `looking at` (ID 25) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core visual engagement when reading text/book/screen is directing gaze at the object ('looking at'). In visual scene graphs, 'looking at' captures the primary perceptual relation. |
| `sitting at` | `at` (ID 6) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Core spatial relation relative to a desk/table is spatial collocation ('at'). Mapping to 'sitting on' is physically false (person does not sit on top of desk), but mapping to 'at' accurately captures spatial presence without distortion. |
| `sitting on` | `sitting on` (ID 40) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; seated posture supported by surface. |
| `standing on` | `standing on` (ID 41) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; upright posture supported by surface. |
| `taking` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic acquisition action; absent in VG150. |
| `talking on` | *None* | AMBIGUOUS | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'using' is overly broad and fails to capture the defining verbal telecommunication relationship (talking on phone). |
| `under` | `under` (ID 43) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; spatial position beneath/underneath. |
| `using` | `using` (ID 44) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; instrumental utilization of an object. |
| `watching` | `watching` (ID 47) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; sustained visual monitoring. |
| `wearing` | `wearing` (ID 48) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; dressed in or carrying apparel. |
| `with` | `with` (ID 50) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; co-presence or accompaniment. |

---
## 6. Open Images V6 Second-Pass Reviewed Mappings (31 Predicates)

| Source Predicate | Canonical VG150 Target | Original Status | Revised Decision | Confidence | Rationale & Justification |
|---|---|:---:|:---:|:---:|---|
| `at` | `at` (ID 6) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; general spatial collocation (e.g. person at desk/table). |
| `holds` | `holding` (ID 21) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both describe the subject physically grasping/supporting the object in the subject→object direction; only grammatical inflection differs (3rd person singular present 'holds' vs participle 'holding'). |
| `wears` | `wearing` (ID 48) | EXACT | **SEMANTIC_EQUIVALENT** | HIGH | Both express the subject wearing attire/garment in the subject→object direction; only grammatical inflection differs (3rd person singular present 'wears' vs participle 'wearing'). Consolidates onto canonical 'wearing' (ID 48); also exact match to duplicate VG150 'wears' (ID 49). |
| `surf` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Water-sport locomotion on a surfboard/wave; absent in VG150. |
| `hang` | `hanging from` (ID 19) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both express physical suspension of the subject from an anchor/support object in the subject→object direction (e.g. painting hang wall vs painting hanging from wall); lexical variation of suspension. |
| `drink` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Liquid consumption; absent in VG150 (VG150 only contains 'eating'). |
| `holding_hands` | `holding` (ID 21) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | MEDIUM | Both entities are engaged in mutual physical manual grasping. In SGG, manual grasp is the core mechanical contact, making 'holding' an acceptable generalization of the physical interaction. |
| `on` | `on` (ID 31) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; surface contact and support. |
| `ride` | `riding` (ID 38) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both express active locomotion/transit operating or mounted upon a vehicle or animal in the subject→object direction; only grammatical inflection differs (base form 'ride' vs gerund 'riding'). |
| `dance` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Rhythmic bodily choreography; absent in VG150. |
| `skateboard` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Action-specific locomotion on a skateboard; absent in VG150. |
| `catch` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic interception/capture of an object in motion; absent in VG150. |
| `highfive` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Interpersonal celebratory palm-slap gesture; absent in VG150. |
| `inside_of` | `in` (ID 22) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both denote spatial containment where subject is enclosed within the interior boundary of the object in the subject→object direction (e.g. food inside_of bowl vs food in bowl). |
| `eat` | `eating` (ID 14) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both express oral ingestion of food by subject in the subject→object direction; lexical inflection variation (base form 'eat' vs participle 'eating'). |
| `cut` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Instrumental material division; absent in VG150. |
| `contain` | `has` (ID 20) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Subject is container, object is contents (e.g. bowl contain fruit). Mapping to 'in' is rejected due to DIRECTIONALITY_CONFLICT. Mapping to 'has' (bowl has fruit) preserves subject→object direction and captures container possession. |
| `handshake` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Interpersonal mutual hand-clasp greeting; absent in VG150. |
| `kiss` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Affectionate oral contact; absent in VG150. |
| `talk_on_phone` | *None* | AMBIGUOUS | **AMBIGUOUS_REJECTED** | HIGH | Mapping to 'using' loses the defining telecommunication/verbal conversation relationship. |
| `interacts_with` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Unspecified generic interaction umbrella category; no direct semantic equivalent in VG150. |
| `under` | `under` (ID 43) | EXACT | **EXACT** | HIGH | Exact lexical and semantic match; spatial placement beneath/underneath. |
| `hug` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Affectionate bodily embrace; absent in VG150. |
| `throw` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Ballistic propulsion/release; absent in VG150. |
| `hits` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Dynamic forceful physical impact; absent in VG150. |
| `snowboard` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Winter-sport locomotion on a snowboard; absent in VG150. |
| `is` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Entity-to-attribute copula link (e.g. 'Table is Wooden', 'Person is Walk'); in VG150 attributes are stored in object attribute arrays, not as relationship triplets. |
| `kick` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Impact strike with the foot; absent in VG150. |
| `ski` | *None* | NO_MATCH | **NO_MATCH** | HIGH | Winter-sport locomotion on skis; absent in VG150. |
| `plays` | `playing` (ID 37) | SEMANTIC_EQUIVALENT | **SEMANTIC_EQUIVALENT** | HIGH | Both express recreational game/sport or musical instrument performance in the subject→object direction; only grammatical inflection differs (3rd person singular 'plays' vs gerund 'playing'). |
| `read` | `looking at` (ID 25) | AMBIGUOUS | **ACCEPTABLE_SEMANTIC_GENERALIZATION** | HIGH | Directing visual gaze at reading material ('looking at') is the prerequisite visual relational act; preserves the perceptual interaction. |

