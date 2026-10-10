# SGG Benchmark — Candidate Semantic Relationship Alignment

> [!IMPORTANT]
> This document represents a **conservative semantic ontology alignment analysis**. 
> It does **NOT** enforce automated mapping into frozen datasets or create the final evaluation vocabulary. 
> Only candidate mappings with rigorous semantic justification are proposed.

## 1. Alignment Methodology & Mapping Categories

Every source predicate is classified into exactly one category against the **50 canonical VG150 predicates**:
- **A. EXACT**: Same predicate meaning, effectively identical relationship semantics and subject→object direction.
- **B. SEMANTIC_EQUIVALENT**: Different lexical or grammatical form, but identical relationship semantics, entity roles, and direction.
- **C. AMBIGUOUS**: Potentially related VG150 candidate exists, but the correspondence is incomplete, loses key semantic constraints, or alters topological/spatial scope.
- **D. NO_MATCH**: No appropriate or valid VG150 predicate exists.

### Critical Semantic Rules Applied:
1. **Directionality Preservation**: Subject→Object orientation is strictly preserved. Inversion (e.g. `contained in` ↔ `contain`) is rejected.
2. **Action Specificity**: Domain-specific physical actions (`surfing`, `kissing`, `throwing`, `cutting`) are never collapsed into generic interaction umbrellas.
3. **Attribute vs. Relation**: Entity-to-attribute copulas (such as Open Images `is`) are excluded from object-object relational mappings.
4. **Contact vs. Proximity**: Distinct physical phenomena (`beside` vs `near`, `touching` vs `on`) are strictly distinguished.

---
## 2. Summary Statistics

| Source Dataset | Total Predicates | EXACT (A) | SEMANTIC_EQUIVALENT (B) | AMBIGUOUS (C) | NO_MATCH (D) | Eligible (A+B) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PSG** | 56 | 22 | 1 | 6 | 27 | **23** |
| **IndoorVG** | 37 | 29 | 0 | 5 | 3 | **29** |
| **Open Images V6** | 31 | 4 | 6 | 4 | 17 | **10** |

### Provisional Common Vocabulary Observations
- **Strict 4-Way Common Core (Intersection across all 4 datasets)**: **6 predicates**
  > `eating, hanging from, holding, in, on, under`
- **Canonical VG150 Predicates Supported by at least one source (Union of A+B)**: **36 / 50 predicates**
  > `above, against, and, at, attached to, behind, between, carrying, covering, eating, for, hanging from, has, holding, in, in front of, laying on, looking at, lying on, mounted on, near, of, on, on back of, over, playing, riding, sitting on, standing on, under, using, walking on, watching, wearing, wears, with`
- *Note: Both common subsets are PROVISIONAL and pending manual review.*

---
## 3. PSG Candidate Alignment (56 Predicates)

| Source Predicate | VG150 Candidate | Mapping Type | Confidence | Semantic Justification & Evidence |
|---|---|:---:|:---:|---|
| `over` | `over` (ID 33) | **EXACT** | HIGH | Exact lexical and semantic match; expresses spatial position higher than/vertically above the object in the subject→object direction. |
| `in front of` | `in front of` (ID 23) | **EXACT** | HIGH | Exact lexical and semantic match; expresses anterior spatial positioning relative to the reference object. |
| `beside` | `near` (ID 29) | **AMBIGUOUS** | MEDIUM | 'beside' specifies strict lateral adjacency (next to), whereas VG150 'near' denotes general spatial proximity without orientation constraints. |
| `on` | `on` (ID 31) | **EXACT** | HIGH | Exact lexical and semantic match; expresses physical contact and surface support. |
| `in` | `in` (ID 22) | **EXACT** | HIGH | Exact lexical and semantic match; expresses spatial containment or interior placement. |
| `attached to` | `attached to` (ID 7) | **EXACT** | HIGH | Exact lexical and semantic match; denotes physical fastening or mechanical connection. |
| `hanging from` | `hanging from` (ID 19) | **EXACT** | HIGH | Exact lexical and semantic match; denotes suspension from a support anchor. |
| `on back of` | `on back of` (ID 32) | **EXACT** | HIGH | Exact lexical and semantic match; denotes surface support upon the posterior or dorsal region of an entity. |
| `falling off` | *None* | **NO_MATCH** | HIGH | Dynamic downward detachment action with no semantic counterpart in VG150. |
| `going down` | *None* | **NO_MATCH** | HIGH | Directional descending movement; no equivalent predicate in VG150. |
| `stopping on` | *None* | **NO_MATCH** | HIGH | Dynamic deceleration/transition to rest on a surface; no general counterpart in VG150. |
| `between` | `between` (ID 10) | **EXACT** | HIGH | Exact lexical and semantic match; denotes spatial position intermediate to reference bounds. |
| `and` | `and` (ID 5) | **EXACT** | HIGH | Exact lexical and semantic match; symmetrical coordination / co-presence relation. |
| `under` | `under` (ID 43) | **EXACT** | HIGH | Exact lexical and semantic match; denotes spatial position beneath/underneath. |
| `riding` | `riding` (ID 38) | **EXACT** | HIGH | Exact lexical and semantic match; denotes active operation/transit upon a mount or vehicle in the subject→object direction. |
| `playing with` | `playing` (ID 37) | **AMBIGUOUS** | MEDIUM | 'playing with' specifies leisure interaction with a specific plaything/toy or peer, whereas VG150 'playing' is broader (e.g. playing games, playing instruments). |
| `sitting on` | `sitting on` (ID 40) | **EXACT** | HIGH | Exact lexical and semantic match; seated bodily posture supported by a surface. |
| `standing on` | `standing on` (ID 41) | **EXACT** | HIGH | Exact lexical and semantic match; upright bodily posture supported by a surface. |
| `jumping on` | *None* | **NO_MATCH** | HIGH | Dynamic vertical propulsion action onto a surface; no VG150 equivalent. |
| `walking on` | `walking on` (ID 46) | **EXACT** | HIGH | Exact lexical and semantic match; pedestrian locomotion supported by a surface. |
| `running on` | *None* | **NO_MATCH** | HIGH | Rapid gaited locomotion on a surface; distinct from walking in gait mechanics and speed; absent in VG150. |
| `climbing` | *None* | **NO_MATCH** | HIGH | Ascending locomotion using limbs; no counterpart in VG150. |
| `jumping to` | *None* | **NO_MATCH** | HIGH | Directed ballistic airborne movement; no counterpart in VG150. |
| `holding` | `holding` (ID 21) | **EXACT** | HIGH | Exact lexical and semantic match; manual grasp/support of an object. |
| `touching` | *None* | **NO_MATCH** | HIGH | Physical surface contact without load-bearing support or containment; absent in VG150. |
| `carrying` | `carrying` (ID 11) | **EXACT** | HIGH | Exact lexical and semantic match; bearing and transporting an object. |
| `eating` | `eating` (ID 14) | **EXACT** | HIGH | Exact lexical and semantic match; consumption/oral ingestion of food. |
| `drinking` | *None* | **NO_MATCH** | HIGH | Liquid ingestion action; absent in VG150 (VG150 only contains 'eating'). |
| `pulling` | *None* | **NO_MATCH** | HIGH | Exerting tensile force to draw an object toward the subject; absent in VG150. |
| `pushing` | *None* | **NO_MATCH** | HIGH | Exerting compressive force to propel an object away; absent in VG150. |
| `throwing` | *None* | **NO_MATCH** | HIGH | Ballistic propulsion/release of a projectile; absent in VG150. |
| `catching` | *None* | **NO_MATCH** | HIGH | Manual interception/capture of a projectile; absent in VG150. |
| `wearing` | `wearing` (ID 48) | **EXACT** | HIGH | Exact lexical and semantic match; wearing attire or accessory. |
| `wears` | `wears` (ID 49) | **EXACT** | HIGH | Exact lexical and semantic match; identical 3rd person singular present predicate present in VG150. |
| `pointing to` | *None* | **NO_MATCH** | HIGH | Deictic gestural indication; absent in VG150. |
| `watching` | `watching` (ID 47) | **EXACT** | HIGH | Exact lexical and semantic match; sustained visual monitoring of a target. |
| `driving` | *None* | **NO_MATCH** | HIGH | Vehicular control and operation; absent in VG150. |
| `cooking` | *None* | **NO_MATCH** | HIGH | Culinary food preparation; absent in VG150. |
| `looking at` | `looking at` (ID 25) | **EXACT** | HIGH | Exact lexical and semantic match; directing gaze toward a target. |
| `kissing` | *None* | **NO_MATCH** | HIGH | Affectionate oral contact; absent in VG150. |
| `hugging` | *None* | **NO_MATCH** | HIGH | Affectionate bodily embrace; absent in VG150. |
| `teaching` | *None* | **NO_MATCH** | HIGH | Instructional interpersonal social interaction; absent in VG150. |
| `talking to` | *None* | **NO_MATCH** | HIGH | Verbal interpersonal communication; absent in VG150 (VG150 'says' represents displayed written text, e.g. 'sign says stop'). |
| `waiting for` | *None* | **NO_MATCH** | HIGH | Temporal expectation/delay; absent in VG150. |
| `cleaning` | *None* | **NO_MATCH** | HIGH | Sanitation/maintenance action on an object; absent in VG150. |
| `playing` | `playing` (ID 37) | **EXACT** | HIGH | Exact lexical and semantic match; recreational activity/sport/game engagement. |
| `chasing` | *None* | **NO_MATCH** | HIGH | Pursuit locomotion; absent in VG150. |
| `feeding` | *None* | **NO_MATCH** | HIGH | Providing nourishment to another entity; absent in VG150. |
| `standing behind` | `behind` (ID 8) | **AMBIGUOUS** | MEDIUM | Combines upright posture ('standing') with posterior spatial relation ('behind'); mapping to 'behind' captures relative position but discards postural semantics. |
| `sitting behind` | `behind` (ID 8) | **AMBIGUOUS** | MEDIUM | Combines seated posture ('sitting') with posterior spatial relation ('behind'); mapping to 'behind' discards postural semantics. |
| `walking with` | `with` (ID 50) | **AMBIGUOUS** | MEDIUM | 'walking with' composes pedestrian locomotion with social company; VG150 lacks 'walking with' and mapping to 'with' discards the locomotion activity. |
| `swinging` | *None* | **NO_MATCH** | HIGH | Periodic oscillatory movement; absent in VG150. |
| `entering` | *None* | **NO_MATCH** | HIGH | Dynamic spatial ingress transition; absent in VG150. |
| `exiting` | *None* | **NO_MATCH** | HIGH | Dynamic spatial egress transition; absent in VG150. |
| `enclosing` | `covering` (ID 13) | **AMBIGUOUS** | LOW | 'enclosing' denotes 3D boundary containment or framing, whereas 'covering' implies 2D surface overlay. Distinct topological relationships. |
| `leaning on` | `against` (ID 3) | **SEMANTIC_EQUIVALENT** | HIGH | Both describe oblique physical contact where subject rests inclined against an upright support in the subject→object direction (e.g. bicycle leaning on wall vs bicycle against wall). |

---
## 4. IndoorVG Candidate Alignment (37 Predicates)

| Source Predicate | VG150 Candidate | Mapping Type | Confidence | Semantic Justification & Evidence |
|---|---|:---:|:---:|---|
| `above` | `above` (ID 1) | **EXACT** | HIGH | Exact lexical and semantic match; vertical elevation without contact. |
| `against` | `against` (ID 3) | **EXACT** | HIGH | Exact lexical and semantic match; physical contact/bearing against an upright surface. |
| `at` | `at` (ID 6) | **EXACT** | HIGH | Exact lexical and semantic match; spatial collocation. |
| `attached to` | `attached to` (ID 7) | **EXACT** | HIGH | Exact lexical and semantic match; physical or mechanical connection. |
| `behind` | `behind` (ID 8) | **EXACT** | HIGH | Exact lexical and semantic match; posterior spatial positioning. |
| `between` | `between` (ID 10) | **EXACT** | HIGH | Exact lexical and semantic match; intermediate spatial positioning. |
| `carrying` | `carrying` (ID 11) | **EXACT** | HIGH | Exact lexical and semantic match; bearing and transporting an object. |
| `covering` | `covering` (ID 13) | **EXACT** | HIGH | Exact lexical and semantic match; overlaying surface contact. |
| `cutting` | *None* | **NO_MATCH** | HIGH | Instrumental separation of material; absent in VG150. |
| `drinking` | *None* | **NO_MATCH** | HIGH | Liquid ingestion action; absent in VG150. |
| `eating` | `eating` (ID 14) | **EXACT** | HIGH | Exact lexical and semantic match; food consumption. |
| `filled with` | `covered in` (ID 12) | **AMBIGUOUS** | LOW | 'filled with' denotes interior volumetric containment of bulk substance (container→substance); mapping to 'covered in' changes containment to exterior surface coating, while mapping to 'in' reverses subject/object direction. |
| `for` | `for` (ID 16) | **EXACT** | HIGH | Exact lexical and semantic match; functional purpose/destination. |
| `hanging from` | `hanging from` (ID 19) | **EXACT** | HIGH | Exact lexical and semantic match; suspension from support anchor. |
| `has` | `has` (ID 20) | **EXACT** | HIGH | Exact lexical and semantic match; meronymic or possession relationship. |
| `holding` | `holding` (ID 21) | **EXACT** | HIGH | Exact lexical and semantic match; manual grasp/support of an object. |
| `in` | `in` (ID 22) | **EXACT** | HIGH | Exact lexical and semantic match; spatial containment/interior placement. |
| `in front of` | `in front of` (ID 23) | **EXACT** | HIGH | Exact lexical and semantic match; anterior spatial positioning. |
| `laying on` | `laying on` (ID 24) | **EXACT** | HIGH | Exact lexical and semantic match; recumbent posture supported by a surface. |
| `looking at` | `looking at` (ID 25) | **EXACT** | HIGH | Exact lexical and semantic match; directing visual gaze. |
| `lying on` | `lying on` (ID 26) | **EXACT** | HIGH | Exact lexical and semantic match; recumbent bodily posture upon a surface. |
| `mounted on` | `mounted on` (ID 28) | **EXACT** | HIGH | Exact lexical and semantic match; fixed fixture attachment on a surface. |
| `near` | `near` (ID 29) | **EXACT** | HIGH | Exact lexical and semantic match; general spatial proximity. |
| `of` | `of` (ID 30) | **EXACT** | HIGH | Exact lexical and semantic match; associative or possessive connection. |
| `on` | `on` (ID 31) | **EXACT** | HIGH | Exact lexical and semantic match; surface contact and support. |
| `playing with` | `playing` (ID 37) | **AMBIGUOUS** | MEDIUM | 'playing with' specifies recreational interaction with a specific plaything/toy or peer, whereas VG150 'playing' is broader (e.g. playing games, sports, instruments). |
| `reading` | `looking at` (ID 25) | **AMBIGUOUS** | MEDIUM | 'reading' implies specific cognitive text engagement; mapping to 'looking at' dilutes the semantic focus. |
| `sitting at` | `sitting on` (ID 40) | **AMBIGUOUS** | MEDIUM | 'sitting at' denotes seated proximity relative to a table/desk without sitting ON that surface (the chair is sat on, the desk is sat at); mapping to 'sitting on' introduces erroneous direct surface support, while mapping to 'at' discards seated posture. |
| `sitting on` | `sitting on` (ID 40) | **EXACT** | HIGH | Exact lexical and semantic match; seated posture supported by surface. |
| `standing on` | `standing on` (ID 41) | **EXACT** | HIGH | Exact lexical and semantic match; upright posture supported by surface. |
| `taking` | *None* | **NO_MATCH** | HIGH | Dynamic acquisition action; absent in VG150. |
| `talking on` | `using` (ID 44) | **AMBIGUOUS** | MEDIUM | Specific telecommunication action (talking on phone); mapping to 'using' loses the verbal/conversational semantic. |
| `under` | `under` (ID 43) | **EXACT** | HIGH | Exact lexical and semantic match; spatial position beneath/underneath. |
| `using` | `using` (ID 44) | **EXACT** | HIGH | Exact lexical and semantic match; instrumental utilization of an object. |
| `watching` | `watching` (ID 47) | **EXACT** | HIGH | Exact lexical and semantic match; sustained visual monitoring. |
| `wearing` | `wearing` (ID 48) | **EXACT** | HIGH | Exact lexical and semantic match; dressed in or carrying apparel. |
| `with` | `with` (ID 50) | **EXACT** | HIGH | Exact lexical and semantic match; co-presence or accompaniment. |

---
## 5. Open Images V6 Candidate Alignment (31 Predicates)

| Source Predicate | VG150 Candidate | Mapping Type | Confidence | Semantic Justification & Evidence |
|---|---|:---:|:---:|---|
| `at` | `at` (ID 6) | **EXACT** | HIGH | Exact lexical and semantic match; general spatial collocation (e.g. person at desk/table). |
| `holds` | `holding` (ID 21) | **SEMANTIC_EQUIVALENT** | HIGH | Both describe the subject physically grasping/supporting the object in the subject→object direction; only grammatical inflection differs (3rd person singular present 'holds' vs participle 'holding'). |
| `wears` | `wears` (ID 49) | **EXACT** | HIGH | Exact lexical and semantic match; identical 3rd person singular present form present in VG150 (and equivalent to 'wearing'). |
| `surf` | *None* | **NO_MATCH** | HIGH | Water-sport locomotion on a surfboard/wave; absent in VG150. |
| `hang` | `hanging from` (ID 19) | **SEMANTIC_EQUIVALENT** | HIGH | Both express physical suspension of the subject from an anchor/support object in the subject→object direction (e.g. painting hang wall vs painting hanging from wall); lexical variation of suspension. |
| `drink` | *None* | **NO_MATCH** | HIGH | Liquid consumption; absent in VG150 (VG150 only contains 'eating'). |
| `holding_hands` | `holding` (ID 21) | **AMBIGUOUS** | LOW | 'holding_hands' is a reciprocal/interpersonal mutual grasp between two persons; VG150 'holding' describes unilateral object manipulation. Mapping to 'holding' loses interpersonal reciprocity. |
| `on` | `on` (ID 31) | **EXACT** | HIGH | Exact lexical and semantic match; surface contact and support. |
| `ride` | `riding` (ID 38) | **SEMANTIC_EQUIVALENT** | HIGH | Both express active locomotion/transit operating or mounted upon a vehicle or animal in the subject→object direction; only grammatical inflection differs (base form 'ride' vs gerund 'riding'). |
| `dance` | *None* | **NO_MATCH** | HIGH | Rhythmic bodily choreography; absent in VG150. |
| `skateboard` | *None* | **NO_MATCH** | HIGH | Action-specific locomotion on a skateboard; absent in VG150. |
| `catch` | *None* | **NO_MATCH** | HIGH | Dynamic interception/capture of an object in motion; absent in VG150. |
| `highfive` | *None* | **NO_MATCH** | HIGH | Interpersonal celebratory palm-slap gesture; absent in VG150. |
| `inside_of` | `in` (ID 22) | **SEMANTIC_EQUIVALENT** | HIGH | Both denote spatial containment where subject is enclosed within the interior boundary of the object in the subject→object direction (e.g. food inside_of bowl vs food in bowl). |
| `eat` | `eating` (ID 14) | **SEMANTIC_EQUIVALENT** | HIGH | Both express oral ingestion of food by subject in the subject→object direction; lexical inflection variation (base form 'eat' vs participle 'eating'). |
| `cut` | *None* | **NO_MATCH** | HIGH | Instrumental material division; absent in VG150. |
| `contain` | `has` (ID 20) | **AMBIGUOUS** | MEDIUM | Direction is container→contained (e.g. bowl contain fruit). Reversing to 'in' (fruit in bowl) violates directionality rules. Mapping to 'has' captures ownership/containment broadly but loses geometric interior enclosure. |
| `handshake` | *None* | **NO_MATCH** | HIGH | Interpersonal mutual hand-clasp greeting; absent in VG150. |
| `kiss` | *None* | **NO_MATCH** | HIGH | Affectionate oral contact; absent in VG150. |
| `talk_on_phone` | `using` (ID 44) | **AMBIGUOUS** | MEDIUM | Telecommunication conversation; mapping to 'using' loses the vocal/communicative component. |
| `interacts_with` | *None* | **NO_MATCH** | HIGH | Unspecified generic interaction umbrella category; no direct semantic equivalent in VG150. |
| `under` | `under` (ID 43) | **EXACT** | HIGH | Exact lexical and semantic match; spatial placement beneath/underneath. |
| `hug` | *None* | **NO_MATCH** | HIGH | Affectionate bodily embrace; absent in VG150. |
| `throw` | *None* | **NO_MATCH** | HIGH | Ballistic propulsion/release; absent in VG150. |
| `hits` | *None* | **NO_MATCH** | HIGH | Dynamic forceful physical impact; absent in VG150. |
| `snowboard` | *None* | **NO_MATCH** | HIGH | Winter-sport locomotion on a snowboard; absent in VG150. |
| `is` | *None* | **NO_MATCH** | HIGH | Entity-to-attribute copula link (e.g. 'Table is Wooden', 'Person is Walk'); in VG150 attributes are stored in object attribute arrays, not as relationship triplets. |
| `kick` | *None* | **NO_MATCH** | HIGH | Impact strike with the foot; absent in VG150. |
| `ski` | *None* | **NO_MATCH** | HIGH | Winter-sport locomotion on skis; absent in VG150. |
| `plays` | `playing` (ID 37) | **SEMANTIC_EQUIVALENT** | HIGH | Both express recreational game/sport or musical instrument performance in the subject→object direction; only grammatical inflection differs (3rd person singular 'plays' vs gerund 'playing'). |
| `read` | `looking at` (ID 25) | **AMBIGUOUS** | MEDIUM | 'read' implies textual comprehension; mapping to 'looking at' discards textual interaction semantics. |

