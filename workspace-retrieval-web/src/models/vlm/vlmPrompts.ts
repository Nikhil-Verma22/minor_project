/**
 * VLM Prompt Templates and Grounding Context Builders.
 */

import { SceneGraph } from '../../graph/graphTypes';
import { PersistentObjectInstance } from '../../memory/memoryTypes';

export function buildVlmPrompt(
  userQuery: string,
  currentGraph: SceneGraph | null,
  activeInstances: PersistentObjectInstance[]
): string {
  const objectsSummary = activeInstances
    .map((inst) => `- ID: ${inst.instanceId}, Category: ${inst.categoryId}`)
    .join('\n');

  const knownRelations = currentGraph
    ? currentGraph.edges
        .map((e) => `<${e.sourceId}, ${e.predicate}, ${e.targetId}>`)
        .join(', ')
    : 'None';

  return `You are a visual assistant analyzing a workspace RGB image for the project "Object Retrieval in Dynamic Workspace Environments".

Known Workspace Objects:
${objectsSummary || 'None registered'}

Known Spatial Relations:
${knownRelations}

User Question: "${userQuery}"

Strict Rules:
1. Answer strictly based on the visible objects in the image.
2. Only reference object IDs listed in Known Workspace Objects or the canonical "workspace_surface_001".
3. Do not invent unseen objects, IDs, or locations.
4. If the image lacks sufficient visual evidence to answer confidently, set unresolved to true.
5. Return your response ONLY as valid JSON in this exact structure:
{
  "answer": "<direct natural language answer>",
  "candidate_relations": [
    {
      "subject_id": "<valid subject ID>",
      "predicate_text": "<visual relationship phrase>",
      "object_id": "<valid object ID>",
      "canonical_predicate": "on" | "under" | "next_to" | "in_front_of" | "behind" | "connected_to",
      "confidence": <number between 0 and 1>,
      "evidence": "<brief visual rationale>"
    }
  ],
  "unresolved": false,
  "limitations": []
}`;
}
