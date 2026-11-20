/**
 * NLQ Query Executor.
 * Executes typed QueryPlans against Scene Graph facts and IndexedDB Memory.
 */

import { QueryPlan, NLQResult, AnswerEvidence } from './queryTypes';
import { SceneGraph } from '../graph/graphTypes';
import { MemoryRepository } from '../memory/memoryRepository';
import { PersistentObjectInstance } from '../memory/memoryTypes';
import { formatHumanTime } from '../utils/time';
import { GraphReasoner } from '../graph/graphReasoner';

export class QueryExecutor {
  static async executeQuery(
    plan: QueryPlan,
    currentGraph: SceneGraph | null,
    workspaceId: string
  ): Promise<NLQResult> {
    const instances = await MemoryRepository.getInstances(workspaceId);
    const frames = await MemoryRepository.getFrames(workspaceId);

    // Helper for fuzzy category/mention matching
    const matchesTarget = (catOrMention: string | undefined, testStrings: (string | undefined)[]) => {
      if (!catOrMention) return false;
      const target = catOrMention.toLowerCase().trim();
      if (!target) return false;
      return testStrings.some((s) => {
        if (!s) return false;
        const norm = s.toLowerCase().trim();
        return norm === target || norm.includes(target) || target.includes(norm);
      });
    };

    // 1. Locate Current Object
    if (plan.intent === 'locate_current_object') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const targetInstanceId = plan.targetObject?.instanceId;

      // Check current frame first
      if (currentGraph) {
        const matchNode = currentGraph.nodes.find((n) => {
          if (targetInstanceId && n.id === targetInstanceId) return true;
          return matchesTarget(targetCategory, [n.categoryId, n.label]);
        });

        if (matchNode && matchNode.bbox) {
          const neighbors = GraphReasoner.getNeighbors(currentGraph, matchNode.id);
          const relStr = neighbors.length > 0
            ? ` It is ${neighbors[0].relation.replace(/_/g, ' ')} ${neighbors[0].neighbor.label}.`
            : '';

          return {
            queryPlan: plan,
            answer: `The ${matchNode.label} is currently observed at [${matchNode.bbox.join(', ')}].${relStr}`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph.frameId],
              timestamps: [currentGraph.timestamp],
              objectInstanceIds: [matchNode.id],
              relationshipIds: currentGraph.edges.filter((e) => e.sourceId === matchNode.id || e.targetId === matchNode.id).map((e) => e.id),
              confidence: matchNode.confidence,
              explanation: `Direct visual detection in current observation frame ${currentGraph.frameId}.`,
              provisional: false,
            },
            requiresVLM: false,
          };
        }
      }

      // If not observed in current frame, check historical memory
      const matchingInstances = instances.filter((inst) => {
        if (targetInstanceId && inst.instanceId === targetInstanceId) return true;
        return matchesTarget(targetCategory, [inst.categoryId, inst.displayName]);
      });

      if (matchingInstances.length > 0) {
        const inst = matchingInstances[0];
        const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
        const lastObs = obsHistory.filter((o) => o.bbox !== null).pop();

        if (lastObs) {
          return {
            queryPlan: plan,
            answer: `The ${inst.displayName} was not observed in the current frame, but was last seen at ${formatHumanTime(lastObs.timestamp)} in frame ${lastObs.frameId}.`,
            evidence: {
              source: 'workspace_memory',
              workspaceId,
              frameIds: [lastObs.frameId],
              timestamps: [lastObs.timestamp],
              objectInstanceIds: [inst.instanceId],
              relationshipIds: [],
              confidence: lastObs.confidence,
              explanation: `Retrieved last known observation from workspace memory.`,
              provisional: false,
            },
            requiresVLM: false,
          };
        }
      }

      // Not found in memory -> trigger VLM fallback
      return {
        queryPlan: plan,
        answer: `Could not locate ${targetCategory || 'object'} in the current scene graph or memory history.`,
        evidence: {
          source: 'workspace_memory',
          workspaceId,
          frameIds: currentGraph ? [currentGraph.frameId] : [],
          timestamps: currentGraph ? [currentGraph.timestamp] : [],
          objectInstanceIds: [],
          relationshipIds: [],
          explanation: 'Object not found in structured facts.',
          provisional: false,
        },
        requiresVLM: true,
        vlmReason: `Object ${targetCategory} is not detected by SGG; visual fallback required to search RGB image.`,
      };
    }

    // 2. Locate Last Seen Object
    if (plan.intent === 'locate_last_seen_object') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const matchingInstances = instances.filter(
        (inst) => (plan.targetObject?.instanceId && inst.instanceId === plan.targetObject.instanceId) ||
                  matchesTarget(targetCategory, [inst.categoryId, inst.displayName])
      );

      if (matchingInstances.length > 0) {
        const inst = matchingInstances[0];
        const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
        const lastObs = obsHistory.filter((o) => o.bbox !== null).pop();

        if (lastObs) {
          const isCurrent = currentGraph && currentGraph.frameId === lastObs.frameId;
          const statusText = isCurrent ? 'currently observed' : `last observed at ${formatHumanTime(lastObs.timestamp)}`;

          return {
            queryPlan: plan,
            answer: `The ${inst.displayName} was ${statusText} in frame ${lastObs.frameId} at bbox [${lastObs.bbox?.join(', ')}].`,
            evidence: {
              source: isCurrent ? 'scene_graph' : 'workspace_memory',
              workspaceId,
              frameIds: [lastObs.frameId],
              timestamps: [lastObs.timestamp],
              objectInstanceIds: [inst.instanceId],
              relationshipIds: [],
              confidence: lastObs.confidence,
              explanation: `Chronological retrieval from workspace temporal memory.`,
              provisional: false,
            },
            requiresVLM: false,
          };
        }
      }

      return {
        queryPlan: plan,
        answer: `No historical observations found for ${targetCategory || 'target object'}.`,
        evidence: {
          source: 'workspace_memory',
          workspaceId,
          frameIds: [],
          timestamps: [],
          objectInstanceIds: [],
          relationshipIds: [],
          explanation: 'No memory record exists for this object.',
          provisional: false,
        },
        requiresVLM: true,
        vlmReason: `No historical memory for ${targetCategory}; checking visual evidence.`,
      };
    }

    // 3. Find Current Relationship
    if (plan.intent === 'find_current_relationship' && currentGraph) {
      const refCategory = plan.referenceObject?.category || plan.referenceObject?.mention;
      const refInstanceId = plan.referenceObject?.instanceId;

      const refNode = currentGraph.nodes.find(
        (n) => (refInstanceId && n.id === refInstanceId) || matchesTarget(refCategory, [n.categoryId, n.label])
      );

      if (refNode) {
        const matches = GraphReasoner.findTriplets(currentGraph, {
          subjectId: plan.predicate === 'on' && plan.targetObject?.mention === '' ? undefined : refNode.id,
          predicate: plan.predicate,
        });

        // Also check if reference is the object
        const incomingMatches = GraphReasoner.findTriplets(currentGraph, {
          objectId: refNode.id,
          predicate: plan.predicate,
        });

        const allMatches = [...matches, ...incomingMatches];

        if (allMatches.length > 0) {
          const best = allMatches[0];
          const neighbor = best.subjectNode.id === refNode.id ? best.objectNode : best.subjectNode;

          return {
            queryPlan: plan,
            answer: `${neighbor.label} is ${best.edge.predicate.replace(/_/g, ' ')} ${refNode.label} (confidence: ${(best.edge.confidence * 100).toFixed(1)}%).`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph.frameId],
              timestamps: [currentGraph.timestamp],
              objectInstanceIds: [best.subjectNode.id, best.objectNode.id],
              relationshipIds: [best.edge.id],
              confidence: best.edge.confidence,
              explanation: `Found directed relationship fact <${best.subjectNode.label}, ${best.edge.predicate}, ${best.objectNode.label}> in current scene graph.`,
              provisional: false,
            },
            requiresVLM: false,
          };
        }
      }

      return {
        queryPlan: plan,
        answer: `Could not verify relationship in current scene graph.`,
        evidence: {
          source: 'scene_graph',
          workspaceId,
          frameIds: [currentGraph.frameId],
          timestamps: [currentGraph.timestamp],
          objectInstanceIds: refNode ? [refNode.id] : [],
          relationshipIds: [],
          explanation: 'No matching relationship triplet in current graph.',
          provisional: false,
        },
        requiresVLM: true,
        vlmReason: `Requested relation '${plan.predicate}' involving '${refCategory}' is not present in SGG graph; invoking VLM visual reasoning.`,
      };
    }

    // 4. Find Historical Relationship
    if (plan.intent === 'find_historical_relationship') {
      if (frames.length < 2) {
        return {
          queryPlan: plan,
          answer: `No previous observation exists in workspace memory (only 1 observation recorded).`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: frames.map((f) => f.frameId),
            timestamps: frames.map((f) => f.timestamp),
            objectInstanceIds: [],
            relationshipIds: [],
            explanation: 'Insufficient history for previous observation query.',
            provisional: false,
          },
          requiresVLM: false,
        };
      }

      const prevFrame = frames[frames.length - 2];
      const prevRelationships = await MemoryRepository.getRelationshipsForFrame(prevFrame.frameId);
      const prevObservations = await MemoryRepository.getObservationsForFrame(prevFrame.frameId);
      const instancesMap = new Map(instances.map((i) => [i.instanceId, i]));

      const refCategory = plan.referenceObject?.category || plan.referenceObject?.mention;

      const matchingRels = prevRelationships.filter((r) => {
        const subjObs = prevObservations.find((o) => o.instanceId === r.subjectId);
        const objObs = prevObservations.find((o) => o.instanceId === r.objectId);
        const predMatch = !plan.predicate || r.predicate === plan.predicate;
        const subjMatch = (r.subjectId === plan.referenceObject?.instanceId) ||
          matchesTarget(refCategory, [subjObs?.categoryId, instancesMap.get(r.subjectId)?.displayName]);
        const objMatch = (r.objectId === plan.referenceObject?.instanceId) ||
          matchesTarget(refCategory, [objObs?.categoryId, instancesMap.get(r.objectId)?.displayName]);
        return predMatch && (subjMatch || objMatch);
      });

      if (matchingRels.length > 0) {
        const rel = matchingRels[0];
        const isSubj = rel.subjectId === plan.referenceObject?.instanceId ||
          matchesTarget(refCategory, [prevObservations.find((o) => o.instanceId === rel.subjectId)?.categoryId]);
        const otherId = isSubj ? rel.objectId : rel.subjectId;
        const otherName = instancesMap.get(otherId)?.displayName || otherId;

        return {
          queryPlan: plan,
          answer: `In previous observation ${prevFrame.frameId} (${formatHumanTime(prevFrame.timestamp)}), ${otherName} was ${rel.predicate.replace(/_/g, ' ')} the referenced object.`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: [prevFrame.frameId],
            timestamps: [prevFrame.timestamp],
            objectInstanceIds: [rel.subjectId, rel.objectId],
            relationshipIds: [rel.relationshipId],
            confidence: rel.confidence,
            explanation: `Historical relation queried from previous observation frame.`,
            provisional: false,
          },
          requiresVLM: false,
        };
      }

      return {
        queryPlan: plan,
        answer: `No historical relationship matching '${plan.predicate || 'relation'}' found in previous frame ${prevFrame.frameId}.`,
        evidence: {
          source: 'workspace_memory',
          workspaceId,
          frameIds: [prevFrame.frameId],
          timestamps: [prevFrame.timestamp],
          objectInstanceIds: [],
          relationshipIds: [],
          explanation: 'No matching relationship in previous observation.',
          provisional: false,
        },
        requiresVLM: true,
        vlmReason: `Historical relationship unverified in previous frame; invoking VLM fallback on frame ${prevFrame.frameId}.`,
      };
    }

    // 5. Compare Object State (Did X move?)
    if (plan.intent === 'compare_object_state') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const matchingInstances = instances.filter(
        (inst) => (plan.targetObject?.instanceId && inst.instanceId === plan.targetObject.instanceId) ||
                  matchesTarget(targetCategory, [inst.categoryId, inst.displayName])
      );

      if (matchingInstances.length === 0) {
        return {
          queryPlan: plan,
          answer: `Cannot determine if ${targetCategory || 'object'} moved because no observations exist in memory.`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: [],
            timestamps: [],
            objectInstanceIds: [],
            relationshipIds: [],
            explanation: 'No memory record for object.',
            provisional: false,
          },
          requiresVLM: false,
        };
      }

      const inst = matchingInstances[0];
      const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
      const validObs = obsHistory.filter((o) => o.bbox !== null);

      if (validObs.length < 2) {
        return {
          queryPlan: plan,
          answer: `Only 1 observation of ${inst.displayName} exists in memory; cannot compare across observations.`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: validObs.map((o) => o.frameId),
            timestamps: validObs.map((o) => o.timestamp),
            objectInstanceIds: [inst.instanceId],
            relationshipIds: [],
            explanation: 'At least two observations required to detect movement.',
            provisional: false,
          },
          requiresVLM: false,
        };
      }

      const prevObs = validObs[validObs.length - 2];
      const currObs = validObs[validObs.length - 1];

      const c1x = (prevObs.bbox![0] + prevObs.bbox![2]) / 2;
      const c1y = (prevObs.bbox![1] + prevObs.bbox![3]) / 2;
      const c2x = (currObs.bbox![0] + currObs.bbox![2]) / 2;
      const c2y = (currObs.bbox![1] + currObs.bbox![3]) / 2;

      const pixelDist = Math.hypot(c1x - c2x, c1y - c2y);
      const moved = pixelDist > 40; // pixel threshold

      const resultText = moved
        ? `Yes, movement detected! ${inst.displayName} moved ~${Math.round(pixelDist)}px between frame ${prevObs.frameId} and ${currObs.frameId}.`
        : `No meaningful movement detected for ${inst.displayName} between frame ${prevObs.frameId} and ${currObs.frameId} (drift: ${Math.round(pixelDist)}px).`;

      return {
        queryPlan: plan,
        answer: resultText,
        evidence: {
          source: 'workspace_memory',
          workspaceId,
          frameIds: [prevObs.frameId, currObs.frameId],
          timestamps: [prevObs.timestamp, currObs.timestamp],
          objectInstanceIds: [inst.instanceId],
          relationshipIds: [],
          explanation: `Computed centroid displacement between sequential observations in fixed viewpoint.`,
          provisional: false,
        },
        requiresVLM: false,
      };
    }

    // Default Fallback
    return {
      queryPlan: plan,
      answer: `The query could not be resolved from current scene graph facts.`,
      evidence: {
        source: 'workspace_memory',
        workspaceId,
        frameIds: currentGraph ? [currentGraph.frameId] : [],
        timestamps: currentGraph ? [currentGraph.timestamp] : [],
        objectInstanceIds: [],
        relationshipIds: [],
        explanation: 'Query intent unhandled or requires visual reasoning.',
        provisional: false,
      },
      requiresVLM: true,
      vlmReason: 'Query requires visual verification.',
    };
  }
}
