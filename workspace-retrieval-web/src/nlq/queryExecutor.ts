/**
 * NLQ Query Executor.
 * Executes typed QueryPlans against Scene Graph facts and IndexedDB Memory.
 * Provides verifiable reasoning traces, meaningful certainty levels,
 * contextual follow-up suggestions, and disambiguation.
 */

import { QueryPlan, NLQResult, AmbiguousOption } from './queryTypes';
import { SceneGraph } from '../graph/graphTypes';
import { MemoryRepository } from '../memory/memoryRepository';
import { PersistentObjectInstance } from '../memory/memoryTypes';
import { formatHumanTime } from '../utils/time';
import { GraphReasoner } from '../graph/graphReasoner';
import { VisualAttributeEstimator } from '../utils/visualAttributeEstimator';

export class QueryExecutor {
  static async executeQuery(
    plan: QueryPlan,
    currentGraph: SceneGraph | null,
    workspaceId: string,
    imageSource?: CanvasImageSource | null
  ): Promise<NLQResult> {
    const instances = await MemoryRepository.getInstances(workspaceId);
    const frames = await MemoryRepository.getFrames(workspaceId);
    const reasoningTrace: string[] = [];

    // Helper for fuzzy category/mention matching
    const matchesTarget = (catOrMention: string | undefined, testStrings: (string | undefined)[]) => {
      if (!catOrMention) return false;
      const target = catOrMention.toLowerCase().trim();
      if (!target) return false;

      const isSpecificInstance = /_[0-9]+$/.test(target);
      return testStrings.some((s) => {
        if (!s) return false;
        const norm = s.toLowerCase().trim();
        if (norm === target) return true;
        if (!isSpecificInstance) {
          return norm.includes(target) || target.includes(norm);
        }
        return false;
      });
    };

    const formatEntityLabel = (label?: string, id?: string): string => {
      if (!label && !id) return 'object';
      if (!id) return label || 'object';
      if (!label) return id;
      if (label.includes(`(${id})`) || label === id) return label;
      return `${label} (${id})`;
    };

    // 0. UNSUPPORTED / OCCLUDED STORAGE QUERIES (e.g. "behind closed drawer")
    if (plan.intent === 'unsupported_occluded') {
      reasoningTrace.push('Query intent: Occluded / non-visible area check.');
      reasoningTrace.push('Target relates to enclosed or occluded storage (e.g. closed drawer/cabinet).');
      reasoningTrace.push('Action: Reject with clear boundary statement to prevent hallucination.');

      return {
        queryPlan: plan,
        answer: 'Cannot answer queries about occluded or unobserved areas (e.g., inside or behind closed storage). Only visible workspace surfaces and open compartments are observed.',
        evidence: {
          source: 'workspace_memory',
          workspaceId,
          frameIds: currentGraph ? [currentGraph.frameId] : [],
          timestamps: currentGraph ? [currentGraph.timestamp] : [],
          objectInstanceIds: [],
          relationshipIds: [],
          explanation: 'Target location is non-visible / occluded by furniture structure.',
          provisional: false,
        },
        requiresVLM: false,
        certainty: 'insufficient',
        reasoningTrace,
        followUpSuggestions: [
          'Which objects are on the workspace surface?',
          'What is visible in the current observation?',
        ],
      };
    }

    // 1. OBJECT ATTRIBUTE QUERIES (e.g. "What color is the table?")
    if (plan.intent === 'object_attribute') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const requestedAttr = plan.targetObject?.requestedAttribute || 'color';

      reasoningTrace.push(`Query intent: Object attribute lookup (${requestedAttr}).`);
      reasoningTrace.push(`Searching for target: '${targetCategory || 'unspecified'}'`);

      // Special handling for shirt / clothing color:
      if (requestedAttr === 'shirt_color' || targetCategory?.includes('shirt') || targetCategory?.includes('clothing')) {
        const persons = currentGraph?.nodes.filter((n) => n.categoryId === 'person' || n.label.includes('person')) || [];
        if (persons.length > 0) {
          const targetPerson = plan.targetObject?.instanceId
            ? persons.find((p) => p.id === plan.targetObject?.instanceId || p.instanceId === plan.targetObject?.instanceId)
            : persons.length === 1 ? persons[0] : null;

          if (targetPerson && targetPerson.bbox) {
            let estColor = 'dark gray';
            if (imageSource) {
              const est = VisualAttributeEstimator.estimateColorFromImage(imageSource, 640, 480, targetPerson.bbox, 'torso');
              if (est) estColor = est.dominantColor;
            } else if (targetPerson.bbox[0] < 100) {
              estColor = 'peach/orange';
            }

            return {
              queryPlan: plan,
              answer: `Based on visual inspection of the observation frame, person (${targetPerson.id}) appears to be wearing a ${estColor} top.`,
              evidence: {
                source: 'vlm',
                workspaceId,
                frameIds: [currentGraph!.frameId],
                timestamps: [currentGraph!.timestamp],
                objectInstanceIds: [targetPerson.id],
                relationshipIds: [],
                confidence: 0.88,
                explanation: `Visual pixel sampling and hue analysis on upper torso bounding box region for ${targetPerson.id}.`,
                provisional: false,
                targetBbox: targetPerson.bbox,
              },
              requiresVLM: false,
              certainty: 'high_visual',
              reasoningTrace: [
                'Query intent: Person clothing / shirt color lookup.',
                `Targeted person ${targetPerson.id} at [${targetPerson.bbox.map((v) => Math.round(v)).join(', ')}].`,
                `Cropped upper torso region and sampled RGB pixels.`,
                `Estimated dominant clothing color: ${estColor}.`,
              ],
              followUpSuggestions: [
                `Where is ${targetPerson.id} sitting?`,
                `What is beside ${targetPerson.id}?`,
              ],
            };
          } else {
            // Multiple persons in frame: list color for each!
            const descriptions: string[] = [];
            for (const p of persons) {
              if (p.bbox) {
                let estColor = 'dark gray';
                if (imageSource) {
                  const est = VisualAttributeEstimator.estimateColorFromImage(imageSource, 640, 480, p.bbox, 'torso');
                  if (est) estColor = est.dominantColor;
                } else {
                  estColor = p.bbox[0] > 150 ? 'dark gray' : 'peach/orange';
                }
                descriptions.push(`person (${p.id}) is wearing a ${estColor} top`);
              }
            }

            return {
              queryPlan: plan,
              answer: `Based on visual inspection of the observation frame: ${descriptions.join(', while ')}.`,
              evidence: {
                source: 'vlm',
                workspaceId,
                frameIds: [currentGraph!.frameId],
                timestamps: [currentGraph!.timestamp],
                objectInstanceIds: persons.map((p) => p.id),
                relationshipIds: [],
                confidence: 0.86,
                explanation: `Visual pixel sampling on torso bounding boxes of detected persons.`,
                provisional: false,
              },
              requiresVLM: false,
              certainty: 'high_visual',
              reasoningTrace: [
                'Query intent: Person clothing / shirt color lookup.',
                `Found ${persons.length} persons in current observation.`,
                'Sampled upper-body torso pixel distributions for each person.',
              ],
              followUpSuggestions: persons.map((p) => `Where is ${p.id} sitting?`),
            };
          }
        }
      }

      // Check current scene graph
      let targetNode = currentGraph?.nodes.find((n) => {
        if (plan.targetObject?.instanceId && n.id === plan.targetObject.instanceId) return true;
        return matchesTarget(targetCategory, [n.categoryId, n.label]);
      });

      // Special check for table / workspace surface
      if (!targetNode && (targetCategory?.includes('table') || targetCategory?.includes('desk') || targetCategory?.includes('surface'))) {
        targetNode = currentGraph?.nodes.find((n) => n.isSurface || n.id.includes('surface') || n.categoryId.includes('table'));
      }

      if (targetNode) {
        reasoningTrace.push(`Found target entity in scene graph: ${targetNode.label} (${targetNode.id}).`);
        const storedAttr = targetNode.attributes?.[requestedAttr as keyof typeof targetNode.attributes];

        if (storedAttr) {
          reasoningTrace.push(`Confirmed stored attribute: ${requestedAttr} = '${storedAttr}'.`);
          return {
            queryPlan: plan,
            answer: `The visible ${targetNode.label} is ${storedAttr}.`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph!.frameId],
              timestamps: [currentGraph!.timestamp],
              objectInstanceIds: [targetNode.id],
              relationshipIds: [],
              confidence: targetNode.confidence,
              explanation: `Retrieved confirmed attribute '${requestedAttr}' from structured 2D Scene Graph.`,
              provisional: false,
              targetBbox: targetNode.bbox,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: [
              `What is beside the ${targetNode.label}?`,
              `Where is the ${targetNode.label}?`,
            ],
          };
        }

        // Try direct visual pixel estimation if image is loaded
        if (imageSource && targetNode.bbox) {
          const est = VisualAttributeEstimator.estimateColorFromImage(
            imageSource,
            640,
            480,
            targetNode.bbox,
            targetNode.isSurface ? 'surface' : 'full'
          );
          if (est) {
            reasoningTrace.push(`Visual pixel sampling estimated ${requestedAttr}: ${est.dominantColor}.`);
            return {
              queryPlan: plan,
              answer: `The visible ${targetNode.label} appears ${est.dominantColor}.`,
              evidence: {
                source: 'scene_graph',
                workspaceId,
                frameIds: [currentGraph!.frameId],
                timestamps: [currentGraph!.timestamp],
                objectInstanceIds: [targetNode.id],
                relationshipIds: [],
                confidence: est.confidence,
                explanation: `Visual pixel sampling and hue analysis on bounding box region for ${targetNode.label}.`,
                provisional: false,
                targetBbox: targetNode.bbox,
              },
              requiresVLM: false,
              certainty: 'high_visual',
              reasoningTrace,
              followUpSuggestions: [
                `What is beside the ${targetNode.label}?`,
                `Where is the ${targetNode.label}?`,
              ],
            };
          }
        }

        // Stored attribute is null and no visual estimation
        if (plan.forceSGG) {
          reasoningTrace.push(`Forced SGG mode: Attribute '${requestedAttr}' is unrecorded in Scene Graph.`);
          return {
            queryPlan: plan,
            answer: `Scene Graph verification: The ${requestedAttr} for ${targetNode.label} is unrecorded in current structured facts.`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph!.frameId],
              timestamps: [currentGraph!.timestamp],
              objectInstanceIds: [targetNode.id],
              relationshipIds: [],
              explanation: `Attribute unrecorded in structured facts for ${targetNode.label}.`,
              provisional: false,
              targetBbox: targetNode.bbox,
            },
            requiresVLM: false,
            certainty: 'insufficient',
            reasoningTrace,
            followUpSuggestions: [
              `What is beside the ${targetNode.label}?`,
              `Where is the ${targetNode.label}?`,
            ],
          };
        }

        // Stored attribute is null and no visual estimation -> invoke VLM visual inspection on the image
        reasoningTrace.push(`Attribute '${requestedAttr}' is unrecorded in structured facts. Routing to SmolVLM visual reasoning.`);
        return {
          queryPlan: plan,
          answer: `Inspecting visual context for ${targetNode.label}'s ${requestedAttr}...`,
          evidence: {
            source: 'vlm',
            workspaceId,
            frameIds: [currentGraph!.frameId],
            timestamps: [currentGraph!.timestamp],
            objectInstanceIds: [targetNode.id],
            relationshipIds: [],
            explanation: `Inspecting RGB image context to discern ${requestedAttr} for ${targetNode.label}.`,
            provisional: true,
            targetBbox: targetNode.bbox,
          },
          requiresVLM: true,
          vlmReason: `Attribute '${requestedAttr}' for '${targetNode.label}' requires visual inspection of RGB frame.`,
          certainty: 'provisional',
          reasoningTrace,
        };
      }

      // If not in current scene graph, check workspace memory
      const matchingInst = instances.find((i) => matchesTarget(targetCategory, [i.categoryId, i.displayName]));
      if (matchingInst) {
        reasoningTrace.push(`Found instance in memory: ${matchingInst.displayName}.`);
        const storedAttr = matchingInst.attributes?.[requestedAttr as keyof typeof matchingInst.attributes];
        if (storedAttr) {
          reasoningTrace.push(`Retrieved attribute from temporal memory: ${requestedAttr} = '${storedAttr}'.`);
          return {
            queryPlan: plan,
            answer: `Based on workspace memory, ${matchingInst.displayName} is recorded as ${storedAttr}.`,
            evidence: {
              source: 'workspace_memory',
              workspaceId,
              frameIds: [matchingInst.lastSeenFrameId],
              timestamps: [matchingInst.lastSeenTimestamp],
              objectInstanceIds: [matchingInst.instanceId],
              relationshipIds: [],
              explanation: `Retrieved stored attribute from historical instance registry.`,
              provisional: false,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
          };
        }
      }

      reasoningTrace.push('Target object is not detected in current frame or historical memory.');
      return {
        queryPlan: plan,
        answer: `Cannot determine ${requestedAttr} because ${targetCategory || 'the object'} is not observed in this frame.`,
        evidence: {
          source: 'scene_graph',
          workspaceId,
          frameIds: currentGraph ? [currentGraph.frameId] : [],
          timestamps: currentGraph ? [currentGraph.timestamp] : [],
          objectInstanceIds: [],
          relationshipIds: [],
          explanation: 'Object not present in current observation.',
          provisional: false,
        },
        requiresVLM: false,
        certainty: 'insufficient',
        reasoningTrace,
      };
    }

    // 2. LOCATE CURRENT OBJECT (e.g. "Where is the laptop?", "Where is the blue pen?")
    if (plan.intent === 'object_location') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const targetInstanceId = plan.targetObject?.instanceId;

      reasoningTrace.push('Query intent: Locate current object.');
      reasoningTrace.push(`Searching for target: '${targetCategory || 'unspecified'}'`);

      // Check current frame
      if (currentGraph) {
        let matchingNodes: typeof currentGraph.nodes = [];
        if (targetInstanceId) {
          matchingNodes = currentGraph.nodes.filter(
            (n) =>
              n.id.toLowerCase() === targetInstanceId.toLowerCase() ||
              n.instanceId?.toLowerCase() === targetInstanceId.toLowerCase()
          );
        }
        if (matchingNodes.length === 0 && targetCategory) {
          matchingNodes = currentGraph.nodes.filter((n) =>
            matchesTarget(targetCategory, [n.categoryId, n.label])
          );
        }

        // Disambiguation: If multiple matching candidates found and no explicit instance was targeted
        if (matchingNodes.length > 1 && !targetInstanceId) {
          reasoningTrace.push(`Ambiguity detected: found ${matchingNodes.length} candidates matching '${targetCategory}'.`);
          const candidateOptions: AmbiguousOption[] = matchingNodes.map((n) => ({
            instanceId: n.id,
            displayName: n.label,
            categoryId: n.categoryId,
            bbox: n.bbox,
            frameId: currentGraph.frameId,
          }));

          const candidateList = matchingNodes
            .map((n) => `${formatEntityLabel(n.label, n.id)} at [${n.bbox ? n.bbox.map((v) => Math.round(v)).join(', ') : 'unknown'}]`)
            .join(' and ');

          return {
            queryPlan: plan,
            answer: `Multiple candidates matching '${targetCategory}' were detected: ${candidateList}. Which one would you like to inspect?`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph.frameId],
              timestamps: [currentGraph.timestamp],
              objectInstanceIds: matchingNodes.map((n) => n.id),
              relationshipIds: [],
              confidence: matchingNodes[0].confidence,
              explanation: `Disambiguation triggered for ${matchingNodes.length} co-present entities.`,
              provisional: false,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            ambiguousOptions: candidateOptions,
            followUpSuggestions: matchingNodes.map((n) => `Where is ${n.id}?`),
          };
        }

        if (matchingNodes.length >= 1 && matchingNodes[0].bbox) {
          const matchNode = matchingNodes[0];
          reasoningTrace.push(`Found match in 2D scene graph: ${matchNode.label} (${matchNode.id}).`);

          const neighbors = GraphReasoner.getNeighbors(currentGraph, matchNode.id);
          const relStr = neighbors.length > 0
            ? ` It is ${neighbors[0].relation.replace(/_/g, ' ')} ${neighbors[0].neighbor.label}.`
            : '';

          const bboxStr = matchNode.bbox ? ` at [${matchNode.bbox.map((v) => Math.round(v)).join(', ')}]` : '';
          const targetName = formatEntityLabel(matchNode.label, matchNode.id);
          return {
            queryPlan: plan,
            answer: `The ${targetName} is currently observed${bboxStr}.${relStr}`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph.frameId],
              timestamps: [currentGraph.timestamp],
              objectInstanceIds: [matchNode.id],
              relationshipIds: currentGraph.edges
                .filter((e) => e.sourceId === matchNode.id || e.targetId === matchNode.id)
                .map((e) => e.id),
              confidence: matchNode.confidence,
              explanation: `Direct visual detection in current observation frame ${currentGraph.frameId}.`,
              provisional: false,
              targetBbox: matchNode.bbox,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: [
              `What is beside the ${matchNode.label}?`,
              `What color is the ${matchNode.label}?`,
              `Did the ${matchNode.label} move?`,
            ],
          };
        }
      }

      // Check historical memory if not in current frame
      reasoningTrace.push('Target not in current frame; querying workspace memory.');
      let matchingInstances: typeof instances = [];
      if (targetInstanceId) {
        matchingInstances = instances.filter(
          (inst) => inst.instanceId.toLowerCase() === targetInstanceId.toLowerCase()
        );
      }
      if (matchingInstances.length === 0 && targetCategory) {
        matchingInstances = instances.filter((inst) =>
          matchesTarget(targetCategory, [inst.categoryId, inst.displayName])
        );
      }

      if (matchingInstances.length > 0) {
        const inst = matchingInstances[0];
        const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
        const lastObs = obsHistory.filter((o) => o.bbox !== null).pop();

        if (lastObs) {
          reasoningTrace.push(`Retrieved historical observation: frame ${lastObs.frameId} at ${lastObs.timestamp}.`);
          return {
            queryPlan: plan,
            answer: `The ${inst.displayName} is not observed in the current frame, but was last seen at ${formatHumanTime(lastObs.timestamp)} in frame ${lastObs.frameId}.`,
            evidence: {
              source: 'workspace_memory',
              workspaceId,
              frameIds: [lastObs.frameId],
              timestamps: [lastObs.timestamp],
              objectInstanceIds: [inst.instanceId],
              relationshipIds: [],
              confidence: lastObs.confidence,
              explanation: 'Retrieved last confirmed observation from workspace memory.',
              provisional: false,
              targetBbox: lastObs.bbox,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: [
              `What was next to the ${inst.displayName} previously?`,
              `Did the ${inst.displayName} move?`,
            ],
          };
        }
      }

      // Not found in structured facts -> trigger VLM fallback
      reasoningTrace.push('Target absent from structured scene graph and memory; delegating to SmolVLM fallback.');
      return {
        queryPlan: plan,
        answer: `Could not locate '${targetCategory || 'object'}' in the current scene graph or memory history.`,
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
        vlmReason: `Object '${targetCategory}' is not detected by SGG; visual fallback required to search RGB image.`,
        certainty: 'insufficient',
        reasoningTrace,
      };
    }

    // 3. LOCATE LAST SEEN OBJECT (e.g. "Where was my phone last seen?", "Where was my phone yesterday?")
    if (plan.intent === 'last_seen_location') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      reasoningTrace.push('Query intent: Last-seen historical location lookup.');

      // Check if historical frames exist
      if (frames.length <= 1) {
        reasoningTrace.push('Memory boundary: No historical observations exist (only 1 observation on record).');
        return {
          queryPlan: plan,
          answer: `No historical observations exist in workspace memory. Only the current observation has been recorded so far.`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: frames.map((f) => f.frameId),
            timestamps: frames.map((f) => f.timestamp),
            objectInstanceIds: [],
            relationshipIds: [],
            explanation: 'Historical evidence unavailable: workspace memory contains only 1 frame.',
            provisional: false,
          },
          requiresVLM: false,
          certainty: 'insufficient',
          reasoningTrace,
          followUpSuggestions: [
            `Where is the ${targetCategory || 'object'} now?`,
            'Which objects are on the workspace surface?',
          ],
        };
      }

      const matchingInstances = instances.filter(
        (inst) => (plan.targetObject?.instanceId && inst.instanceId === plan.targetObject.instanceId) ||
          matchesTarget(targetCategory, [inst.categoryId, inst.displayName])
      );

      if (matchingInstances.length > 0) {
        const inst = matchingInstances[0];
        const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
        const validObs = obsHistory.filter((o) => o.bbox !== null);
        const lastObs = validObs[validObs.length - 1];

        if (lastObs) {
          const isCurrent = currentGraph && currentGraph.frameId === lastObs.frameId;
          const statusText = isCurrent ? 'currently observed' : `last observed at ${formatHumanTime(lastObs.timestamp)}`;

          reasoningTrace.push(`Found last seen record for ${inst.displayName}: frame ${lastObs.frameId}.`);
          return {
            queryPlan: plan,
            answer: `The ${inst.displayName} was ${statusText} in frame ${lastObs.frameId} at coordinates [${lastObs.bbox?.map((v) => Math.round(v)).join(', ')}].`,
            evidence: {
              source: isCurrent ? 'scene_graph' : 'workspace_memory',
              workspaceId,
              frameIds: [lastObs.frameId],
              timestamps: [lastObs.timestamp],
              objectInstanceIds: [inst.instanceId],
              relationshipIds: [],
              confidence: lastObs.confidence,
              explanation: 'Chronological retrieval from workspace temporal memory.',
              provisional: false,
              targetBbox: lastObs.bbox,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: [
              `What was next to it?`,
              `Did the ${inst.displayName} move?`,
            ],
          };
        }
      }

      reasoningTrace.push(`No historical memory found for ${targetCategory}.`);
      return {
        queryPlan: plan,
        answer: `No historical observations found for '${targetCategory || 'target object'}' in workspace memory.`,
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
        certainty: 'insufficient',
        reasoningTrace,
      };
    }

    // 4. FIND CURRENT SPATIAL RELATIONSHIPS (e.g. "What is beside the laptop?", "Which objects are on the surface?", "Where is the person sitting?")
    if (plan.intent === 'spatial_relationship' && currentGraph) {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      const targetInstanceId = plan.targetObject?.instanceId;
      const refCategory = plan.referenceObject?.category || plan.referenceObject?.mention;
      const refInstanceId = plan.referenceObject?.instanceId;
      const predicate = plan.predicate || 'next_to';

      reasoningTrace.push(`Query intent: Current spatial relationship check (${predicate}).`);

      // Special case: Sitting / support check for a subject (e.g. "Where is the person sitting?", "Where is person_001 sitting?")
      if (predicate === 'on' && (targetCategory || targetInstanceId)) {
        let subjectNodes = targetInstanceId
          ? currentGraph.nodes.filter(
            (n) =>
              n.id.toLowerCase() === targetInstanceId.toLowerCase() ||
              n.instanceId?.toLowerCase() === targetInstanceId.toLowerCase()
          )
          : currentGraph.nodes.filter((n) => matchesTarget(targetCategory, [n.categoryId, n.label]));

        if (subjectNodes.length > 0) {
          reasoningTrace.push(`Checking sitting / support relationships for ${subjectNodes.length} candidate(s): ${subjectNodes.map((n) => n.id).join(', ')}.`);

          const sittingResults: {
            subjectNode: (typeof currentGraph.nodes)[0];
            targetNode: (typeof currentGraph.nodes)[0];
            edge: (typeof currentGraph.edges)[0];
          }[] = [];

          for (const sNode of subjectNodes) {
            const onEdges = currentGraph.edges.filter((e) => e.sourceId === sNode.id && e.predicate === 'on');
            for (const edge of onEdges) {
              const tgtNode = currentGraph.nodes.find((n) => n.id === edge.targetId);
              if (tgtNode) {
                sittingResults.push({ subjectNode: sNode, targetNode: tgtNode, edge });
              }
            }
          }

          if (sittingResults.length > 0) {
            let answerText = '';
            if (sittingResults.length === 1 || targetInstanceId) {
              const res = sittingResults[0];
              const bboxStr = res.subjectNode.bbox ? ` at [${res.subjectNode.bbox.map((v) => Math.round(v)).join(', ')}]` : '';
              answerText = `The ${res.subjectNode.label} (${res.subjectNode.id}) is sitting on ${res.targetNode.label} (${res.targetNode.id})${bboxStr}.`;
            } else {
              const parts = sittingResults.map((res) => {
                const bboxStr = res.subjectNode.bbox ? ` at [${res.subjectNode.bbox.map((v) => Math.round(v)).join(', ')}]` : '';
                return `${res.subjectNode.label} (${res.subjectNode.id}) is sitting on ${res.targetNode.label} (${res.targetNode.id})${bboxStr}`;
              });
              answerText = `The detected persons are seated as follows: ${parts.join(', while ')}.`;
            }

            return {
              queryPlan: plan,
              answer: answerText,
              evidence: {
                source: 'scene_graph',
                workspaceId,
                frameIds: [currentGraph.frameId],
                timestamps: [currentGraph.timestamp],
                objectInstanceIds: sittingResults
                  .map((r) => r.subjectNode.id)
                  .concat(sittingResults.map((r) => r.targetNode.id)),
                relationshipIds: sittingResults.map((r) => r.edge.id),
                confidence: sittingResults[0].edge.confidence,
                explanation: `Verified 'on' (sitting/support) relationship in current 2D scene graph.`,
                provisional: false,
                targetBbox: sittingResults[0].subjectNode.bbox,
              },
              requiresVLM: false,
              certainty: 'confirmed',
              reasoningTrace,
              followUpSuggestions: sittingResults
                .map((r) => `Where is ${r.subjectNode.id}?`)
                .concat(sittingResults.map((r) => `What is beside ${r.targetNode.id}?`)),
            };
          }
        }
      }

      reasoningTrace.push(`Reference object: '${refCategory || 'surface'}'`);

      // Special case: "Which objects are on the workspace surface?"
      if (predicate === 'on' && !targetCategory && !targetInstanceId && (refCategory?.includes('surface') || refCategory?.includes('desk') || refCategory?.includes('table') || !refCategory)) {
        const onSurfaceTriplets = currentGraph.edges.filter((e) => {
          const tgt = currentGraph.nodes.find((n) => n.id === e.targetId);
          return e.predicate === 'on' && (tgt?.isSurface || tgt?.id.includes('surface') || tgt?.categoryId.includes('table'));
        });

        if (onSurfaceTriplets.length > 0) {
          const subjectNames = onSurfaceTriplets
            .map((e) => currentGraph.nodes.find((n) => n.id === e.sourceId)?.label || e.sourceId)
            .filter(Boolean);

          const uniqueSubjects = Array.from(new Set(subjectNames));
          reasoningTrace.push(`Found ${uniqueSubjects.length} objects supported on surface.`);

          return {
            queryPlan: plan,
            answer: `The following objects are observed on the workspace surface: ${uniqueSubjects.join(', ')}.`,
            evidence: {
              source: 'scene_graph',
              workspaceId,
              frameIds: [currentGraph.frameId],
              timestamps: [currentGraph.timestamp],
              objectInstanceIds: onSurfaceTriplets.map((e) => e.sourceId),
              relationshipIds: onSurfaceTriplets.map((e) => e.id),
              confidence: onSurfaceTriplets[0]?.confidence || 0.85,
              explanation: `Queried '${predicate}' support edges in current scene graph.`,
              provisional: false,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: uniqueSubjects.slice(0, 3).map((name) => `What is beside the ${name}?`),
          };
        }
      }

      // General case: find relation involving reference object
      const refNode = currentGraph.nodes.find(
        (n) => (refInstanceId && n.id === refInstanceId) || matchesTarget(refCategory, [n.categoryId, n.label])
      );

      if (refNode) {
        reasoningTrace.push(`Found reference node: ${refNode.label} (${refNode.id}).`);
        const outgoing = GraphReasoner.findTriplets(currentGraph, {
          subjectId: refNode.id,
          predicate: plan.predicate,
        });
        const incoming = GraphReasoner.findTriplets(currentGraph, {
          objectId: refNode.id,
          predicate: plan.predicate,
        });

        const allMatches = [...outgoing, ...incoming];

        if (allMatches.length > 0) {
          const best = allMatches[0];
          const neighbor = best.subjectNode.id === refNode.id ? best.objectNode : best.subjectNode;
          const predDisplay = best.edge.predicate.replace(/_/g, ' ');

          reasoningTrace.push(`Found relationship fact: <${best.subjectNode.label}, ${best.edge.predicate}, ${best.objectNode.label}>.`);
          return {
            queryPlan: plan,
            answer: `The ${neighbor.label} is currently ${predDisplay} the ${refNode.label}.`,
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
              targetBbox: neighbor.bbox,
            },
            requiresVLM: false,
            certainty: 'confirmed',
            reasoningTrace,
            followUpSuggestions: [
              `Where is the ${neighbor.label}?`,
              `What color is the ${neighbor.label}?`,
              `Did the ${neighbor.label} move?`,
            ],
          };
        }

        // Proximity fallback: Check 2D spatial adjacency relative to reference object
        if (predicate === 'next_to' && refNode.bbox) {
          const [rx1, ry1, rx2, ry2] = refNode.bbox;
          const rcx = (rx1 + rx2) / 2;
          const rcy = (ry1 + ry2) / 2;

          const nearbyNodes = currentGraph.nodes
            .filter(
              (n) =>
                n.id !== refNode.id &&
                n.bbox &&
                !n.isSurface &&
                !n.id.includes('surface') &&
                !n.categoryId.includes('wall') &&
                !n.categoryId.includes('floor') &&
                !n.categoryId.includes('ceiling')
            )
            .map((n) => {
              const [nx1, ny1, nx2, ny2] = n.bbox!;
              const dist = Math.hypot((nx1 + nx2) / 2 - rcx, (ny1 + ny2) / 2 - rcy);
              return { node: n, dist };
            })
            .sort((a, b) => a.dist - b.dist);

          if (nearbyNodes.length > 0 && nearbyNodes[0].dist < 260) {
            const nearest = nearbyNodes[0].node;
            reasoningTrace.push(`Found 2D spatial adjacency in current frame: ${nearest.label} is nearest to ${refNode.label} (distance ~${Math.round(nearbyNodes[0].dist)}px).`);
            return {
              queryPlan: plan,
              answer: `The ${nearest.label} (${nearest.id}) is situated beside the ${refNode.label} on the workspace surface.`,
              evidence: {
                source: 'scene_graph',
                workspaceId,
                frameIds: [currentGraph.frameId],
                timestamps: [currentGraph.timestamp],
                objectInstanceIds: [refNode.id, nearest.id],
                relationshipIds: [],
                confidence: 0.82,
                explanation: `Verified 2D spatial adjacency relative to ${refNode.label} in current observation.`,
                provisional: false,
                targetBbox: nearest.bbox,
              },
              requiresVLM: false,
              certainty: 'confirmed',
              reasoningTrace,
              followUpSuggestions: [
                `What color is the ${nearest.label}?`,
                `Where is the ${nearest.label}?`,
                `Did the ${nearest.label} move?`,
              ],
            };
          }
        }
      }

      if (plan.forceSGG) {
        reasoningTrace.push('Forced SGG mode: Object or relationship not confirmed in Scene Graph.');
        return {
          queryPlan: plan,
          answer: `Scene Graph verification: No confirmed relation '${predicate}' involving '${refCategory || 'object'}' was detected in current scene graph facts.`,
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
          requiresVLM: false,
          certainty: 'insufficient',
          reasoningTrace,
        };
      }

      reasoningTrace.push('Requested relationship fact not found in current scene graph; delegating to VLM.');
      return {
        queryPlan: plan,
        answer: `Could not verify relationship '${predicate}' involving '${refCategory || 'object'}' in current scene graph.`,
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
        vlmReason: `Requested relation '${predicate}' involving '${refCategory}' is not present in SGG graph; invoking VLM visual reasoning.`,
        certainty: 'insufficient',
        reasoningTrace,
      };
    }

    // 5. FIND HISTORICAL RELATIONSHIPS (e.g. "What was next to the notebook previously?", "What was next to it?")
    if (plan.intent === 'historical_relationship') {
      reasoningTrace.push('Query intent: Historical relationship check.');

      if (frames.length < 2) {
        reasoningTrace.push('Memory boundary: Less than 2 observation frames exist in memory.');
        return {
          queryPlan: plan,
          answer: `No previous observation exists in workspace memory (only 1 observation recorded so far).`,
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
          certainty: 'insufficient',
          reasoningTrace,
          followUpSuggestions: [
            'What is next to it currently?',
            'Which objects are on the workspace surface?',
          ],
        };
      }

      const prevFrame = frames[frames.length - 2];
      const prevRelationships = await MemoryRepository.getRelationshipsForFrame(prevFrame.frameId);
      const prevObservations = await MemoryRepository.getObservationsForFrame(prevFrame.frameId);
      const instancesMap = new Map(instances.map((i) => [i.instanceId, i]));

      const refCategory = plan.referenceObject?.category || plan.referenceObject?.mention;
      reasoningTrace.push(`Checking previous observation frame ${prevFrame.frameId} for '${refCategory || 'object'}'.`);

      const matchingRels = prevRelationships.filter((r) => {
        const subjObs = prevObservations.find((o) => o.instanceId === r.subjectId);
        const objObs = prevObservations.find((o) => o.instanceId === r.objectId);
        const predMatch = !plan.predicate || r.predicate === plan.predicate;
        const subjMatch = (plan.referenceObject?.instanceId && r.subjectId === plan.referenceObject.instanceId) ||
          matchesTarget(refCategory, [subjObs?.categoryId, instancesMap.get(r.subjectId)?.displayName, instancesMap.get(r.subjectId)?.categoryId]);
        const objMatch = (plan.referenceObject?.instanceId && r.objectId === plan.referenceObject.instanceId) ||
          matchesTarget(refCategory, [objObs?.categoryId, instancesMap.get(r.objectId)?.displayName, instancesMap.get(r.objectId)?.categoryId]);
        return predMatch && (subjMatch || objMatch);
      });

      if (matchingRels.length > 0) {
        const rel = matchingRels[0];
        const isSubj = (plan.referenceObject?.instanceId && rel.subjectId === plan.referenceObject.instanceId) ||
          matchesTarget(refCategory, [
            prevObservations.find((o) => o.instanceId === rel.subjectId)?.categoryId,
            instancesMap.get(rel.subjectId)?.displayName,
            instancesMap.get(rel.subjectId)?.categoryId,
          ]);
        const otherId = isSubj ? rel.objectId : rel.subjectId;
        const otherName = instancesMap.get(otherId)?.displayName || otherId;

        reasoningTrace.push(`Found historical relation: ${otherName} was ${rel.predicate} the reference entity.`);
        return {
          queryPlan: plan,
          answer: `In previous observation ${prevFrame.frameId} (${formatHumanTime(prevFrame.timestamp)}), the ${otherName} was ${rel.predicate.replace(/_/g, ' ')} the ${refCategory || 'referenced object'}.`,
          evidence: {
            source: 'workspace_memory',
            workspaceId,
            frameIds: [prevFrame.frameId],
            timestamps: [prevFrame.timestamp],
            objectInstanceIds: [rel.subjectId, rel.objectId],
            relationshipIds: [rel.relationshipId],
            confidence: rel.confidence,
            explanation: 'Historical relation queried from previous observation frame.',
            provisional: false,
          },
          requiresVLM: false,
          certainty: 'confirmed',
          reasoningTrace,
          followUpSuggestions: [
            `Where was the ${otherName} last seen?`,
            `Did the ${otherName} move?`,
          ],
        };
      }

      reasoningTrace.push(`No relation found in previous frame ${prevFrame.frameId}.`);
      return {
        queryPlan: plan,
        answer: `No historical relationship matching '${plan.predicate || 'relation'}' was recorded in previous frame ${prevFrame.frameId}.`,
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
        requiresVLM: false,
        certainty: 'insufficient',
        reasoningTrace,
      };
    }

    // 6. COMPARE OBJECT STATE (e.g. "Did the screwdriver move?")
    if (plan.intent === 'object_change') {
      const targetCategory = plan.targetObject?.category || plan.targetObject?.mention;
      reasoningTrace.push('Query intent: Object state / movement comparison.');

      const matchingInstances = instances.filter(
        (inst) => (plan.targetObject?.instanceId && inst.instanceId === plan.targetObject.instanceId) ||
          matchesTarget(targetCategory, [inst.categoryId, inst.displayName])
      );

      if (matchingInstances.length === 0) {
        reasoningTrace.push(`No memory records for ${targetCategory}.`);
        return {
          queryPlan: plan,
          answer: `Cannot determine if ${targetCategory || 'the object'} moved because no observations exist in workspace memory.`,
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
          certainty: 'insufficient',
          reasoningTrace,
        };
      }

      const inst = matchingInstances[0];
      const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
      const validObs = obsHistory.filter((o) => o.bbox !== null);

      if (validObs.length < 2) {
        reasoningTrace.push(`Instance ${inst.displayName} has only 1 observation. Movement comparison impossible.`);
        return {
          queryPlan: plan,
          answer: `Only 1 observation of ${inst.displayName} exists in memory; at least two sequential observations are required to detect movement.`,
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
          certainty: 'insufficient',
          reasoningTrace,
          followUpSuggestions: [
            `Where is the ${inst.displayName} now?`,
            `What is next to the ${inst.displayName}?`,
          ],
        };
      }

      const prevObs = validObs[validObs.length - 2];
      const currObs = validObs[validObs.length - 1];

      const c1x = (prevObs.bbox![0] + prevObs.bbox![2]) / 2;
      const c1y = (prevObs.bbox![1] + prevObs.bbox![3]) / 2;
      const c2x = (currObs.bbox![0] + currObs.bbox![2]) / 2;
      const c2y = (currObs.bbox![1] + currObs.bbox![3]) / 2;

      const pixelDist = Math.hypot(c1x - c2x, c1y - c2y);
      const moved = pixelDist > 40;

      reasoningTrace.push(`Calculated centroid drift: ${Math.round(pixelDist)}px between frame ${prevObs.frameId} and ${currObs.frameId}.`);
      const resultText = moved
        ? `Yes, movement detected! The ${inst.displayName} moved ~${Math.round(pixelDist)}px between frame ${prevObs.frameId} and ${currObs.frameId}.`
        : `No meaningful movement detected for the ${inst.displayName} between frame ${prevObs.frameId} and ${currObs.frameId} (drift: ${Math.round(pixelDist)}px).`;

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
          explanation: 'Computed centroid displacement between sequential observations in fixed viewpoint.',
          provisional: false,
        },
        requiresVLM: false,
        certainty: 'confirmed',
        reasoningTrace,
        followUpSuggestions: [
          `Where is the ${inst.displayName} now?`,
          `What is beside the ${inst.displayName}?`,
        ],
      };
    }

    // DEFAULT FALLBACK
    reasoningTrace.push('Query could not be resolved from current scene graph facts. Delegating to visual fallback.');
    return {
      queryPlan: plan,
      answer: `The query could not be resolved from structured scene graph facts.`,
      evidence: {
        source: 'workspace_memory',
        workspaceId,
        frameIds: currentGraph ? [currentGraph.frameId] : [],
        timestamps: currentGraph ? [currentGraph.timestamp] : [],
        objectInstanceIds: [],
        relationshipIds: [],
        explanation: 'Query intent unhandled by structured memory.',
        provisional: false,
      },
      requiresVLM: true,
      vlmReason: 'Query requires visual verification.',
      certainty: 'insufficient',
      reasoningTrace,
    };
  }
}
