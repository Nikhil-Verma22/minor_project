/**
 * Pipeline Orchestrator.
 * Central manager enforcing the strict execution sequence:
 * Upload RGB image -> REACT++ SGG -> Identity Association -> 2D Scene Graph -> Workspace Memory -> NLQ -> Graph/Memory Reasoning -> (only if unresolved) SmolVLM Fallback -> Validated Answer.
 */

import { PipelineStage, PipelineCallbacks } from './pipelineTypes';
import { SGGEngine, SGGInferenceResult, ModelStatus } from '../models/sgg/SGGEngine';
import { ReactPlusPlusOnnxWebEngine } from '../models/sgg/ReactPlusPlusOnnxWebEngine';
import { VLMEngine } from '../models/vlm/VLMEngine';
import { SmolVLMExecuTorchEngine } from '../models/vlm/SmolVLMExecuTorchEngine';
import { IdentityAssociation } from '../memory/identityAssociation';
import { GraphBuilder } from '../graph/graphBuilder';
import { SceneGraph } from '../graph/graphTypes';
import { MemoryRepository } from '../memory/memoryRepository';
import { QueryParser } from '../nlq/parser';
import { QueryExecutor } from '../nlq/queryExecutor';
import { NLQResult } from '../nlq/queryTypes';
import { generateId, formatFrameId } from '../utils/ids';
import { getCurrentIsoTimestamp } from '../utils/time';
import { logMessage } from '../utils/logging';
import {
  WorkspaceRecord,
  SequenceRecord,
  FrameRecord,
  RelationshipFact,
  VLMCandidate,
} from '../memory/memoryTypes';

export class PipelineOrchestrator {
  private sggEngine: SGGEngine;
  private vlmEngine: VLMEngine | null = null;
  private parser = new QueryParser();

  private currentStage: PipelineStage = 'idle';
  private callbacks: PipelineCallbacks = {};

  private currentWorkspaceId = 'ws_default';
  private currentDomainId = 'student_workspace';
  private currentSequenceId = 'seq_001';
  private frameCount = 0;

  private currentImageBitmap: ImageBitmap | null = null;
  private currentImageBlob: Blob | null = null;
  private currentGraph: SceneGraph | null = null;
  private latestSggResult: SGGInferenceResult | null = null;

  constructor(callbacks?: PipelineCallbacks) {
    this.sggEngine = new ReactPlusPlusOnnxWebEngine('yolo12n');
    if (callbacks) this.callbacks = callbacks;
  }

  setCallbacks(callbacks: PipelineCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  getStage(): PipelineStage {
    return this.currentStage;
  }

  getCurrentGraph(): SceneGraph | null {
    return this.currentGraph;
  }

  getLatestSggResult(): SGGInferenceResult | null {
    return this.latestSggResult;
  }

  private setStage(stage: PipelineStage, msg?: string): void {
    this.currentStage = stage;
    this.callbacks.onStageChange?.(stage, msg);
    logMessage('info', 'PipelineOrchestrator', `Stage changed to: ${stage} (${msg || ''})`);
  }

  async initialize(workspaceId = 'ws_default', domainId = 'student_workspace'): Promise<ModelStatus> {
    this.currentWorkspaceId = workspaceId;
    this.currentDomainId = domainId;
    this.setStage('idle', 'Initializing REACT++...');
    const status = await this.sggEngine.initialize();
    return status;
  }

  setWorkspace(workspaceId: string, domainId: string): void {
    this.currentWorkspaceId = workspaceId;
    this.currentDomainId = domainId;
    this.frameCount = 0;
    this.currentSequenceId = generateId('seq');
  }

  /**
   * Process an uploaded RGB workspace image through SGG, Identity, Graph, and Memory.
   */
  async processImage(
    imageSource: ImageBitmap | HTMLImageElement | HTMLCanvasElement,
    imageBlob?: Blob,
    filename = 'workspace_upload.jpg'
  ): Promise<{ sggResult: SGGInferenceResult; graph: SceneGraph }> {
    try {
      this.currentImageBlob = imageBlob || null;
      if (imageSource instanceof ImageBitmap) {
        this.currentImageBitmap = imageSource;
      }

      this.setStage('image_loaded', 'Image loaded, starting REACT++ SGG inference...');
      this.setStage('sgg_running', 'Running REACT++ YOLOv12n forward pass...');

      const sggResult = await this.sggEngine.infer(imageSource);
      this.latestSggResult = sggResult;
      this.callbacks.onSggResult?.(sggResult);
      this.setStage('sgg_completed', `Detected ${sggResult.objects.length} objects, ${sggResult.relationships.length} relations`);

      // 2. Identity Association
      this.setStage('identity_association', 'Associating object identities against workspace memory...');
      const timestamp = getCurrentIsoTimestamp();
      this.frameCount += 1;
      const frameId = formatFrameId(this.currentSequenceId, this.frameCount);

      const existingInstances = await MemoryRepository.getInstances(this.currentWorkspaceId);
      const latestObsMap = new Map();
      for (const inst of existingInstances) {
        const obsHistory = await MemoryRepository.getObservationsForInstance(inst.instanceId);
        if (obsHistory.length > 0) {
          latestObsMap.set(inst.instanceId, obsHistory[obsHistory.length - 1]);
        }
      }

      const association = IdentityAssociation.associateDetections(
        sggResult.objects,
        existingInstances,
        latestObsMap,
        this.currentWorkspaceId,
        frameId,
        timestamp,
        sggResult.imageWidth,
        sggResult.imageHeight
      );

      // Attach associated instanceId to sggResult objects
      for (const obj of sggResult.objects) {
        const inst = association.instanceMap.get(obj.detectionId);
        if (inst) {
          obj.instanceId = inst.instanceId;
        }
      }

      // 3. Build Relationships Facts
      const relationshipFacts: RelationshipFact[] = [];
      for (const rel of sggResult.relationships) {
        const subjInst = association.instanceMap.get(rel.subjectDetectionId);
        const objInst = association.instanceMap.get(rel.objectDetectionId);

        if (subjInst && objInst && rel.canonicalPredicate) {
          relationshipFacts.push({
            relationshipId: generateId('rel'),
            frameId,
            workspaceId: this.currentWorkspaceId,
            subjectId: subjInst.instanceId,
            predicate: rel.canonicalPredicate,
            objectId: objInst.instanceId,
            confidence: rel.confidence,
            source: 'reactpp',
            timestamp,
            provisional: false,
          });
        }
      }

      // 4. Graph Building
      this.setStage('graph_building', 'Constructing 2D Scene Graph...');
      const instancesMap = new Map<string, any>();
      for (const inst of existingInstances) instancesMap.set(inst.instanceId, inst);
      for (const inst of association.newInstances) instancesMap.set(inst.instanceId, inst);

      const graph = GraphBuilder.buildGraph(
        frameId,
        timestamp,
        association.observations,
        relationshipFacts,
        instancesMap
      );
      this.currentGraph = graph;
      this.callbacks.onGraphUpdate?.(graph);

      // 5. Transactional Memory Update
      this.setStage('memory_update', 'Writing frame facts to local IndexedDB...');

      const workspaceRecord: WorkspaceRecord = {
        workspaceId: this.currentWorkspaceId,
        domainId: this.currentDomainId,
        name: `Workspace (${this.currentDomainId})`,
        createdAt: timestamp,
      };

      const sequenceRecord: SequenceRecord = {
        sequenceId: this.currentSequenceId,
        workspaceId: this.currentWorkspaceId,
        cameraId: 'cam_fixed_01',
        viewpoint: 'overhead_desk',
        createdAt: timestamp,
      };

      const frameRecord: FrameRecord = {
        frameId,
        sequenceId: this.currentSequenceId,
        workspaceId: this.currentWorkspaceId,
        frameIndex: this.frameCount - 1,
        timestamp,
        imageWidth: sggResult.imageWidth,
        imageHeight: sggResult.imageHeight,
        filename,
      };

      await MemoryRepository.saveObservationFrameTransaction({
        workspace: workspaceRecord,
        sequence: sequenceRecord,
        frame: frameRecord,
        newInstances: association.newInstances,
        updatedInstances: association.updatedInstances,
        observations: association.observations,
        relationships: relationshipFacts,
        events: association.events,
      });

      this.setStage('ready_for_query', 'Scene understanding ready. Enter a query.');
      return { sggResult, graph };
    } catch (err: any) {
      this.setStage('error', err.message || String(err));
      this.callbacks.onError?.(err.message || String(err));
      throw err;
    }
  }

  /**
   * Execute natural language query with deterministic reasoning and selective VLM fallback.
   */
  async answerQuery(queryText: string): Promise<NLQResult> {
    try {
      this.setStage('query_parsing', `Parsing query: "${queryText}"`);
      const activeInstances = await MemoryRepository.getInstances(this.currentWorkspaceId);
      const plan = this.parser.parseQuery(queryText, activeInstances);

      this.setStage('graph_memory_reasoning', 'Querying Scene Graph & Temporal Memory...');
      const nlqResult = await QueryExecutor.executeQuery(plan, this.currentGraph, this.currentWorkspaceId);

      // If sufficient evidence was found in graph/memory, answer directly!
      if (!nlqResult.requiresVLM) {
        this.setStage('answer_ready', 'Answer derived from Scene Graph / Memory');
        this.callbacks.onQueryResult?.(nlqResult);
        return nlqResult;
      }

      // SELECTIVE VLM FALLBACK: Only invoked when Graph/Memory is unresolved
      logMessage('info', 'PipelineOrchestrator', 'Invoking VLM Fallback:', nlqResult.vlmReason);
      this.setStage('vlm_loading', 'Lazy loading SmolVLM-500M WebGPU runtime...');

      if (!this.vlmEngine) {
        this.vlmEngine = new SmolVLMExecuTorchEngine();
      }

      const vlmStatus = await this.vlmEngine.initialize((progress) => {
        this.callbacks.onVlmProgress?.(progress);
      });

      if (!vlmStatus.loaded) {
        // Return clear limitation message if WebGPU unavailable
        const safeResult: NLQResult = {
          ...nlqResult,
          answer: `${nlqResult.answer} (VLM fallback unavailable: ${vlmStatus.error || 'WebGPU not available'}).`,
        };
        this.setStage('answer_ready', 'Completed with VLM limitation noted');
        this.callbacks.onQueryResult?.(safeResult);
        return safeResult;
      }

      this.setStage('vlm_reasoning', 'Running SmolVLM-500M visual fallback reasoning...');

      if (!this.currentImageBitmap && !this.currentImageBlob) {
        throw new Error('No current image available for VLM reasoning');
      }

      const vlmResponse = await this.vlmEngine.answer(
        (this.currentImageBitmap || this.currentImageBlob)!,
        queryText,
        {
          currentGraph: this.currentGraph,
          activeInstances,
        }
      );

      // Save provisional VLM candidates
      for (const cand of vlmResponse.candidate_relations) {
        const vlmCandidateRecord: VLMCandidate = {
          candidateId: generateId('vlm_cand'),
          frameId: this.currentGraph?.frameId || 'frame_001',
          workspaceId: this.currentWorkspaceId,
          subjectId: cand.subject_id,
          freeTextPredicate: cand.predicate_text,
          canonicalPredicate: cand.canonical_predicate,
          objectId: cand.object_id,
          confidence: cand.confidence,
          evidence: cand.evidence,
          status: 'provisional',
          timestamp: getCurrentIsoTimestamp(),
        };
        await MemoryRepository.saveVlmCandidate(vlmCandidateRecord);
      }

      const finalResult: NLQResult = {
        queryPlan: plan,
        answer: vlmResponse.answer,
        evidence: {
          source: 'vlm',
          workspaceId: this.currentWorkspaceId,
          frameIds: this.currentGraph ? [this.currentGraph.frameId] : [],
          timestamps: this.currentGraph ? [this.currentGraph.timestamp] : [],
          objectInstanceIds: vlmResponse.candidate_relations.map((r) => r.subject_id),
          relationshipIds: [],
          confidence: vlmResponse.candidate_relations[0]?.confidence ?? 0.7,
          explanation: `Visual fallback reasoned over RGB image context. Found ${vlmResponse.candidate_relations.length} provisional candidate relation(s).`,
          provisional: true,
        },
        requiresVLM: false,
      };

      this.setStage('answer_ready', 'Answer derived from SmolVLM-500M fallback');
      this.callbacks.onQueryResult?.(finalResult);
      return finalResult;
    } catch (err: any) {
      this.setStage('error', err.message || String(err));
      this.callbacks.onError?.(err.message || String(err));
      throw err;
    }
  }

  async dispose(): Promise<void> {
    await this.sggEngine.dispose();
    if (this.vlmEngine) {
      await this.vlmEngine.dispose();
    }
    this.setStage('idle', 'Pipeline disposed');
  }
}
