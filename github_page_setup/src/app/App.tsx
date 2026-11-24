/**
 * Main Application Component.
 * Complete Multi-Frame Workspace Observation & Scene Intelligence:
 * - Global drag-and-drop across the entire application viewport
 * - Unified "+ Upload Images" control handling single and batch uploads
 * - Simplified one-click timeline frame selection
 * - Safe individual frame deletion with confirmation dialog
 * - Persistent session logging & audit system
 * - Strict selected-frame data isolation (preventing stale object/graph leaks)
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { AlertCircle, Layers, Loader2, Upload, X } from 'lucide-react';
import { ModelStatus } from '../components/ModelStatus';
import { WorkspaceObservation } from '../components/WorkspaceObservation';
import { FrameItem } from '../components/FrameSelector';
import { SceneGraphPanel } from '../components/SceneGraphPanel';
import { ObjectList } from '../components/ObjectList';
import { WorkspaceAssistant } from '../components/WorkspaceAssistant';
import { HistoryTimeline } from '../components/HistoryTimeline';
import { DiagnosticsPanel } from '../components/DiagnosticsPanel';
import { PipelineOrchestrator } from '../pipeline/pipelineOrchestrator';
import { MemoryRepository } from '../memory/memoryRepository';
import { SGGInferenceResult, SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { NLQResult, ConversationContext, QueryScope } from '../nlq/queryTypes';
import { FrameRecord, PersistentObjectInstance } from '../memory/memoryTypes';
import { PipelineStage } from '../pipeline/pipelineTypes';
import { buildEntityColorMap } from '../utils/colors';
import { VocabularyAdapter } from '../models/sgg/vocabularyAdapter';
import { getSelectedFrameData, FrameObservationData } from '../utils/frameSelection';
import { sessionLogger } from '../logging/sessionLogger';

export const App: React.FC = () => {
  const orchestratorRef = useRef<PipelineOrchestrator | null>(null);

  // Model & Pipeline State
  const [sggStatus, setSggStatus] = useState<any>({ loaded: false, provider: 'unavailable', variant: 'yolo12n' });
  const [vlmStatus, setVlmStatus] = useState<any>({ loaded: false, provider: 'unavailable' });
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle');
  const [stageMessage, setStageMessage] = useState<string>('');

  // Workspace & Image State
  const [selectedDomain, setSelectedDomain] = useState<string>('student_workspace');
  const [workspaceId, setWorkspaceId] = useState<string>('ws_default');
  const [currentImageSrc, setCurrentImageSrc] = useState<string | null>(null);
  const [currentImageBlob, setCurrentImageBlob] = useState<Blob | null>(null);
  const [imgDimensions, setImgDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  // Scene Graph & Results
  const [sggResult, setSggResult] = useState<SGGInferenceResult | null>(null);
  const [queryResult, setQueryResult] = useState<NLQResult | null>(null);

  // Interaction State
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Global Drag-and-Drop State
  const [isGlobalDragging, setIsGlobalDragging] = useState(false);
  const [uploadStatusMsg, setUploadStatusMsg] = useState<string | null>(null);
  const [uploadErrorMsg, setUploadErrorMsg] = useState<string | null>(null);
  const dragCounterRef = useRef(0);

  // Multi-Frame Sequence & Selection State
  const [frames, setFrames] = useState<FrameRecord[]>([]);
  const [instances, setInstances] = useState<PersistentObjectInstance[]>([]);
  const [selectedFrameIds, setSelectedFrameIds] = useState<string[]>([]);
  const [frameCache, setFrameCache] = useState<Map<string, FrameObservationData>>(new Map());

  // Initialize pipeline orchestrator on mount
  useEffect(() => {
    sessionLogger.info('application', 'APP_INITIALIZED', 'Workspace Retrieval Application mounted', {
      workspaceId,
      selectedDomain,
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Unknown',
    });

    const orchestrator = new PipelineOrchestrator({
      onStageChange: (stage, msg) => {
        setPipelineStage(stage);
        if (msg) setStageMessage(msg);
      },
      onSggResult: (res) => {
        setSggResult(res);
        setImgDimensions({ width: res.imageWidth, height: res.imageHeight });
        sessionLogger.info('inference', 'INFERENCE_COMPLETED', `SGG completed for frame ${res.frameId}: ${res.objects.length} objects, ${res.relationships.length} relations`, {
          frameId: res.frameId,
          objectCount: res.objects.length,
          relationCount: res.relationships.length,
          executionProvider: res.executionProvider,
          timings: res.timings,
        });
      },
      onGraphUpdate: (_g) => {
        refreshMemoryState();
      },
      onQueryResult: (res) => {
        setQueryResult(res);
        sessionLogger.info('query', 'QUERY_COMPLETED', `Query answered (${res.evidence.source}): "${res.answer}"`, {
          intent: res.queryPlan?.intent,
          source: res.evidence.source,
          evidenceObjectIds: res.evidence.objectInstanceIds,
          evidenceFrameIds: res.evidence.frameIds,
          answer: res.answer,
        });
      },
      onError: (err: any) => {
        sessionLogger.error('error', 'PIPELINE_ERROR', `Pipeline error: ${err.message || String(err)}`, {
          error: err.message || String(err),
          stack: err.stack,
        });
        console.error('[App] Pipeline error:', err);
      },
    });

    orchestratorRef.current = orchestrator;

    orchestrator.initialize(workspaceId, selectedDomain).then((status) => {
      setSggStatus(status);
      sessionLogger.info('application', 'MODEL_INITIALIZED', `SGG model initialized on ${status.provider}`, {
        provider: status.provider,
        loaded: status.loaded,
      });
    });

    refreshMemoryState(true);

    return () => {
      orchestrator.dispose();
      sessionLogger.info('application', 'APP_DISPOSED', 'Application unmounted');
    };
  }, []);

  // Synchronize logger context with active frame selection
  useEffect(() => {
    sessionLogger.setContext({
      workspaceId,
      selectedFrameIds,
    });
  }, [workspaceId, selectedFrameIds]);

  const refreshMemoryState = async (autoSelectLatest = false) => {
    try {
      const fList = await MemoryRepository.getFrames(workspaceId);
      const iList = await MemoryRepository.getInstances(workspaceId);
      setFrames(fList);
      setInstances(iList);

      // Hydrate frame cache for any newly discovered frames
      setFrameCache((prevCache) => {
        const nextCache = new Map(prevCache);
        fList.forEach(async (f) => {
          if (!nextCache.has(f.frameId)) {
            const obs = await MemoryRepository.getObservationsForFrame(f.frameId);
            const rels = await MemoryRepository.getRelationshipsForFrame(f.frameId);
            const sggObjs: SGGObjectDetection[] = obs.map((o) => {
              const cleanLabel = o.categoryId.replace(/_[0-9]+$/, '');
              const mapped = VocabularyAdapter.mapObjectCategory(cleanLabel);
              return {
                detectionId: o.annotationId,
                instanceId: o.instanceId,
                frameId: f.frameId,
                modelLabel: cleanLabel,
                projectCategory: mapped.mapped ? mapped.projectCategory : null,
                bbox: o.bbox || [0, 0, 0, 0],
                confidence: o.confidence || 0.9,
                attributes: o.attributes,
              };
            });
            const sggRels: SGGRelation[] = rels.map((r) => ({
              relationId: r.relationshipId,
              subjectDetectionId: r.subjectId,
              predicateModelLabel: r.predicate,
              canonicalPredicate: r.predicate,
              objectDetectionId: r.objectId,
              confidence: r.confidence,
              source: r.source as any,
              provisional: r.provisional,
              frameId: f.frameId,
            }));

            nextCache.set(f.frameId, {
              frameId: f.frameId,
              frameIndex: f.frameIndex,
              timestamp: f.timestamp,
              imageSrc: f.thumbnailDataUrl || '',
              imgWidth: f.imageWidth,
              imgHeight: f.imageHeight,
              objects: sggObjs,
              relationships: sggRels,
            });
          }
        });
        return nextCache;
      });

      if (autoSelectLatest && fList.length > 0 && selectedFrameIds.length === 0) {
        setSelectedFrameIds([fList[fList.length - 1].frameId]);
      }
    } catch (err) {
      console.error('Failed to load memory state:', err);
    }
  };

  const handleImageSelected = (fileOrBlob: File | Blob, name = 'uploaded_image.jpg') => {
    const normalizedFile =
      fileOrBlob instanceof File
        ? fileOrBlob
        : new File([fileOrBlob], name, { type: fileOrBlob.type || 'image/jpeg' });
    addImagesToObservationSequence([normalizedFile]);
  };

  const handleDomainChange = (domainId: string) => {
    setSelectedDomain(domainId);
    orchestratorRef.current?.initialize(workspaceId, domainId);
    sessionLogger.info('application', 'DOMAIN_CHANGED', `Domain switched to ${domainId}`);
  };

  const processSingleImage = async (fileOrBlob: File | Blob, filename: string): Promise<string | null> => {
    if (!orchestratorRef.current) return null;
    const url = URL.createObjectURL(fileOrBlob);

    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Failed to load image '${filename}' for inference`));
      img.src = url;
      if (img.complete) resolve();
    });

    const res = await orchestratorRef.current.processImage(img, fileOrBlob, filename);
    const frameId = res.graph.frameId;
    setSggStatus((prev: any) =>
      prev.loaded ? prev : { loaded: true, provider: res.sggResult.executionProvider, variant: 'yolo12n' }
    );

    const frameData: FrameObservationData = {
      frameId,
      frameIndex: res.graph.nodes.length > 0 ? frames.length + 1 : 1,
      timestamp: res.graph.timestamp,
      imageSrc: url,
      imgWidth: res.sggResult.imageWidth,
      imgHeight: res.sggResult.imageHeight,
      objects: res.sggResult.objects.map((o) => ({ ...o, frameId })),
      relationships: res.sggResult.relationships.map((r) => ({ ...r, frameId })),
    };

    setFrameCache((prev) => new Map(prev).set(frameId, frameData));
    await refreshMemoryState();
    return frameId;
  };

  // Central unified upload handler for single & batch image ingestion
  const addImagesToObservationSequence = async (files: FileList | File[]): Promise<string[]> => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0 || !orchestratorRef.current) return [];

    const newlyAddedFrameIds: string[] = [];
    const uploadBatchId = `batch_${Date.now()}`;
    const validExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'];
    setUploadErrorMsg(null);
    setUploadStatusMsg(
      sggStatus.loaded
        ? `Processing ${fileArray.length} image(s) with REACT++...`
        : 'Waiting for REACT++ model initialization (45.4 MB) before processing image...'
    );

    sessionLogger.info('upload', 'UPLOAD_BATCH_STARTED', `Starting upload of ${fileArray.length} image(s)`, {
      batchId: uploadBatchId,
      fileCount: fileArray.length,
      filenames: fileArray.map((f) => f.name || 'uploaded_image.jpg'),
    });

    for (let i = 0; i < fileArray.length; i++) {
      const rawFile = fileArray[i];
      const safeName = rawFile.name || `uploaded_image_${i + 1}.jpg`;
      const file =
        rawFile instanceof File && rawFile.name
          ? rawFile
          : new File([rawFile], safeName, { type: rawFile.type || 'image/jpeg' });

      const dotIdx = safeName.lastIndexOf('.');
      const ext = dotIdx >= 0 ? safeName.substring(dotIdx).toLowerCase() : '';
      const isValid = file.type.startsWith('image/') || validExtensions.includes(ext);

      if (!isValid) {
        sessionLogger.warn('upload', 'IMAGE_VALIDATION_FAILED', `Unsupported file type for '${safeName}'`, {
          filename: safeName,
          type: file.type,
          size: file.size,
        });
        setUploadErrorMsg(`File '${safeName}' is not a supported image (PNG, JPG, WebP). Skipping.`);
        continue;
      }

      sessionLogger.info('upload', 'IMAGE_UPLOAD_STARTED', `Ingesting image '${safeName}' (${Math.round(file.size / 1024)} KB)`, {
        batchId: uploadBatchId,
        filename: safeName,
        type: file.type,
        sizeBytes: file.size,
      });

      try {
        setUploadStatusMsg(`Running REACT++ scene understanding on '${safeName}'...`);
        const fId = await processSingleImage(file, safeName);
        if (fId) {
          newlyAddedFrameIds.push(fId);
          sessionLogger.info('upload', 'IMAGE_UPLOAD_COMPLETED', `Successfully stored frame ${fId} for '${safeName}'`, {
            frameId: fId,
            filename: safeName,
          });
        }
      } catch (err: any) {
        const msg = err.message || String(err);
        sessionLogger.error('upload', 'IMAGE_UPLOAD_FAILED', `Failed to process image '${safeName}': ${msg}`, {
          filename: safeName,
          error: msg,
        });
        console.error(`Error processing file ${safeName}:`, err);
        setUploadErrorMsg(`Failed to process '${safeName}': ${msg}`);
      }
    }

    setUploadStatusMsg(null);
    if (newlyAddedFrameIds.length > 0) {
      setSelectedFrameIds(newlyAddedFrameIds);
      sessionLogger.info('frame', 'FRAME_SELECTION_CHANGED', `Active selection updated to ${newlyAddedFrameIds.length} newly added frame(s)`, {
        selectedFrameIds: newlyAddedFrameIds,
      });
    }

    return newlyAddedFrameIds;
  };

  const handleRunSgg = async () => {
    if (!currentImageSrc || !orchestratorRef.current) return;
    try {
      const frameId = await processSingleImage(currentImageBlob || new Blob(), 'workspace_upload.jpg');
      if (frameId) {
        setSelectedFrameIds([frameId]);
      }
    } catch (err: any) {
      console.error('[App] Error running SGG pipeline:', err);
      alert('Error during Scene Understanding: ' + (err.message || String(err)));
    }
  };

  // Safe frame deletion with atomic semantics
  const handleDeleteFrame = async (frameId: string) => {
    try {
      sessionLogger.info('deletion', 'FRAME_DELETION_STARTED', `Deleting frame ${frameId}`, { frameId });
      await MemoryRepository.deleteFrame(frameId);

      setFrameCache((prev) => {
        const next = new Map(prev);
        next.delete(frameId);
        return next;
      });

      setSelectedFrameIds((prev) => {
        const next = prev.filter((id) => id !== frameId);
        sessionLogger.info('deletion', 'FRAME_DELETED', `Frame ${frameId} removed from memory and timeline`, {
          deletedFrameId: frameId,
          remainingSelectedFrameIds: next,
        });
        return next;
      });

      await refreshMemoryState(false);
    } catch (err: any) {
      sessionLogger.error('deletion', 'FRAME_DELETION_FAILED', `Failed to delete frame ${frameId}: ${err.message || String(err)}`, {
        frameId,
        error: err.message || String(err),
      });
      console.error(`Failed to delete frame ${frameId}:`, err);
      alert(`Failed to delete frame: ${err.message || String(err)}`);
    }
  };

  // Simplified one-click timeline selection toggles
  const handleToggleFrameSelect = (frameId: string) => {
    setSelectedFrameIds((prev) => {
      const next = prev.includes(frameId)
        ? prev.filter((id) => id !== frameId)
        : [...prev, frameId];
      sessionLogger.info('frame', 'FRAME_SELECTION_CHANGED', `Frame selection toggled: ${next.length} frame(s) active`, {
        toggledFrameId: frameId,
        previousFrameIds: prev,
        selectedFrameIds: next,
        selectedFrameCount: next.length,
      });
      return next;
    });
  };

  const handleSelectAllFrames = () => {
    const allIds = frames.map((f) => f.frameId);
    setSelectedFrameIds(allIds);
    sessionLogger.info('frame', 'FRAME_SELECTION_CHANGED', `Selected all ${allIds.length} frames`, {
      selectedFrameIds: allIds,
    });
  };

  const handleClearFrameSelection = () => {
    setSelectedFrameIds([]);
    sessionLogger.info('frame', 'FRAME_SELECTION_CHANGED', 'Cleared frame selection (0 active frames)', {
      selectedFrameIds: [],
    });
  };

  // Multi-turn Natural Language Query with active frame context
  const handleAskQuery = async (
    queryText: string,
    context?: ConversationContext,
    scope?: QueryScope,
    forceVLM?: boolean,
    forceSGG?: boolean,
    activeFrameIds?: string[]
  ) => {
    if (!orchestratorRef.current) return;
    const framesToQuery = activeFrameIds || selectedFrameIds;

    sessionLogger.info('query', 'QUERY_SUBMITTED', `User query submitted: "${queryText}"`, {
      queryText,
      selectedFrameIdsAtSubmission: selectedFrameIds,
      selectedFrameIdsUsedByExecutor: framesToQuery,
      forceVLM,
      forceSGG,
      scope,
    });

    try {
      await orchestratorRef.current.answerQuery(
        queryText,
        context,
        scope,
        forceVLM,
        forceSGG,
        framesToQuery
      );
    } catch (err: any) {
      sessionLogger.error('query', 'QUERY_FAILED', `Query failed: ${err.message || String(err)}`, {
        queryText,
        framesToQuery,
        error: err.message || String(err),
      });
      console.error('Error executing query:', err);
    }
  };

  const handleSelectHistoricalFrame = async (frameId: string, targetObjectId?: string) => {
    setSelectedFrameIds([frameId]);
    sessionLogger.info('frame', 'FRAME_SELECTION_CHANGED', `Historical frame focused: ${frameId}`, {
      selectedFrameIds: [frameId],
      targetObjectId,
    });

    if (targetObjectId) {
      setSelectedId(targetObjectId);
    }
    const elem = document.querySelector('.workspace-main');
    if (elem) {
      elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleClearMemory = async () => {
    if (confirm('Clear all local workspace memory records?')) {
      sessionLogger.info('memory', 'MEMORY_CLEARED', 'Cleared all workspace memory');
      await MemoryRepository.clearWorkspace(workspaceId);
      await refreshMemoryState();
      setFrameCache(new Map());
      setSelectedFrameIds([]);
      setQueryResult(null);
      setSelectedId(null);
      setHoveredId(null);
    }
  };

  // Format FrameItems for FrameSelector
  const frameItems: FrameItem[] = useMemo(() => {
    return frames.map((f) => {
      const cached = frameCache.get(f.frameId);
      return {
        frameId: f.frameId,
        frameIndex: f.frameIndex,
        timestamp: f.timestamp,
        imageWidth: f.imageWidth,
        imageHeight: f.imageHeight,
        thumbnailDataUrl: f.thumbnailDataUrl,
        imageSrc: cached?.imageSrc || f.thumbnailDataUrl,
        status: f.status || 'completed',
        objectCount: cached?.objects.length,
        relationCount: cached?.relationships.length,
      };
    });
  }, [frames, frameCache]);

  // SINGLE AUTHORITATIVE SOURCE of selected frame data
  // Strictly isolates active visual & query state to selectedFrameIds
  const frameSelectionContext = useMemo(() => {
    return getSelectedFrameData(frameCache, selectedFrameIds);
  }, [frameCache, selectedFrameIds]);

  const activeObjects = frameSelectionContext.selectedDetections;
  const activeRelationships = frameSelectionContext.selectedRelationships;
  const activeGraph = frameSelectionContext.combinedSceneGraph;
  const selectedFramesData = frameSelectionContext.selectedFrames;

  // Compute unified entity color map for synchronized highlighting across views
  const entityColors = useMemo(() => {
    return buildEntityColorMap(activeObjects, activeGraph?.nodes);
  }, [activeObjects, activeGraph]);

  // Global Drag-and-Drop Event Handlers
  const handleGlobalDragEnter = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault();
      dragCounterRef.current++;
      setIsGlobalDragging(true);
    }
  };

  const handleGlobalDragOver = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  };

  const handleGlobalDragLeave = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault();
      dragCounterRef.current--;
      if (dragCounterRef.current <= 0) {
        dragCounterRef.current = 0;
        setIsGlobalDragging(false);
      }
    }
  };

  const handleGlobalDrop = (e: React.DragEvent) => {
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault();
      dragCounterRef.current = 0;
      setIsGlobalDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        addImagesToObservationSequence(e.dataTransfer.files);
      }
    }
  };

  return (
    <div
      className="app-shell relative min-h-screen"
      onDragEnter={handleGlobalDragEnter}
      onDragOver={handleGlobalDragOver}
      onDragLeave={handleGlobalDragLeave}
      onDrop={handleGlobalDrop}
    >
      {/* Global Full-Screen Drag and Drop Overlay */}
      {isGlobalDragging && (
        <div className="fixed inset-0 z-50 pointer-events-none flex items-center justify-center bg-black/80 backdrop-blur-sm p-8 transition-opacity duration-150">
          <div className="w-full max-w-xl h-72 rounded-2xl border-2 border-dashed border-cyan-400 bg-cyan-950/40 flex flex-col items-center justify-center p-8 text-center shadow-2xl shadow-cyan-950/60">
            <div className="w-16 h-16 rounded-2xl bg-cyan-900/50 border border-cyan-500/60 flex items-center justify-center text-cyan-400 mb-4 animate-bounce">
              <Upload size={32} />
            </div>
            <h3 className="text-xl font-bold text-white mb-2 tracking-wide">Upload Images</h3>
            <p className="text-sm text-cyan-200/90 mb-3">
              Drop images anywhere to add workspace observations
            </p>
            <span className="text-xs text-zinc-400 font-mono tracking-wider bg-zinc-900/80 px-3 py-1 rounded-full border border-zinc-700/60">
              PNG &bull; JPG &bull; WebP
            </span>
          </div>
        </div>
      )}

      {/* A. Compact Top Navigation Header */}
      <header className="app-header">
        <div className="app-brand">
          <div className="brand-icon-wrapper">
            <Layers size={20} />
          </div>
          <div className="app-title-group">
            <div className="app-title-row">
              <h1 className="app-title">Object Retrieval</h1>
              <span className="badge-local-first">Local-First</span>
            </div>
            <p className="app-subtitle">Dynamic Workspace Intelligence</p>
          </div>
        </div>

        <ModelStatus
          sggStatus={sggStatus}
          vlmStatus={vlmStatus}
          instanceCount={instances.length}
          frameCount={frames.length}
          stageMessage={stageMessage}
        />
      </header>

      {/* B. Main Two-Column Workspace Grid */}
      <main className="workspace-grid">
        {/* Left Column (65-68%): Unified Workspace Observation */}
        <section className="workspace-main">
          {uploadStatusMsg && (
            <div className="mb-2 px-3 py-2 rounded-lg bg-cyan-950/60 border border-cyan-700/60 text-cyan-200 text-xs flex items-center gap-2">
              <Loader2 size={14} className="animate-spin text-cyan-400 flex-shrink-0" />
              <span>
                {!sggStatus.loaded && stageMessage ? stageMessage : uploadStatusMsg}
              </span>
            </div>
          )}
          {uploadErrorMsg && (
            <div className="mb-2 px-3 py-2 rounded-lg bg-rose-950/70 border border-rose-700/60 text-rose-200 text-xs flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <AlertCircle size={14} className="text-rose-400 flex-shrink-0" />
                <span>{uploadErrorMsg}</span>
              </div>
              <button
                type="button"
                className="text-rose-300 hover:text-white p-0.5"
                onClick={() => setUploadErrorMsg(null)}
                aria-label="Dismiss error"
              >
                <X size={13} />
              </button>
            </div>
          )}
          <WorkspaceObservation
            frames={frameItems}
            selectedFrameIds={selectedFrameIds}
            onToggleFrameSelect={handleToggleFrameSelect}
            onSelectAllFrames={handleSelectAllFrames}
            onClearFrameSelection={handleClearFrameSelection}
            onUploadImages={addImagesToObservationSequence}
            onDeleteFrame={handleDeleteFrame}
            selectedFramesData={selectedFramesData}
            imageSrc={currentImageSrc}
            imgWidth={imgDimensions.width}
            imgHeight={imgDimensions.height}
            objects={activeObjects}
            relationships={activeRelationships}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
            onImageSelected={handleImageSelected}
            onRunSgg={handleRunSgg}
            selectedDomain={selectedDomain}
            onDomainChange={handleDomainChange}
            isRunning={Boolean(uploadStatusMsg) || pipelineStage === 'sgg_running' || pipelineStage === 'image_loaded'}
            hasPriorObservations={frames.length > 0}
          />
        </section>

        {/* Right Column (32-35%): Scene Intelligence Sidebar */}
        <aside className="workspace-sidebar">
          <SceneGraphPanel
            graph={activeGraph}
            hoveredId={hoveredId}
            onHoverNode={setHoveredId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
            selectedFrames={selectedFramesData.map((f) => ({
              frameId: f.frameId,
              frameIndex: f.frameIndex,
            }))}
          />

          <ObjectList
            objects={activeObjects}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
            selectedFrames={selectedFramesData.map((f) => ({
              frameId: f.frameId,
              frameIndex: f.frameIndex,
            }))}
          />

          <HistoryTimeline
            frames={frames}
            currentFrameId={selectedFrameIds[0] || null}
            onSelectFrame={handleSelectHistoricalFrame}
            onClearMemory={handleClearMemory}
            onRefreshHistory={() => refreshMemoryState(false)}
          />
        </aside>
      </main>

      {/* C. Full-Width Prominent Workspace Assistant */}
      <section className="assistant-section">
        <WorkspaceAssistant
          onAskQuery={handleAskQuery}
          queryResult={queryResult}
          pipelineStage={pipelineStage}
          isEnabled={activeObjects.length > 0 || frames.length > 0}
          onSelectEntity={(id) => {
            setSelectedId(id);
            const elem = document.querySelector('.workspace-main');
            if (elem) {
              elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }}
          onSelectHistoricalFrame={handleSelectHistoricalFrame}
          availableFramesCount={frames.length}
          selectedFrameIds={selectedFrameIds}
        />
      </section>

      {/* D. Collapsible System Diagnostics & Telemetry with Session Logs */}
      <footer className="diagnostics-section">
        <DiagnosticsPanel
          sggResult={sggResult}
          instanceCount={instances.length}
          frameCount={frames.length}
        />
      </footer>
    </div>
  );
};
