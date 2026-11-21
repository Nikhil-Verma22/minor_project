/**
 * Main Application Component.
 * Complete Multi-Frame Workspace Observation & Scene Intelligence:
 * - Timeline / Filmstrip frame selector with single & batch uploading
 * - Multi-frame side-by-side comparative observation viewer
 * - Strict per-frame coordinate and bounding-box isolation
 * - Union 2D Scene Graph with Frame Filtering and source badges
 * - Multi-frame object grouping and persistent instance tracking
 * - Evidence-based Workspace Assistant reasoning over selected frame context
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Layers } from 'lucide-react';
import { ModelStatus } from '../components/ModelStatus';
import { WorkspaceObservation, FrameObservationData } from '../components/WorkspaceObservation';
import { FrameItem } from '../components/FrameSelector';
import { SceneGraphPanel } from '../components/SceneGraphPanel';
import { ObjectList } from '../components/ObjectList';
import { WorkspaceAssistant } from '../components/WorkspaceAssistant';
import { HistoryTimeline } from '../components/HistoryTimeline';
import { DiagnosticsPanel } from '../components/DiagnosticsPanel';
import { PipelineOrchestrator } from '../pipeline/pipelineOrchestrator';
import { MemoryRepository } from '../memory/memoryRepository';
import { SGGInferenceResult, SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { SceneGraph } from '../graph/graphTypes';
import { NLQResult, ConversationContext, QueryScope } from '../nlq/queryTypes';
import { FrameRecord, PersistentObjectInstance } from '../memory/memoryTypes';
import { PipelineStage } from '../pipeline/pipelineTypes';
import { buildEntityColorMap } from '../utils/colors';
import { VocabularyAdapter } from '../models/sgg/vocabularyAdapter';

export const App: React.FC = () => {
  const orchestratorRef = useRef<PipelineOrchestrator | null>(null);
  const addFrameInputRef = useRef<HTMLInputElement>(null);

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
  const [objects, setObjects] = useState<SGGObjectDetection[]>([]);
  const [relationships, setRelationships] = useState<SGGRelation[]>([]);
  const [graph, setGraph] = useState<SceneGraph | null>(null);
  const [queryResult, setQueryResult] = useState<NLQResult | null>(null);

  // Interaction State
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Multi-Frame Sequence & Selection State
  const [frames, setFrames] = useState<FrameRecord[]>([]);
  const [instances, setInstances] = useState<PersistentObjectInstance[]>([]);
  const [selectedFrameIds, setSelectedFrameIds] = useState<string[]>([]);
  const [frameCache, setFrameCache] = useState<Map<string, FrameObservationData>>(new Map());

  // Initialize pipeline orchestrator on mount
  useEffect(() => {
    const orchestrator = new PipelineOrchestrator({
      onStageChange: (stage, msg) => {
        setPipelineStage(stage);
        if (msg) setStageMessage(msg);
      },
      onSggResult: (res) => {
        setSggResult(res);
        setObjects([...res.objects]);
        setRelationships([...res.relationships]);
        setImgDimensions({ width: res.imageWidth, height: res.imageHeight });
      },
      onGraphUpdate: (g) => {
        setGraph(g);
        refreshMemoryState();
      },
      onQueryResult: (res) => {
        setQueryResult(res);
      },
      onError: (err) => {
        console.error('[App] Pipeline error:', err);
      },
    });

    orchestratorRef.current = orchestrator;

    orchestrator.initialize(workspaceId, selectedDomain).then((status) => {
      setSggStatus(status);
    });

    refreshMemoryState(true);

    return () => {
      orchestrator.dispose();
    };
  }, []);

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

  const handleImageSelected = (fileOrBlob: File | Blob, filename: string) => {
    setCurrentImageBlob(fileOrBlob);
    const url = URL.createObjectURL(fileOrBlob);
    setCurrentImageSrc(url);

    const img = new Image();
    img.onload = () => {
      setImgDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    setSggResult(null);
    setObjects([]);
    setRelationships([]);
    setGraph(null);
    setQueryResult(null);
    setSelectedId(null);
    setHoveredId(null);
  };

  const handleDomainChange = (domainId: string) => {
    setSelectedDomain(domainId);
    orchestratorRef.current?.initialize(workspaceId, domainId);
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

    const frameData: FrameObservationData = {
      frameId,
      frameIndex: res.graph.nodes.length > 0 ? frames.length + 1 : 1,
      timestamp: res.graph.timestamp,
      imageSrc: url,
      imgWidth: res.sggResult.imageWidth,
      imgHeight: res.sggResult.imageHeight,
      objects: res.sggResult.objects,
      relationships: res.sggResult.relationships,
    };

    setFrameCache((prev) => new Map(prev).set(frameId, frameData));
    await refreshMemoryState();
    return frameId;
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

  // Add Next Frame (appends one image to observation sequence)
  const handleAddNextObservation = () => {
    addFrameInputRef.current?.click();
  };

  const handleAddFrameFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      try {
        const frameId = await processSingleImage(file, file.name);
        if (frameId) {
          setSelectedFrameIds((prev) => [...prev, frameId]);
        }
      } catch (err: any) {
        console.error('Failed to add next frame:', err);
        alert('Failed to process next frame: ' + (err.message || String(err)));
      }
      e.target.value = '';
    }
  };

  // Batch Image Upload (processes multiple images sequentially)
  const handleBatchUpload = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    if (fileArray.length === 0 || !orchestratorRef.current) return;

    const newlyAddedFrameIds: string[] = [];
    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      try {
        const fId = await processSingleImage(file, file.name);
        if (fId) newlyAddedFrameIds.push(fId);
      } catch (err) {
        console.error(`Error processing batch file ${file.name}:`, err);
      }
    }

    if (newlyAddedFrameIds.length > 0) {
      setSelectedFrameIds(newlyAddedFrameIds);
    }
  };

  // Delete Frame
  const handleDeleteFrame = async (frameId: string) => {
    try {
      await MemoryRepository.deleteFrame(frameId);
      setFrameCache((prev) => {
        const next = new Map(prev);
        next.delete(frameId);
        return next;
      });
      setSelectedFrameIds((prev) => prev.filter((id) => id !== frameId));
      await refreshMemoryState();
    } catch (err) {
      console.error(`Failed to delete frame ${frameId}:`, err);
    }
  };

  // Frame selection toggles
  const handleToggleFrameSelect = (frameId: string) => {
    setSelectedFrameIds((prev) =>
      prev.includes(frameId) ? prev.filter((id) => id !== frameId) : [...prev, frameId]
    );
  };

  const handleSelectAllFrames = () => {
    setSelectedFrameIds(frames.map((f) => f.frameId));
  };

  const handleClearFrameSelection = () => {
    setSelectedFrameIds([]);
  };

  const handleSelectOnlyFrame = (frameId: string) => {
    setSelectedFrameIds([frameId]);
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
    try {
      const framesToQuery = activeFrameIds || selectedFrameIds;
      await orchestratorRef.current.answerQuery(
        queryText,
        context,
        scope,
        forceVLM,
        forceSGG,
        framesToQuery
      );
    } catch (err) {
      console.error('Error executing query:', err);
    }
  };

  const handleSelectHistoricalFrame = async (frameId: string, targetObjectId?: string) => {
    setSelectedFrameIds([frameId]);
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
      await MemoryRepository.clearWorkspace(workspaceId);
      await refreshMemoryState();
      setFrameCache(new Map());
      setSelectedFrameIds([]);
      setGraph(null);
      setObjects([]);
      setRelationships([]);
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

  // Selected observations data array for multi-frame viewer
  const selectedFramesData: FrameObservationData[] = useMemo(() => {
    return selectedFrameIds
      .map((id) => frameCache.get(id))
      .filter(Boolean) as FrameObservationData[];
  }, [selectedFrameIds, frameCache]);

  // Active combined objects across selected frames
  const activeObjects = useMemo(() => {
    if (selectedFramesData.length > 0) {
      return selectedFramesData.flatMap((f) => f.objects);
    }
    return objects;
  }, [selectedFramesData, objects]);

  // Active combined relationships across selected frames
  const activeRelationships = useMemo(() => {
    if (selectedFramesData.length > 0) {
      return selectedFramesData.flatMap((f) => f.relationships);
    }
    return relationships;
  }, [selectedFramesData, relationships]);

  // Active combined scene graph for SceneGraphPanel
  const activeGraph = useMemo<SceneGraph | null>(() => {
    if (selectedFramesData.length === 0) return graph;
    if (selectedFramesData.length === 1) {
      const sf = selectedFramesData[0];
      if (graph && graph.frameId === sf.frameId) return graph;
      const nodesMap = new Map();
      for (const obj of sf.objects) {
        const id = obj.instanceId || obj.detectionId;
        nodesMap.set(id, {
          id,
          instanceId: id,
          detectionId: obj.detectionId,
          categoryId: obj.projectCategory || obj.modelLabel,
          label: obj.modelLabel,
          bbox: obj.bbox,
          confidence: obj.confidence,
          isSurface: false,
          attributes: obj.attributes || { color: null, type: null },
        });
      }
      return {
        frameId: sf.frameId,
        timestamp: sf.timestamp,
        nodes: Array.from(nodesMap.values()),
        edges: sf.relationships.map((r) => ({
          id: r.relationId,
          sourceId: r.subjectDetectionId,
          targetId: r.objectDetectionId,
          predicate: r.canonicalPredicate,
          confidence: r.confidence,
          source: r.source,
          provisional: r.provisional,
          frameId: sf.frameId,
        })),
      };
    }

    // Multi-frame union graph with observedFrames attribution
    const nodesMap = new Map();
    const edgeMap = new Map();

    for (const sf of selectedFramesData) {
      for (const obj of sf.objects) {
        const id = obj.instanceId || obj.detectionId;
        if (!nodesMap.has(id)) {
          nodesMap.set(id, {
            id,
            instanceId: id,
            detectionId: obj.detectionId,
            categoryId: obj.projectCategory || obj.modelLabel,
            label: obj.modelLabel,
            bbox: obj.bbox,
            confidence: obj.confidence,
            isSurface: false,
            attributes: obj.attributes || { color: null, type: null },
          });
        }
      }

      for (const r of sf.relationships) {
        const key = `${r.subjectDetectionId}__${r.canonicalPredicate}__${r.objectDetectionId}`;
        if (edgeMap.has(key)) {
          const existing = edgeMap.get(key);
          if (!existing.observedFrames.includes(sf.frameId)) {
            existing.observedFrames.push(sf.frameId);
          }
        } else {
          edgeMap.set(key, {
            id: r.relationId,
            sourceId: r.subjectDetectionId,
            targetId: r.objectDetectionId,
            predicate: r.canonicalPredicate,
            confidence: r.confidence,
            source: r.source,
            provisional: r.provisional,
            frameId: sf.frameId,
            observedFrames: [sf.frameId],
          });
        }
      }
    }

    return {
      frameId: 'combined_selection',
      timestamp: selectedFramesData[selectedFramesData.length - 1].timestamp,
      nodes: Array.from(nodesMap.values()),
      edges: Array.from(edgeMap.values()),
    };
  }, [selectedFramesData, graph]);

  // Compute unified entity color map for synchronized highlighting across views
  const entityColors = useMemo(() => {
    return buildEntityColorMap(activeObjects, activeGraph?.nodes);
  }, [activeObjects, activeGraph]);

  return (
    <div className="app-shell">
      {/* Hidden Add Next Frame File Input */}
      <input
        ref={addFrameInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleAddFrameFileChange}
      />

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
        />
      </header>

      {/* B. Main Two-Column Workspace Grid */}
      <main className="workspace-grid">
        {/* Left Column (65-68%): Unified Workspace Observation */}
        <section className="workspace-main">
          <WorkspaceObservation
            frames={frameItems}
            selectedFrameIds={selectedFrameIds}
            onToggleFrameSelect={handleToggleFrameSelect}
            onSelectAllFrames={handleSelectAllFrames}
            onClearFrameSelection={handleClearFrameSelection}
            onSelectOnlyFrame={handleSelectOnlyFrame}
            onAddNextObservation={handleAddNextObservation}
            onBatchUpload={handleBatchUpload}
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
            isRunning={pipelineStage === 'sgg_running'}
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

      {/* D. Collapsible System Diagnostics & Telemetry */}
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
