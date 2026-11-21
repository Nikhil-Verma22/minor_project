/**
 * Main Application Component.
 * Complete UI/UX Redesign:
 * - Wide, responsive .app-shell container (up to 1560px+)
 * - Compact top navigation header with runtime status chips
 * - Two-column workspace grid (65-68% observation viewer, 32-35% scene intelligence sidebar)
 * - Prominent full-width Natural Language Query (NLQ) composer
 * - Collapsible performance telemetry and diagnostics at bottom
 */

import React, { useEffect, useRef, useState } from 'react';
import { Layers } from 'lucide-react';
import { ModelStatus } from '../components/ModelStatus';
import { WorkspaceObservation } from '../components/WorkspaceObservation';
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

  // Memory Records State
  const [frames, setFrames] = useState<FrameRecord[]>([]);
  const [instances, setInstances] = useState<PersistentObjectInstance[]>([]);

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

    refreshMemoryState();

    return () => {
      orchestrator.dispose();
    };
  }, []);

  const refreshMemoryState = async () => {
    try {
      const fList = await MemoryRepository.getFrames(workspaceId);
      const iList = await MemoryRepository.getInstances(workspaceId);
      setFrames(fList);
      setInstances(iList);
    } catch (err) {
      console.error('Failed to load memory state:', err);
    }
  };

  const handleImageSelected = (fileOrBlob: File | Blob, filename: string) => {
    setCurrentImageBlob(fileOrBlob);
    const url = URL.createObjectURL(fileOrBlob);
    setCurrentImageSrc(url);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setImgDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    // Reset previous frame predictions
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

  const handleRunSgg = async () => {
    if (!currentImageSrc || !orchestratorRef.current) return;
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load image for inference'));
        img.src = currentImageSrc;
        if (img.complete) resolve();
      });

      await orchestratorRef.current.processImage(img, currentImageBlob || undefined);
      await refreshMemoryState();
    } catch (err: any) {
      console.error('[App] Error running SGG pipeline:', err);
      alert('Error during Scene Understanding: ' + (err.message || String(err)));
    }
  };

  const handleAddNextObservation = async () => {
    if (!currentImageSrc || !orchestratorRef.current) return;
    handleRunSgg();
  };

  const handleAskQuery = async (
    queryText: string,
    context?: ConversationContext,
    scope?: QueryScope,
    forceVLM?: boolean,
    forceSGG?: boolean
  ) => {
    if (!orchestratorRef.current) return;
    try {
      await orchestratorRef.current.answerQuery(queryText, context, scope, forceVLM, forceSGG);
    } catch (err) {
      console.error('Error executing query:', err);
    }
  };

  const handleSelectHistoricalFrame = async (frameId: string, targetObjectId?: string) => {
    try {
      const frame = await MemoryRepository.getFrame(frameId);
      if (frame) {
        if (frame.thumbnailDataUrl) {
          setCurrentImageSrc(frame.thumbnailDataUrl);
        }
        setImgDimensions({ width: frame.imageWidth, height: frame.imageHeight });

        const obs = await MemoryRepository.getObservationsForFrame(frameId);
        const rels = await MemoryRepository.getRelationshipsForFrame(frameId);

        setObjects(
          obs.map((o) => {
            const cleanLabel = o.categoryId.replace(/_[0-9]+$/, '');
            const mapped = VocabularyAdapter.mapObjectCategory(cleanLabel);
            return {
              detectionId: o.instanceId || o.annotationId,
              instanceId: o.instanceId,
              modelLabel: cleanLabel,
              projectCategory: mapped.mapped ? mapped.projectCategory : null,
              bbox: o.bbox || [0, 0, 0, 0],
              confidence: o.confidence || 0.9,
              attributes: o.attributes,
            };
          })
        );

        setRelationships(
          rels.map((r) => ({
            relationId: r.relationshipId,
            subjectDetectionId: r.subjectId,
            predicateModelLabel: r.predicate,
            canonicalPredicate: r.predicate,
            objectDetectionId: r.objectId,
            confidence: r.confidence,
            source: r.source as any,
            provisional: r.provisional,
          }))
        );
      }

      if (targetObjectId) {
        setSelectedId(targetObjectId);
      }

      // Smooth scroll image viewer into view
      const elem = document.querySelector('.workspace-main');
      if (elem) {
        elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (err) {
      console.error('Failed to load historical frame:', err);
    }
  };

  const handleClearMemory = async () => {
    if (confirm('Clear all local workspace memory records?')) {
      await MemoryRepository.clearWorkspace(workspaceId);
      await refreshMemoryState();
      setGraph(null);
      setObjects([]);
      setRelationships([]);
      setQueryResult(null);
      setSelectedId(null);
      setHoveredId(null);
    }
  };

  // Compute unified entity color map for synchronized highlighting across views
  const entityColors = React.useMemo(() => {
    return buildEntityColorMap(objects, graph?.nodes);
  }, [objects, graph]);

  return (
    <div className="app-shell">
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
            imageSrc={currentImageSrc}
            imgWidth={imgDimensions.width}
            imgHeight={imgDimensions.height}
            objects={objects}
            relationships={relationships}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
            onImageSelected={handleImageSelected}
            onRunSgg={handleRunSgg}
            onAddNextObservation={handleAddNextObservation}
            selectedDomain={selectedDomain}
            onDomainChange={handleDomainChange}
            isRunning={pipelineStage === 'sgg_running'}
            hasPriorObservations={frames.length > 0}
          />
        </section>

        {/* Right Column (32-35%): Scene Intelligence Sidebar */}
        <aside className="workspace-sidebar">
          <SceneGraphPanel
            graph={graph}
            hoveredId={hoveredId}
            onHoverNode={setHoveredId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
          />

          <ObjectList
            objects={objects}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
            colorMap={entityColors.colorMap}
            isEntityMatch={entityColors.isEntityMatch}
          />

          <HistoryTimeline
            frames={frames}
            currentFrameId={graph?.frameId || null}
            onSelectFrame={handleSelectHistoricalFrame}
            onClearMemory={handleClearMemory}
            onRefreshHistory={refreshMemoryState}
          />
        </aside>
      </main>

      {/* C. Full-Width Prominent Workspace Assistant */}
      <section className="assistant-section">
        <WorkspaceAssistant
          onAskQuery={handleAskQuery}
          queryResult={queryResult}
          pipelineStage={pipelineStage}
          isEnabled={objects.length > 0 || frames.length > 0}
          onSelectEntity={(id) => {
            setSelectedId(id);
            const elem = document.querySelector('.workspace-main');
            if (elem) {
              elem.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }}
          onSelectHistoricalFrame={handleSelectHistoricalFrame}
          availableFramesCount={frames.length}
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
