/**
 * Main Application Component.
 * Integrates REACT++ SGG, 2D Scene Graph, IndexedDB Memory, NLQ Engine, and SmolVLM-500M WebGPU fallback.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Sparkles, RefreshCw } from 'lucide-react';
import { ModelStatus } from '../components/ModelStatus';
import { ImageUploader } from '../components/ImageUploader';
import { ImageViewer } from '../components/ImageViewer';
import { SceneGraphPanel } from '../components/SceneGraphPanel';
import { ObjectList } from '../components/ObjectList';
import { QueryPanel } from '../components/QueryPanel';
import { HistoryTimeline } from '../components/HistoryTimeline';
import { DiagnosticsPanel } from '../components/DiagnosticsPanel';
import { PipelineOrchestrator } from '../pipeline/pipelineOrchestrator';
import { MemoryRepository } from '../memory/memoryRepository';
import { SGGInferenceResult, SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { SceneGraph } from '../graph/graphTypes';
import { NLQResult } from '../nlq/queryTypes';
import { FrameRecord, PersistentObjectInstance } from '../memory/memoryTypes';
import { PipelineStage } from '../pipeline/pipelineTypes';
import { generateId } from '../utils/ids';

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
        setObjects(res.objects);
        setRelationships(res.relationships);
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
      console.error('Failed to refresh memory state:', err);
    }
  };

  const handleDomainChange = (domain: string) => {
    setSelectedDomain(domain);
    const newWsId = `ws_${domain}`;
    setWorkspaceId(newWsId);
    orchestratorRef.current?.setWorkspace(newWsId, domain);
    refreshMemoryState();
  };

  const handleImageSelected = (fileOrBlob: File | Blob, filename: string) => {
    if (currentImageSrc) {
      URL.revokeObjectURL(currentImageSrc);
    }
    const url = URL.createObjectURL(fileOrBlob);
    setCurrentImageSrc(url);
    setCurrentImageBlob(fileOrBlob);
    setQueryResult(null);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setImgDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;
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
      console.error('[App] Error during Scene Understanding:', err);
      alert('Error during Scene Understanding: ' + (err.message || String(err)));
    }
  };

  const handleAddNextObservation = async () => {
    if (!currentImageSrc || !orchestratorRef.current) return;
    handleRunSgg();
  };

  const handleAskQuery = async (queryText: string) => {
    if (!orchestratorRef.current) return;
    await orchestratorRef.current.answerQuery(queryText);
  };

  const handleSelectHistoricalFrame = async (frameId: string) => {
    try {
      const obs = await MemoryRepository.getObservationsForFrame(frameId);
      const rels = await MemoryRepository.getRelationshipsForFrame(frameId);
      const f = await MemoryRepository.getFrame(frameId);

      if (f) {
        setImgDimensions({ width: f.imageWidth, height: f.imageHeight });
      }

      setObjects(
        obs.map((o) => ({
          detectionId: o.annotationId,
          modelLabel: o.categoryId,
          projectCategory: o.categoryId,
          bbox: o.bbox || [0, 0, 0, 0],
          confidence: o.confidence || 0.9,
          attributes: o.attributes,
        }))
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
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-header-left">
          <div className="app-title-row">
            <h1 className="app-title">
              <span className="text-cyan">Object Retrieval</span> in Dynamic Workspace Environments
            </h1>
            <span className="badge-tag">v1.0 Local-First</span>
          </div>
          <p className="app-subtitle">
            Browser-First Scene Graph Generation (REACT++), Persistent Memory (IndexedDB), and SmolVLM-500M WebGPU Fallback
          </p>
        </div>

        <ModelStatus
          sggStatus={sggStatus}
          vlmStatus={vlmStatus}
          instanceCount={instances.length}
          frameCount={frames.length}
        />
      </header>

      {/* Main Grid Layout */}
      <main className="app-main-grid">
        {/* Left Column: Image, Overlays, and Query Console */}
        <section className="col-left">
          <ImageUploader
            onImageSelected={handleImageSelected}
            onRunSgg={handleRunSgg}
            onAddNextObservation={handleAddNextObservation}
            selectedDomain={selectedDomain}
            onDomainChange={handleDomainChange}
            hasImage={!!currentImageSrc}
            isRunning={pipelineStage === 'sgg_running'}
            hasPriorObservations={frames.length > 0}
          />

          <ImageViewer
            imageSrc={currentImageSrc}
            imgWidth={imgDimensions.width}
            imgHeight={imgDimensions.height}
            objects={objects}
            relationships={relationships}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />

          <QueryPanel
            onAskQuery={handleAskQuery}
            queryResult={queryResult}
            pipelineStage={pipelineStage}
            isEnabled={objects.length > 0 || frames.length > 0}
          />
        </section>

        {/* Right Column: Scene Graph, Objects, and Timeline History */}
        <section className="col-right">
          <SceneGraphPanel
            graph={graph}
            hoveredId={hoveredId}
            onHoverNode={setHoveredId}
          />

          <ObjectList
            objects={objects}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />

          <HistoryTimeline
            frames={frames}
            currentFrameId={graph?.frameId || null}
            onSelectFrame={handleSelectHistoricalFrame}
            onClearMemory={handleClearMemory}
            onRefreshHistory={refreshMemoryState}
          />
        </section>
      </main>

      {/* Footer Diagnostics */}
      <footer className="app-footer">
        <DiagnosticsPanel
          sggResult={sggResult}
          instanceCount={instances.length}
          frameCount={frames.length}
        />
      </footer>
    </div>
  );
};
