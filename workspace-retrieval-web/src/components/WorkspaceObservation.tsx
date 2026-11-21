/**
 * WorkspaceObservation Component.
 * Supports multi-frame observation sessions, frame selection,
 * side-by-side comparative views, independent bounding-box overlays,
 * and synchronized instance highlighting.
 */

import React, { useRef, useState, useEffect } from 'react';
import {
  Upload,
  Play,
  PlusCircle,
  Sparkles,
  Focus,
  Box,
  Network,
  Image as ImageIcon,
  SlidersHorizontal,
  X,
  Maximize2,
  Minimize2,
  Clock,
  Layers,
  ChevronDown,
} from 'lucide-react';
import { BoundingBoxOverlay, OverlayMode } from './BoundingBoxOverlay';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { WORKSPACE_DOMAINS } from '../config/workspaceDomains';
import { FrameSelector, FrameItem } from './FrameSelector';
import { formatHumanTime } from '../utils/time';

export interface FrameObservationData {
  frameId: string;
  frameIndex: number;
  timestamp: string;
  imageSrc: string;
  imgWidth: number;
  imgHeight: number;
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
}

interface WorkspaceObservationProps {
  // Sequence & Frame Selector
  frames: FrameItem[];
  selectedFrameIds: string[];
  onToggleFrameSelect: (frameId: string) => void;
  onSelectAllFrames: () => void;
  onClearFrameSelection: () => void;
  onSelectOnlyFrame: (frameId: string) => void;
  onAddNextObservation: () => void;
  onBatchUpload: (files: FileList | File[]) => void;
  onDeleteFrame?: (frameId: string) => void;

  // Selected observations data
  selectedFramesData: FrameObservationData[];

  // Fallback single observation properties (for legacy / new image before processing)
  imageSrc: string | null;
  imgWidth: number;
  imgHeight: number;
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];

  // Interactive highlighting
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;

  // Actions & Config
  onImageSelected: (file: File | Blob, filename: string) => void;
  onRunSgg: () => void;
  selectedDomain: string;
  onDomainChange: (domain: string) => void;
  isRunning: boolean;
  hasPriorObservations: boolean;
}

export const WorkspaceObservation: React.FC<WorkspaceObservationProps> = ({
  frames,
  selectedFrameIds,
  onToggleFrameSelect,
  onSelectAllFrames,
  onClearFrameSelection,
  onSelectOnlyFrame,
  onAddNextObservation,
  onBatchUpload,
  onDeleteFrame,
  selectedFramesData,
  imageSrc,
  imgWidth,
  imgHeight,
  objects,
  relationships,
  hoveredId,
  onHover,
  selectedId,
  onSelect,
  colorMap,
  isEntityMatch,
  onImageSelected,
  onRunSgg,
  selectedDomain,
  onDomainChange,
  isRunning,
  hasPriorObservations,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [mode, setMode] = useState<OverlayMode>('focus');
  const [annotationOpacity, setAnnotationOpacity] = useState<number>(100);
  const [maximizedFrameId, setMaximizedFrameId] = useState<string | null>(null);

  // Keyboard shortcut: Escape clears selection, hover, or modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (maximizedFrameId) {
          setMaximizedFrameId(null);
        } else {
          onSelect(null);
          onHover(null);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [maximizedFrameId, onSelect, onHover]);

  const handleFile = (file: File | Blob, name = 'uploaded_image.jpg') => {
    onImageSelected(file, name);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (e.dataTransfer.files.length > 1) {
        onBatchUpload(e.dataTransfer.files);
        return;
      }
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        handleFile(file, file.name);
        return;
      }
    }

    const uri = e.dataTransfer.getData('text/uri-list');
    const html = e.dataTransfer.getData('text/html');
    let imageUrl = uri;
    if (!imageUrl && html) {
      const match = html.match(/src=["'](https?:\/\/[^"']+)["']/i);
      if (match) imageUrl = match[1];
    }

    if (imageUrl) {
      try {
        const res = await fetch(imageUrl);
        const blob = await res.blob();
        handleFile(blob, 'web_dropped_image.jpg');
      } catch (err) {
        console.warn('Could not fetch web image directly:', err);
      }
    }
  };

  const loadSampleImage = async () => {
    try {
      const res = await fetch('/samples/sample_workspace_1.jpg');
      const blob = await res.blob();
      handleFile(blob, 'sample_workspace_1.jpg');
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  // Find currently selected object (if any)
  const allCurrentObjects = selectedFramesData.length > 0
    ? selectedFramesData.flatMap((f) => f.objects)
    : objects;

  const selectedObj = selectedId
    ? allCurrentObjects.find((o) => {
        if (isEntityMatch) return isEntityMatch(selectedId, o.detectionId) || isEntityMatch(selectedId, o.instanceId);
        return o.detectionId === selectedId || o.instanceId === selectedId;
      })
    : null;

  const selectedColor = selectedObj
    ? colorMap?.get(selectedObj.detectionId) || (selectedObj.instanceId ? colorMap?.get(selectedObj.instanceId) : undefined) || '#38bdf8'
    : '#38bdf8';

  const selectedConnectedRels = selectedObj
    ? (selectedFramesData.length > 0 ? selectedFramesData.flatMap((f) => f.relationships) : relationships).filter((r) => {
        if (isEntityMatch) {
          return (
            isEntityMatch(selectedObj.detectionId, r.subjectDetectionId) ||
            isEntityMatch(selectedObj.instanceId, r.subjectDetectionId) ||
            isEntityMatch(selectedObj.detectionId, r.objectDetectionId) ||
            isEntityMatch(selectedObj.instanceId, r.objectDetectionId)
          );
        }
        return (
          r.subjectDetectionId === selectedObj.detectionId ||
          r.objectDetectionId === selectedObj.detectionId ||
          (selectedObj.instanceId &&
            (r.subjectDetectionId === selectedObj.instanceId || r.objectDetectionId === selectedObj.instanceId))
        );
      })
    : [];

  const hasFrames = frames.length > 0;
  const isMultiFrame = selectedFramesData.length > 1;

  return (
    <div className="observation-panel-card">
      {/* 1. Observation Timeline / Filmstrip */}
      {hasFrames && (
        <FrameSelector
          frames={frames}
          selectedFrameIds={selectedFrameIds}
          onToggleFrameSelect={onToggleFrameSelect}
          onSelectAll={onSelectAllFrames}
          onClearSelection={onClearFrameSelection}
          onSelectOnlyFrame={onSelectOnlyFrame}
          onAddNextFrame={onAddNextObservation}
          onBatchUpload={onBatchUpload}
          onDeleteFrame={onDeleteFrame}
          isProcessing={isRunning}
        />
      )}

      {/* 2. Top Header Toolbar */}
      <div className="observation-toolbar-header">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-sm tracking-wide text-zinc-100 flex items-center gap-2">
            <Layers size={16} className="text-cyan-400" />
            {isMultiFrame
              ? `Workspace Multi-Frame View (${selectedFramesData.length} Selected)`
              : selectedFramesData.length === 1
              ? `Workspace Observation · Frame ${selectedFramesData[0].frameIndex}`
              : 'Workspace Observation'}
          </span>
          {isMultiFrame && (
            <span className="badge-counter text-[11px] bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
              Comparative Mode Active
            </span>
          )}
        </div>

        {/* Domain Selector & Actions */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5" title="Select workspace domain vocabulary">
            <span className="text-[11px] text-zinc-400 font-medium">Domain:</span>
            <select
              className="domain-select"
              value={selectedDomain}
              onChange={(e) => onDomainChange(e.target.value)}
              disabled={isRunning}
            >
              {WORKSPACE_DOMAINS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0], e.target.files[0].name);
                e.target.value = '';
              }
            }}
          />

          {!hasFrames && (
            <button
              type="button"
              className="btn-upload-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRunning}
              title="Upload RGB workspace photograph"
            >
              <Upload size={13} />
              <span>Upload Image</span>
            </button>
          )}

          {/* Run Inference Button (for newly loaded pending image) */}
          {imageSrc && (!hasFrames || objects.length === 0) && (
            <button
              type="button"
              className="btn-run-inference"
              onClick={onRunSgg}
              disabled={isRunning}
              title="Run REACT++ YOLOv12n scene graph generation"
            >
              <Play size={13} className={isRunning ? 'animate-spin' : ''} />
              <span>{isRunning ? 'Processing...' : 'Run Understanding'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Body */}
      {!hasFrames && !imageSrc ? (
        /* Empty State Dropzone */
        <div
          className={`observation-dropzone-large ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="flex flex-col items-center justify-center p-8 text-center max-w-md mx-auto">
            <div className="w-14 h-14 rounded-2xl bg-cyan-950/40 border border-cyan-800/50 flex items-center justify-center text-cyan-400 mb-4 shadow-lg shadow-cyan-950/30">
              <Upload size={24} />
            </div>
            <h4 className="text-base font-semibold text-zinc-100 mb-1">
              Select or Drop Workspace Photographs
            </h4>
            <p className="text-xs text-zinc-400 mb-5 leading-relaxed">
              Upload RGB workspace images (single or sequential) to extract 2D scene graphs, bounding boxes, spatial relationships, and persistent object memory.
            </p>
            <div className="flex items-center gap-3">
              <button
                type="button"
                className="btn-upload-primary"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                <Upload size={13} /> Browse Images
              </button>
              <button
                type="button"
                className="btn-sample-image"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSampleImage();
                }}
              >
                <Sparkles size={13} className="text-cyan-400" /> Sample Image
              </button>
            </div>
          </div>
        </div>
      ) : hasFrames && selectedFramesData.length === 0 ? (
        /* 0 Frames Selected State */
        <div className="observation-dropzone-large">
          <div className="flex flex-col items-center justify-center p-12 text-center max-w-md mx-auto">
            <div className="w-12 h-12 rounded-full bg-zinc-800/60 border border-zinc-700/60 flex items-center justify-center text-zinc-400 mb-3">
              <Clock size={20} />
            </div>
            <h4 className="text-sm font-semibold text-zinc-200 mb-1">
              No Frames Selected
            </h4>
            <p className="text-xs text-zinc-500 mb-4">
              Select one or more observation frames from the timeline above to inspect bounding boxes, scene graphs, and spatial reasoning.
            </p>
            <button
              type="button"
              className="btn-upload-primary"
              onClick={onSelectAllFrames}
            >
              Select All Frames
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Overlay Visibility Strip (Global for all selected frames) */}
          <div className="overlay-controls-strip">
            <div className="overlay-mode-segmented">
              <button
                type="button"
                className={`overlay-mode-chip ${mode === 'focus' ? 'active' : ''}`}
                onClick={() => setMode('focus')}
                title="Focus Mode: Clean default, automatically emphasizes hovered or selected object"
              >
                <Focus size={12} /> Focus
              </button>
              <button
                type="button"
                className={`overlay-mode-chip ${mode === 'objects' ? 'active' : ''}`}
                onClick={() => setMode('objects')}
                title="Objects Mode: Display bounding boxes only"
              >
                <Box size={12} /> Objects
              </button>
              <button
                type="button"
                className={`overlay-mode-chip ${mode === 'relationships' ? 'active' : ''}`}
                onClick={() => setMode('relationships')}
                title="Relationships Mode: Display all bounding boxes and relationship lines"
              >
                <Network size={12} /> Relationships
              </button>
              <button
                type="button"
                className={`overlay-mode-chip ${mode === 'clean' ? 'active' : ''}`}
                onClick={() => setMode('clean')}
                title="Clean Image: View original photograph without overlays"
              >
                <ImageIcon size={12} /> Clean
              </button>
            </div>

            {/* Master Annotation Opacity Slider */}
            <div className="overlay-slider-control" title="Adjust overlay annotation opacity across all frames">
              <SlidersHorizontal size={12} />
              <input
                type="range"
                min="10"
                max="100"
                value={annotationOpacity}
                onChange={(e) => setAnnotationOpacity(Number(e.target.value))}
                className="overlay-slider-input"
              />
              <span className="font-mono text-[11px] min-w-[26px]">
                {annotationOpacity}%
              </span>
            </div>
          </div>

          {/* Selected Object Inspector Banner (when locked) */}
          {selectedObj && (
            <div className="selected-inspector-card">
              <div className="selected-inspector-meta">
                <span
                  className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0"
                  style={{
                    backgroundColor: selectedColor,
                    boxShadow: `0 0 6px ${selectedColor}99`,
                  }}
                />
                <span className="font-semibold text-white capitalize">
                  {selectedObj.modelLabel || selectedObj.projectCategory}
                </span>
                <span
                  className="selected-id-badge"
                  style={{
                    backgroundColor: `${selectedColor}22`,
                    color: selectedColor,
                    border: `1px solid ${selectedColor}55`,
                  }}
                >
                  {selectedObj.instanceId || selectedObj.detectionId}
                </span>
                {selectedObj.instanceId && selectedObj.detectionId !== selectedObj.instanceId && (
                  <span className="text-zinc-500 font-mono text-[10px]">
                    ({selectedObj.detectionId})
                  </span>
                )}
                <span className="text-zinc-400">
                  Confidence: <strong className="text-cyan-400">{(selectedObj.confidence * 100).toFixed(0)}%</strong>
                </span>
                <span className="text-zinc-500 font-mono text-[11px]">
                  [{selectedObj.bbox.join(', ')}]
                </span>
                <span className="text-zinc-400">
                  &bull; {selectedConnectedRels.length} relation{selectedConnectedRels.length === 1 ? '' : 's'}
                </span>
              </div>
              <button
                type="button"
                className="btn-dismiss-selection"
                onClick={() => onSelect(null)}
                title="Clear selection (or press Escape)"
              >
                <X size={13} />
                <span>Esc</span>
              </button>
            </div>
          )}

          {/* Observation Viewers Grid */}
          {selectedFramesData.length === 1 ? (
            /* Single Frame Observation Stage */
            <div className="observation-stage">
              <div className="observation-overlay-wrapper">
                <img
                  src={selectedFramesData[0].imageSrc || imageSrc || ''}
                  alt={`Frame ${selectedFramesData[0].frameIndex}`}
                  className="observation-viewer-img"
                />
                <BoundingBoxOverlay
                  objects={selectedFramesData[0].objects}
                  relationships={selectedFramesData[0].relationships}
                  imgWidth={selectedFramesData[0].imgWidth || imgWidth}
                  imgHeight={selectedFramesData[0].imgHeight || imgHeight}
                  mode={mode}
                  masterOpacity={annotationOpacity / 100}
                  hoveredId={hoveredId}
                  onHover={onHover}
                  selectedId={selectedId}
                  onSelect={onSelect}
                  colorMap={colorMap}
                  isEntityMatch={isEntityMatch}
                />
              </div>
            </div>
          ) : (
            /* Multi-Frame Side-by-Side / Grid Layout */
            <div
              className={`multi-frame-grid ${
                selectedFramesData.length === 2 ? 'grid-2-col' : 'grid-multi-col'
              }`}
            >
              {selectedFramesData.map((fData) => (
                <div key={fData.frameId} className="multi-frame-card">
                  {/* Per-Frame Header */}
                  <div className="multi-frame-header">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-semibold text-xs text-zinc-200">
                        Frame {fData.frameIndex}
                      </span>
                      <span className="text-[11px] text-zinc-500 font-mono truncate">
                        {formatHumanTime(fData.timestamp)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-zinc-400 font-mono">
                        {fData.objects.length} obj &bull; {fData.relationships.length} rel
                      </span>
                      <button
                        type="button"
                        className="btn-expand-frame"
                        onClick={() => setMaximizedFrameId(fData.frameId)}
                        title="Maximize this frame in large view"
                      >
                        <Maximize2 size={12} />
                      </button>
                    </div>
                  </div>

                  {/* Independent Canvas Stage with strict coordinate isolation */}
                  <div className="multi-frame-stage">
                    <div className="observation-overlay-wrapper">
                      <img
                        src={fData.imageSrc}
                        alt={`Frame ${fData.frameIndex}`}
                        className="observation-viewer-img"
                      />
                      <BoundingBoxOverlay
                        objects={fData.objects}
                        relationships={fData.relationships}
                        imgWidth={fData.imgWidth}
                        imgHeight={fData.imgHeight}
                        mode={mode}
                        masterOpacity={annotationOpacity / 100}
                        hoveredId={hoveredId}
                        onHover={onHover}
                        selectedId={selectedId}
                        onSelect={onSelect}
                        colorMap={colorMap}
                        isEntityMatch={isEntityMatch}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Footer Metadata */}
          <div className="observation-meta-footer">
            <div className="flex items-center gap-3">
              {selectedFramesData.length === 1 ? (
                <>
                  <span>
                    {selectedFramesData[0].imgWidth} &times; {selectedFramesData[0].imgHeight} px
                  </span>
                  <span>&bull;</span>
                  <span className="text-zinc-300 font-medium">
                    {selectedFramesData[0].objects.length} detected objects
                  </span>
                  <span>&bull;</span>
                  <span className="text-zinc-300 font-medium">
                    {selectedFramesData[0].relationships.length} relationships
                  </span>
                </>
              ) : (
                <>
                  <span className="text-zinc-300 font-medium">
                    {selectedFramesData.length} observations selected
                  </span>
                  <span>&bull;</span>
                  <span className="text-zinc-300 font-medium">
                    {selectedFramesData.reduce((acc, f) => acc + f.objects.length, 0)} total detections
                  </span>
                  <span>&bull;</span>
                  <span className="text-zinc-300 font-medium">
                    {selectedFramesData.reduce((acc, f) => acc + f.relationships.length, 0)} total relations
                  </span>
                </>
              )}
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
              Model: react_pp_yolo12n &bull; Domain: {selectedDomain}
            </div>
          </div>
        </>
      )}

      {/* Maximized Frame Modal */}
      {maximizedFrameId && (() => {
        const maxFrame = selectedFramesData.find((f) => f.frameId === maximizedFrameId);
        if (!maxFrame) return null;

        return (
          <div
            className="frame-modal-backdrop"
            onClick={() => setMaximizedFrameId(null)}
          >
            <div
              className="frame-modal-content"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="frame-modal-header">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm text-zinc-100">
                    Frame {maxFrame.frameIndex} ({maxFrame.frameId})
                  </span>
                  <span className="text-xs text-zinc-400 font-mono">
                    {formatHumanTime(maxFrame.timestamp)} &bull; {maxFrame.imgWidth}&times;{maxFrame.imgHeight}px
                  </span>
                </div>
                <button
                  type="button"
                  className="btn-modal-close"
                  onClick={() => setMaximizedFrameId(null)}
                >
                  <X size={16} />
                </button>
              </div>

              <div className="frame-modal-body">
                <div className="observation-overlay-wrapper">
                  <img
                    src={maxFrame.imageSrc}
                    alt={`Frame ${maxFrame.frameIndex}`}
                    className="observation-viewer-img"
                  />
                  <BoundingBoxOverlay
                    objects={maxFrame.objects}
                    relationships={maxFrame.relationships}
                    imgWidth={maxFrame.imgWidth}
                    imgHeight={maxFrame.imgHeight}
                    mode={mode}
                    masterOpacity={annotationOpacity / 100}
                    hoveredId={hoveredId}
                    onHover={onHover}
                    selectedId={selectedId}
                    onSelect={onSelect}
                    colorMap={colorMap}
                    isEntityMatch={isEntityMatch}
                  />
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
