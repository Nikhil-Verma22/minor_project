/**
 * WorkspaceObservation Component.
 * Unified workspace observation card combining image selection, domain configuration,
 * inference execution, overlay mode controls, and responsive image viewer.
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
  FileImage,
  RefreshCw,
  Layers,
  ChevronDown,
} from 'lucide-react';
import { BoundingBoxOverlay, OverlayMode } from './BoundingBoxOverlay';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { WORKSPACE_DOMAINS } from '../config/workspaceDomains';

interface WorkspaceObservationProps {
  imageSrc: string | null;
  imgWidth: number;
  imgHeight: number;
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
  onImageSelected: (file: File | Blob, filename: string) => void;
  onRunSgg: () => void;
  onAddNextObservation: () => void;
  selectedDomain: string;
  onDomainChange: (domain: string) => void;
  isRunning: boolean;
  hasPriorObservations: boolean;
}

export const WorkspaceObservation: React.FC<WorkspaceObservationProps> = ({
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
  onAddNextObservation,
  selectedDomain,
  onDomainChange,
  isRunning,
  hasPriorObservations,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [mode, setMode] = useState<OverlayMode>('focus');
  const [annotationOpacity, setAnnotationOpacity] = useState<number>(100);

  // Keyboard shortcut: Escape key clears selection and hover
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onSelect(null);
        onHover(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSelect, onHover]);

  const handleFile = (file: File | Blob, name = 'uploaded_image.jpg') => {
    onImageSelected(file, name);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    // 1. Check local dropped files
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        handleFile(file, file.name);
        return;
      }
    }

    // 2. Check dragged web image URLs
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
        console.warn('Could not fetch web image directly, fallback to canvas:', err);
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0);
          canvas.toBlob((b) => {
            if (b) handleFile(b, 'web_image.jpg');
          }, 'image/jpeg');
        };
        img.src = imageUrl;
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
  const selectedObj = selectedId
    ? objects.find((o) => {
        if (isEntityMatch) return isEntityMatch(selectedId, o.detectionId) || isEntityMatch(selectedId, o.instanceId);
        return o.detectionId === selectedId || o.instanceId === selectedId;
      })
    : null;

  const selectedColor = selectedObj
    ? colorMap?.get(selectedObj.detectionId) || (selectedObj.instanceId ? colorMap?.get(selectedObj.instanceId) : undefined) || '#38bdf8'
    : '#38bdf8';

  const selectedConnectedRels = selectedObj
    ? relationships.filter((r) => {
        if (isEntityMatch) {
          return (
            isEntityMatch(selectedObj.detectionId, r.subjectDetectionId) ||
            isEntityMatch(selectedObj.detectionId, r.objectDetectionId)
          );
        }
        return (
          r.subjectDetectionId === selectedObj.detectionId ||
          r.objectDetectionId === selectedObj.detectionId
        );
      })
    : [];

  return (
    <div className="ai-panel">
      {/* 1. Header & Actions Toolbar */}
      <div className="observation-toolbar">
        <div className="observation-actions-left">
          <h2 className="ai-panel-title">
            <Layers size={18} className="ai-panel-title-icon" />
            Workspace Observation
          </h2>

          {/* Domain Selector */}
          <select
            className="domain-select"
            value={selectedDomain}
            onChange={(e) => onDomainChange(e.target.value)}
            title="Active workspace domain ontology"
          >
            {WORKSPACE_DOMAINS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        {/* Action Buttons */}
        <div className="observation-actions-right">
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0], e.target.files[0].name);
              }
            }}
          />

          {imageSrc ? (
            <>
              <button
                type="button"
                className="btn-secondary-action"
                onClick={() => fileInputRef.current?.click()}
                title="Replace current observation image"
              >
                <FileImage size={14} /> Replace
              </button>

              {hasPriorObservations && (
                <button
                  type="button"
                  className="btn-secondary-action"
                  onClick={onAddNextObservation}
                  disabled={isRunning}
                  title="Record this image as a chronological sequential observation"
                >
                  <PlusCircle size={14} /> Add Next Observation
                </button>
              )}

              <button
                type="button"
                className="btn-primary-action"
                onClick={onRunSgg}
                disabled={isRunning}
                title="Run REACT++ Scene Graph Generation"
              >
                {isRunning ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" />
                    Inferring...
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    Run Scene Understanding
                  </>
                )}
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn-secondary-action"
              onClick={loadSampleImage}
            >
              <Sparkles size={14} className="text-cyan-400" /> Load Sample
            </button>
          )}
        </div>
      </div>

      {/* 2. Main Content Area: Empty Dropzone OR Image Viewer */}
      {!imageSrc ? (
        <div
          className={`empty-dropzone ${dragOver ? 'drag-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="dropzone-icon">
            <Upload size={24} />
          </div>
          <div className="dropzone-title">Upload Workspace Observation</div>
          <div className="dropzone-subtitle">
            Drag &amp; drop an RGB workspace image here, browse your files, or drag an image directly from another browser window.
          </div>
          <div className="dropzone-actions" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="btn-primary-action"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} /> Choose File
            </button>
            <button
              type="button"
              className="btn-secondary-action"
              onClick={loadSampleImage}
            >
              <Sparkles size={14} className="text-cyan-400" /> Load Sample Workspace
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* 3. Overlay Visibility Strip */}
          <div className="overlay-controls-strip">
            {/* Mode Segmented Controls */}
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
            <div className="overlay-slider-control" title="Adjust overlay annotation opacity">
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

          {/* 4. Selected Object Inspector Banner (when locked) */}
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

          {/* 5. Observation Stage (Image + SVG Overlay) */}
          <div className="observation-stage">
            <div className="observation-overlay-wrapper">
              <img
                src={imageSrc}
                alt="Workspace Observation"
                className="observation-viewer-img"
              />
              <BoundingBoxOverlay
                objects={objects}
                relationships={relationships}
                imgWidth={imgWidth}
                imgHeight={imgHeight}
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

          {/* 6. Metadata Footer */}
          <div className="observation-meta-footer">
            <div className="flex items-center gap-3">
              <span>{imgWidth} &times; {imgHeight} px</span>
              <span>&bull;</span>
              <span className="text-zinc-300 font-medium">{objects.length} detected objects</span>
              <span>&bull;</span>
              <span className="text-zinc-300 font-medium">{relationships.length} relationships</span>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
              Model: react_pp_yolo12n &bull; Domain: {selectedDomain}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
