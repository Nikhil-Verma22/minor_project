/**
 * ImageViewer Component.
 * Responsive image canvas with focus mode controls, annotation opacity slider,
 * and a persistent selection inspector.
 */

import React, { useState, useEffect } from 'react';
import { Layers, Eye, Focus, Network, Box, Image as ImageIcon, SlidersHorizontal, X } from 'lucide-react';
import { BoundingBoxOverlay, OverlayMode } from './BoundingBoxOverlay';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { getPaletteColor } from '../utils/colors';

interface ImageViewerProps {
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
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
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
}) => {
  // Visibility Mode: 'focus' is the default smart focus hierarchy
  const [mode, setMode] = useState<OverlayMode>('focus');
  // Master annotation opacity: 0 to 100%
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

  if (!imageSrc) {
    return (
      <div className="image-viewer-placeholder">
        <Eye size={40} className="text-zinc-600 mb-2" />
        <div className="text-sm text-zinc-400">No workspace image loaded yet</div>
      </div>
    );
  }

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
    <div className="panel-card overflow-hidden">
      {/* 1. Header Toolbar & Mode Selector */}
      <div className="overlay-toolbar">
        <div className="flex items-center gap-2 text-zinc-400">
          <Layers size={14} className="text-cyan-400" />
          <span>
            {imgWidth} &times; {imgHeight} px &bull; {objects.length} objects &bull; {relationships.length} relations
          </span>
        </div>

        {/* Mode Selector Group & Opacity Slider */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Mode Switcher */}
          <div className="overlay-mode-group">
            <button
              type="button"
              className={`overlay-mode-btn ${mode === 'focus' ? 'active' : ''}`}
              onClick={() => setMode('focus')}
              title="Focus Mode: Clean default, automatically emphasizes hovered or selected object"
            >
              <Focus size={13} />
              <span>Focus</span>
            </button>
            <button
              type="button"
              className={`overlay-mode-btn ${mode === 'objects' ? 'active' : ''}`}
              onClick={() => setMode('objects')}
              title="Objects Mode: Display bounding boxes only"
            >
              <Box size={13} />
              <span>Objects</span>
            </button>
            <button
              type="button"
              className={`overlay-mode-btn ${mode === 'relationships' ? 'active' : ''}`}
              onClick={() => setMode('relationships')}
              title="Relationships Mode: Display all bounding boxes and relationship lines"
            >
              <Network size={13} />
              <span>Relationships</span>
            </button>
            <button
              type="button"
              className={`overlay-mode-btn ${mode === 'clean' ? 'active' : ''}`}
              onClick={() => setMode('clean')}
              title="Clean Image: View original photograph without overlays"
            >
              <ImageIcon size={13} />
              <span>Clean</span>
            </button>
          </div>

          {/* Opacity Slider */}
          <div className="overlay-slider-container" title="Adjust annotation overlay opacity">
            <SlidersHorizontal size={12} className="text-zinc-500" />
            <input
              type="range"
              min="10"
              max="100"
              value={annotationOpacity}
              onChange={(e) => setAnnotationOpacity(Number(e.target.value))}
              className="overlay-slider"
            />
            <span className="text-[11px] font-mono text-zinc-400 min-w-[28px]">
              {annotationOpacity}%
            </span>
          </div>
        </div>
      </div>

      {/* 2. Persistent Selection Inspector Banner */}
      {selectedObj && (
        <div className="selected-inspector-banner">
          <div className="selected-inspector-left">
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
              className="selected-inspector-pill"
              style={{
                backgroundColor: `${selectedColor}20`,
                color: selectedColor,
                border: `1px solid ${selectedColor}50`,
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
              Score: <strong className="text-cyan-400">{(selectedObj.confidence * 100).toFixed(0)}%</strong>
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
            className="selected-inspector-clear-btn"
            onClick={() => onSelect(null)}
            title="Deselect (or press Escape)"
          >
            <X size={14} />
            <span className="text-[11px] ml-1">Esc</span>
          </button>
        </div>
      )}

      {/* 3. Image + SVG Overlay Container */}
      <div className="image-viewer-stage">
        <div className="image-overlay-wrapper">
          <img
            src={imageSrc}
            alt="Workspace Observation"
            className="viewer-image"
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
    </div>
  );
};
