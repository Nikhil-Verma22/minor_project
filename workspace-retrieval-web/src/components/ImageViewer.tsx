/**
 * ImageViewer Component.
 * Responsive image canvas with overlay controls and dimensions readout.
 */

import React, { useState } from 'react';
import { Layers, Eye, Tag, Hash, Network } from 'lucide-react';
import { BoundingBoxOverlay } from './BoundingBoxOverlay';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';

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
  const [showLabels, setShowLabels] = useState(true);
  const [showScores, setShowScores] = useState(true);
  const [showInstanceIds, setShowInstanceIds] = useState(true);
  const [showRelations, setShowRelations] = useState(true);

  if (!imageSrc) {
    return (
      <div className="image-viewer-placeholder">
        <Eye size={40} className="text-zinc-600 mb-2" />
        <div className="text-sm text-zinc-400">No workspace image loaded yet</div>
      </div>
    );
  }

  return (
    <div className="panel-card overflow-hidden">
      {/* Controls Bar */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-800 text-xs">
        <div className="flex items-center gap-2 text-zinc-400">
          <Layers size={14} className="text-cyan-400" />
          <span>
            {imgWidth} &times; {imgHeight} px &bull; {objects.length} objects &bull; {relationships.length} relations
          </span>
        </div>

        <div className="flex items-center gap-3">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            <Tag size={12} /> Labels
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showScores}
              onChange={(e) => setShowScores(e.target.checked)}
            />
            Scores
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showInstanceIds}
              onChange={(e) => setShowInstanceIds(e.target.checked)}
            />
            <Hash size={12} /> IDs
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showRelations}
              onChange={(e) => setShowRelations(e.target.checked)}
            />
            <Network size={12} className="text-cyan-400" /> Relations
          </label>
        </div>
      </div>

      {/* Image + SVG Container */}
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
            showLabels={showLabels}
            showScores={showScores}
            showInstanceIds={showInstanceIds}
            showRelations={showRelations}
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
