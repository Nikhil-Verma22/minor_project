/**
 * ObjectList Component.
 * High-density object cards replacing the cramped table.
 * Displays category name, confidence score, detection ID, mapped workspace category,
 * and expandable bounding-box coordinates on demand.
 */

import React, { useState } from 'react';
import { Box, Check, Layers, ChevronDown, ChevronRight } from 'lucide-react';
import { SGGObjectDetection } from '../models/sgg/SGGEngine';
import { getPaletteColor } from '../utils/colors';

interface ObjectListProps {
  objects: SGGObjectDetection[];
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
  selectedFrames?: { frameId: string; frameIndex: number }[];
}

export const ObjectList: React.FC<ObjectListProps> = ({
  objects,
  hoveredId,
  onHover,
  selectedId,
  onSelect,
  colorMap,
  isEntityMatch,
  selectedFrames = [],
}) => {
  const [expandedBboxId, setExpandedBboxId] = useState<string | null>(null);

  const uniqueInstancesCount = new Set(objects.map((o) => o.instanceId || o.detectionId)).size;

  if (objects.length === 0) {
    return (
      <div className="ai-panel">
        <div className="ai-panel-header">
          <h3 className="ai-panel-title">
            <Box size={16} className="ai-panel-title-icon" />
            Detected Objects
          </h3>
        </div>
        <div className="text-center py-8 text-zinc-500 text-xs italic">
          {selectedFrames.length === 0
            ? 'No frames selected. Click a frame in the observation timeline to inspect.'
            : 'No objects detected in selected observation.'}
        </div>
      </div>
    );
  }

  const checkMatch = (targetId: string | null | undefined, candidateId: string | null | undefined) => {
    if (!targetId || !candidateId) return false;
    if (isEntityMatch) return isEntityMatch(targetId, candidateId);
    return targetId === candidateId;
  };

  return (
    <div className="ai-panel">
      {/* Header */}
      <div className="ai-panel-header">
        <div className="flex items-center gap-2">
          <h3 className="ai-panel-title">
            <Box size={16} className="ai-panel-title-icon" />
            Detected Objects
          </h3>
          {selectedFrames.length > 1 ? (
            <span className="badge-counter text-[11px] font-mono text-cyan-300 bg-cyan-950/80 border border-cyan-800/60">
              {selectedFrames.length} Frames &bull; {objects.length} Detections &bull; {uniqueInstancesCount} Instances
            </span>
          ) : (
            <span className="badge-counter">{objects.length} objects</span>
          )}
        </div>
        <span className="text-[11px] text-zinc-500">Click to focus</span>
      </div>

      {/* Modern Compact Object Cards */}
      <div className="objects-scroll-container">
        {objects.map((obj, i) => {
          const color =
            colorMap?.get(obj.detectionId) ||
            (obj.instanceId ? colorMap?.get(obj.instanceId) : undefined) ||
            getPaletteColor(i);

          const isHovered =
            checkMatch(hoveredId, obj.detectionId) ||
            (obj.instanceId ? checkMatch(hoveredId, obj.instanceId) : false);

          const isSelected =
            checkMatch(selectedId, obj.detectionId) ||
            (obj.instanceId ? checkMatch(selectedId, obj.instanceId) : false);

          const isBboxOpen = expandedBboxId === obj.detectionId;

          return (
            <div
              key={obj.detectionId}
              className={`object-card-item ${isSelected ? 'selected' : ''}`}
              style={{
                borderLeftColor: color,
                backgroundColor: isSelected
                  ? 'rgba(57, 169, 245, 0.12)'
                  : isHovered
                  ? 'var(--bg-hover)'
                  : 'var(--bg-surface)',
                boxShadow: isSelected ? `0 0 10px ${color}22` : 'none',
              }}
              onMouseEnter={() => onHover(obj.detectionId)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(isSelected ? null : obj.detectionId)}
            >
              {/* Row 1: Label, Dot, Score */}
              <div className="object-card-row-top">
                <div className="object-card-label-group flex items-center gap-1.5 min-w-0">
                  <span
                    className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0"
                    style={{
                      backgroundColor: color,
                      boxShadow: `0 0 6px ${color}88`,
                    }}
                  />
                  <span className="object-card-title capitalize font-medium text-zinc-100 flex items-center gap-1.5 min-w-0">
                    <span className="truncate">{obj.modelLabel}</span>
                    <span className="font-mono text-[11px] font-semibold text-cyan-400 bg-cyan-950/70 border border-cyan-800/60 px-1.5 py-0.5 rounded lowercase flex-shrink-0">
                      {obj.instanceId || obj.detectionId}
                    </span>
                  </span>
                </div>
                <span className="object-card-score flex-shrink-0">
                  {(obj.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Row 2: Instance Identity, Category Tag, BBox Action */}
              <div className="object-card-row-bottom">
                <div className="object-card-meta flex items-center gap-1.5 flex-wrap">
                  {obj.projectCategory ? (
                    <span className="category-tag">
                      <Check size={11} /> {obj.projectCategory}
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-zinc-400 text-[11px] px-1.5 py-0.5 rounded bg-zinc-800/60 border border-zinc-700/50">
                      <Layers size={10} className="text-zinc-400" /> Detected Entity
                    </span>
                  )}
                  {obj.instanceId && obj.detectionId !== obj.instanceId && (
                    <>
                      <span>&bull;</span>
                      <span className="font-mono text-[10px] text-zinc-500">
                        slot: {obj.detectionId}
                      </span>
                    </>
                  )}
                </div>

                <button
                  type="button"
                  className="bbox-peek-btn flex items-center gap-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedBboxId(isBboxOpen ? null : obj.detectionId);
                  }}
                  title="Toggle bounding box coordinates"
                >
                  {isBboxOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                  BBox
                </button>
              </div>

              {/* Expandable BBox Coordinates on Demand */}
              {isBboxOpen && (
                <div
                  className="mt-1 pt-1 border-t border-zinc-800 text-[11px] font-mono text-zinc-400 flex items-center justify-between"
                  onClick={(e) => e.stopPropagation()}
                >
                  <span>[{obj.bbox.join(', ')}]</span>
                  <span className="text-zinc-500">
                    {Math.round(obj.bbox[2] - obj.bbox[0])} &times; {Math.round(obj.bbox[3] - obj.bbox[1])} px
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
