/**
 * BoundingBoxOverlay Component.
 * Focus-based SGG visualization overlay.
 *
 * Implements a strict visual hierarchy:
 * - Default: Faint 1.2px box outlines (0.28 opacity), NO labels, NO cluttering relationship lines.
 * - Focus mode (hover/click): Level 1 (Focus object, 1.0 opacity + complete label),
 *   Level 2 (Connected neighbors, 0.60 opacity + connected edges & predicates),
 *   Level 3 (Unrelated objects, 0.08 opacity, labels hidden).
 * - Click locks focus state; clicking empty space or Escape clears it.
 */

import React from 'react';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';
import { getPaletteColor } from '../utils/colors';

export type OverlayMode = 'focus' | 'relationships' | 'objects' | 'clean';

interface BoundingBoxOverlayProps {
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
  imgWidth: number;
  imgHeight: number;
  mode?: OverlayMode;
  masterOpacity?: number; // 0.0 to 1.0
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
}

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  objects,
  relationships,
  imgWidth,
  imgHeight,
  mode = 'focus',
  masterOpacity = 1.0,
  hoveredId,
  onHover,
  selectedId,
  onSelect,
  colorMap,
  isEntityMatch,
}) => {
  if (imgWidth === 0 || imgHeight === 0) return null;

  // Mode: Clean image shows zero overlays
  if (mode === 'clean' || masterOpacity <= 0.01) {
    return (
      <svg
        className="image-overlay-svg"
        viewBox={`0 0 ${imgWidth} ${imgHeight}`}
        preserveAspectRatio="none"
      >
        {/* Background click listener to clear any persistent selection */}
        <rect
          width={imgWidth}
          height={imgHeight}
          fill="transparent"
          onClick={() => onSelect(null)}
        />
      </svg>
    );
  }

  const checkMatch = (targetId: string | null | undefined, candidateId: string | null | undefined) => {
    if (!targetId || !candidateId) return false;
    if (isEntityMatch) return isEntityMatch(targetId, candidateId);
    return targetId === candidateId;
  };

  const boxMap = new Map<string, SGGObjectDetection>(objects.map((o) => [o.detectionId, o]));

  // Active target is locked selection if set, otherwise temporary hover
  const activeId = selectedId || hoveredId;
  const hasActive = !!activeId;

  // Identify connected relationships and neighbor object IDs
  const connectedRels: SGGRelation[] = [];
  const connectedObjIds = new Set<string>();

  if (hasActive) {
    for (const rel of relationships) {
      const isSubj = checkMatch(activeId, rel.subjectDetectionId);
      const isObj = checkMatch(activeId, rel.objectDetectionId);
      if (isSubj || isObj) {
        connectedRels.push(rel);
        connectedObjIds.add(rel.subjectDetectionId);
        connectedObjIds.add(rel.objectDetectionId);
      }
    }
  }

  // Determine which relationships to display
  let visibleRels: { rel: SGGRelation; isConnectedToActive: boolean; index: number }[] = [];
  if (mode === 'relationships') {
    visibleRels = relationships.map((rel, idx) => ({
      rel,
      isConnectedToActive: !hasActive || checkMatch(activeId, rel.subjectDetectionId) || checkMatch(activeId, rel.objectDetectionId),
      index: idx,
    }));
  } else if (mode === 'focus') {
    // In focus mode, display relationship lines ONLY when an object is hovered or selected!
    if (hasActive) {
      visibleRels = connectedRels.map((rel, idx) => ({
        rel,
        isConnectedToActive: true,
        index: idx,
      }));
    }
  }

  return (
    <svg
      className="image-overlay-svg"
      viewBox={`0 0 ${imgWidth} ${imgHeight}`}
      preserveAspectRatio="none"
    >
      <defs>
        {/* Normal Arrowhead */}
        <marker
          id="arrowhead-normal"
          markerWidth="7"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
        >
          <polygon points="0 0, 7 2.5, 0 5" fill="#38bdf8" />
        </marker>
        {/* Focused Active Arrowhead */}
        <marker
          id="arrowhead-focus"
          markerWidth="8"
          markerHeight="6"
          refX="7"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 8 3, 0 6" fill="#fbbf24" />
        </marker>
      </defs>

      {/* 0. Invisible background to catch clicks and clear selection */}
      <rect
        width={imgWidth}
        height={imgHeight}
        fill="transparent"
        onClick={() => onSelect(null)}
      />

      {/* 1. Directed Relationship Edges & Predicate Pills */}
      {visibleRels.map(({ rel, isConnectedToActive, index }) => {
        const subj = boxMap.get(rel.subjectDetectionId);
        const obj = boxMap.get(rel.objectDetectionId);
        if (!subj || !obj) return null;

        const x1 = (subj.bbox[0] + subj.bbox[2]) / 2;
        const y1 = (subj.bbox[1] + subj.bbox[3]) / 2;
        const x2 = (obj.bbox[0] + obj.bbox[2]) / 2;
        const y2 = (obj.bbox[1] + obj.bbox[3]) / 2;

        const dx = x2 - x1;
        const dy = y2 - y1;
        const dist = Math.hypot(dx, dy) || 1;

        // Anti-collision curvature: alternate offsets so overlapping pairs separate cleanly
        const curveOffset = ((index % 2 === 0 ? 1 : -1) * (0.12 + (index % 3) * 0.05)) * dist;
        const normalX = -dy / dist;
        const normalY = dx / dist;

        const cx = (x1 + x2) / 2 + normalX * curveOffset;
        const cy = (y1 + y2) / 2 + normalY * curveOffset;

        const edgeStroke = isConnectedToActive ? '#38bdf8' : '#64748b';
        const edgeOpacity = (isConnectedToActive ? 0.88 : 0.06) * masterOpacity;
        const edgeWidth = isConnectedToActive ? 2.2 : 1.0;

        const predicateLabel = rel.canonicalPredicate || rel.predicateModelLabel;
        const pillWidth = Math.max(52, predicateLabel.length * 6.5 + 14);

        // Clamp predicate label position inside image
        const pillX = Math.min(Math.max(pillWidth / 2 + 4, cx), imgWidth - pillWidth / 2 - 4);
        const pillY = Math.min(Math.max(12, cy), imgHeight - 12);

        return (
          <g key={rel.relationId} className="transition-opacity duration-150">
            {/* Relationship line */}
            <path
              d={`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`}
              fill="none"
              stroke={edgeStroke}
              strokeWidth={edgeWidth}
              strokeOpacity={edgeOpacity}
              strokeDasharray={rel.provisional ? '4 3' : 'none'}
              markerEnd={isConnectedToActive ? 'url(#arrowhead-focus)' : 'url(#arrowhead-normal)'}
            />

            {/* Predicate Label Pill (displayed when edge is prominent) */}
            {isConnectedToActive && (
              <g transform={`translate(${pillX}, ${pillY})`}>
                <rect
                  x={-pillWidth / 2}
                  y={-9}
                  width={pillWidth}
                  height={18}
                  rx={4}
                  fill="#09090b"
                  fillOpacity={0.92 * masterOpacity}
                  stroke="#3f3f46"
                  strokeWidth={1}
                  strokeOpacity={0.9 * masterOpacity}
                />
                <text
                  x={0}
                  y={3.5}
                  fill="#f1f5f9"
                  fontSize="10"
                  fontFamily="system-ui, -apple-system, sans-serif"
                  fontWeight="600"
                  textAnchor="middle"
                  style={{ userSelect: 'none' }}
                >
                  {predicateLabel}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* 2. Object Bounding Boxes & Focused Labels */}
      {objects.map((obj, i) => {
        const [x1, y1, x2, y2] = obj.bbox;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);

        const color =
          colorMap?.get(obj.detectionId) ||
          (obj.instanceId ? colorMap?.get(obj.instanceId) : undefined) ||
          getPaletteColor(i);

        // Classification into Visual Hierarchy Levels:
        const isLevel1 =
          checkMatch(activeId, obj.detectionId) ||
          (obj.instanceId ? checkMatch(activeId, obj.instanceId) : false);

        const isLevel2 =
          !isLevel1 &&
          hasActive &&
          (connectedObjIds.has(obj.detectionId) ||
            (obj.instanceId ? connectedObjIds.has(obj.instanceId) : false));

        const isLevel3 = !isLevel1 && !isLevel2 && hasActive;
        const isDefault = !hasActive;

        // Visual parameters strictly adhering to requirements:
        let strokeOpacity: number;
        let strokeWidth: number;
        let fillOpacity: number;

        if (isLevel1) {
          strokeOpacity = 1.0 * masterOpacity;
          strokeWidth = 2.4;
          fillOpacity = 0.07 * masterOpacity;
        } else if (isLevel2) {
          strokeOpacity = 0.60 * masterOpacity;
          strokeWidth = 1.8;
          fillOpacity = 0.02 * masterOpacity;
        } else if (isLevel3) {
          // Unrelated objects become barely visible (0.08 opacity)
          strokeOpacity = 0.08 * masterOpacity;
          strokeWidth = 1.0;
          fillOpacity = 0.0;
        } else {
          // Default state: very low visual prominence (0.28 opacity, 1.2px outline)
          strokeOpacity = 0.28 * masterOpacity;
          strokeWidth = 1.2;
          fillOpacity = 0.0;
        }

        // Selected / Locked indicator
        const isSelected =
          checkMatch(selectedId, obj.detectionId) ||
          (obj.instanceId ? checkMatch(selectedId, obj.instanceId) : false);

        // Label details for Level 1 focus
        const baseName = obj.projectCategory || obj.modelLabel;
        const scorePercent = (obj.confidence * 100).toFixed(0);
        const fullLabelText = `${baseName} · ${scorePercent}% · ${obj.detectionId}`;

        // Width calculation
        const labelPillWidth = fullLabelText.length * 6.8 + 24;
        const labelPillHeight = 20;

        // Position label above box, clamped inside image viewport
        const labelX = Math.min(Math.max(4, x1), Math.max(4, imgWidth - labelPillWidth - 4));
        const labelY = y1 >= 24 ? y1 : y1 + labelPillHeight + 4; // If box at top edge, place inside

        return (
          <g
            key={obj.detectionId}
            onMouseEnter={() => onHover(obj.detectionId)}
            onMouseLeave={() => onHover(null)}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(isSelected ? null : obj.detectionId);
            }}
            className="cursor-pointer"
          >
            {/* Bounding Box Outline */}
            <rect
              x={x1}
              y={y1}
              width={width}
              height={height}
              fill={color}
              fillOpacity={fillOpacity}
              stroke={color}
              strokeWidth={strokeWidth}
              strokeOpacity={strokeOpacity}
              className="transition-all duration-150"
            />

            {/* LEVEL 1: Focused Detection Label */}
            {isLevel1 && masterOpacity > 0.1 && (
              <g
                transform={`translate(${labelX}, ${labelY - labelPillHeight})`}
                className="transition-opacity duration-150"
              >
                {/* Semi-transparent dark pill background (0.92 opacity) with solid accent border */}
                <rect
                  x={0}
                  y={0}
                  width={labelPillWidth}
                  height={labelPillHeight}
                  rx={4}
                  fill="#09090b"
                  fillOpacity={0.92 * masterOpacity}
                  stroke={color}
                  strokeWidth={1.5}
                  strokeOpacity={1.0 * masterOpacity}
                />

                {/* Color Dot Indicator */}
                <circle
                  cx={10}
                  cy={labelPillHeight / 2}
                  r={3.5}
                  fill={color}
                  stroke="#ffffff"
                  strokeWidth={0.8}
                />

                {/* Compact, readable text: Object name · Confidence% · Detection ID */}
                <text
                  x={18}
                  y={13.5}
                  fill="#ffffff"
                  fontSize="11"
                  fontFamily="system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
                  fontWeight="600"
                  style={{
                    filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.9))',
                    userSelect: 'none',
                  }}
                >
                  {fullLabelText}
                </text>
              </g>
            )}

            {/* LEVEL 2: Compact Neighbor Tag for connected objects */}
            {isLevel2 && masterOpacity > 0.3 && (
              <g
                transform={`translate(${Math.min(Math.max(4, x1), imgWidth - (baseName.length * 6.5 + 14)), Math.max(16, y1)})`}
              >
                <rect
                  x={0}
                  y={-14}
                  width={baseName.length * 6.5 + 12}
                  height={15}
                  rx={3}
                  fill="#09090b"
                  fillOpacity={0.80 * masterOpacity}
                  stroke={color}
                  strokeWidth={1}
                  strokeOpacity={0.65 * masterOpacity}
                />
                <text
                  x={6}
                  y={-3}
                  fill="#e2e8f0"
                  fontSize="9.5"
                  fontFamily="system-ui, sans-serif"
                  fontWeight="500"
                  style={{ userSelect: 'none' }}
                >
                  {baseName}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
};
