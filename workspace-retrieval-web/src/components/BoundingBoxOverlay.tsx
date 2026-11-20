/**
 * BoundingBoxOverlay Component.
 * SVG overlay rendering bounding boxes, labels, confidence scores, and directed relationship arrows.
 */

import React from 'react';
import { SGGObjectDetection, SGGRelation } from '../models/sgg/SGGEngine';

interface BoundingBoxOverlayProps {
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
  imgWidth: number;
  imgHeight: number;
  showLabels: boolean;
  showScores: boolean;
  showInstanceIds: boolean;
  showRelations: boolean;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const CATEGORY_COLORS = [
  '#38bdf8', // sky-400
  '#34d399', // emerald-400
  '#f472b6', // pink-400
  '#fbbf24', // amber-400
  '#a78bfa', // violet-400
  '#f87171', // red-400
  '#2dd4bf', // teal-400
  '#fb923c', // orange-400
];

export const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({
  objects,
  relationships,
  imgWidth,
  imgHeight,
  showLabels,
  showScores,
  showInstanceIds,
  showRelations,
  hoveredId,
  onHover,
  selectedId,
  onSelect,
}) => {
  if (imgWidth === 0 || imgHeight === 0) return null;

  const boxMap = new Map<string, SGGObjectDetection>(objects.map((o) => [o.detectionId, o]));

  const getColor = (index: number) => CATEGORY_COLORS[index % CATEGORY_COLORS.length];

  return (
    <svg
      className="image-overlay-svg"
      viewBox={`0 0 ${imgWidth} ${imgHeight}`}
      preserveAspectRatio="none"
    >
      <defs>
        <marker
          id="arrowhead"
          markerWidth="8"
          markerHeight="6"
          refX="7"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 8 3, 0 6" fill="#38bdf8" />
        </marker>
        <marker
          id="arrowhead-highlight"
          markerWidth="9"
          markerHeight="7"
          refX="8"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 9 3.5, 0 7" fill="#f59e0b" />
        </marker>
      </defs>

      {/* 1. Directed Relationship Curves */}
      {showRelations &&
        relationships.map((rel) => {
          const subj = boxMap.get(rel.subjectDetectionId);
          const obj = boxMap.get(rel.objectDetectionId);
          if (!subj || !obj) return null;

          const x1 = (subj.bbox[0] + subj.bbox[2]) / 2;
          const y1 = (subj.bbox[1] + subj.bbox[3]) / 2;
          const x2 = (obj.bbox[0] + obj.bbox[2]) / 2;
          const y2 = (obj.bbox[1] + obj.bbox[3]) / 2;

          const isHighlighted =
            hoveredId === rel.subjectDetectionId ||
            hoveredId === rel.objectDetectionId ||
            selectedId === rel.subjectDetectionId ||
            selectedId === rel.objectDetectionId;

          const dx = x2 - x1;
          const dy = y2 - y1;
          const cx = (x1 + x2) / 2 - dy * 0.15;
          const cy = (y1 + y2) / 2 + dx * 0.15;

          const predicateLabel = rel.canonicalPredicate || rel.predicateModelLabel;

          return (
            <g key={rel.relationId} className="transition-opacity duration-200">
              <path
                d={`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`}
                fill="none"
                stroke={isHighlighted ? '#f59e0b' : '#38bdf8'}
                strokeWidth={isHighlighted ? 3 : 1.5}
                strokeDasharray={rel.provisional ? '4 3' : 'none'}
                markerEnd={isHighlighted ? 'url(#arrowhead-highlight)' : 'url(#arrowhead)'}
                opacity={isHighlighted ? 1 : 0.65}
              />
              {/* Predicate Label pill */}
              <rect
                x={cx - Math.max(30, (predicateLabel.length * 6.5 + 12) / 2)}
                y={cy - 10}
                width={Math.max(60, predicateLabel.length * 6.5 + 12)}
                height={18}
                rx={4}
                fill="#09090b"
                stroke={isHighlighted ? '#f59e0b' : '#27272a'}
                strokeWidth={1}
                opacity={0.88}
              />
              <text
                x={cx}
                y={cy + 3}
                fill={isHighlighted ? '#fef08a' : '#e4e4e7'}
                fontSize="10"
                fontFamily="system-ui, sans-serif"
                fontWeight="500"
                textAnchor="middle"
              >
                {predicateLabel}
              </text>
            </g>
          );
        })}

      {/* 2. Object Bounding Boxes */}
      {objects.map((obj, i) => {
        const [x1, y1, x2, y2] = obj.bbox;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const color = getColor(i);
        const isHovered = hoveredId === obj.detectionId;
        const isSelected = selectedId === obj.detectionId;

        const baseName = obj.projectCategory || obj.modelLabel;
        const labelText = showInstanceIds ? `${baseName} (${obj.detectionId})` : baseName;
        const scoreText = showScores ? ` ${(obj.confidence * 100).toFixed(0)}%` : '';

        return (
          <g
            key={obj.detectionId}
            onMouseEnter={() => onHover(obj.detectionId)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onSelect(isSelected ? null : obj.detectionId)}
            className="cursor-pointer"
          >
            {/* Box Rect */}
            <rect
              x={x1}
              y={y1}
              width={width}
              height={height}
              fill={isSelected || isHovered ? color : 'none'}
              fillOpacity={isSelected ? 0.25 : isHovered ? 0.15 : 0}
              stroke={color}
              strokeWidth={isSelected ? 3 : isHovered ? 2.5 : 1.8}
              strokeDasharray={obj.projectCategory ? 'none' : '4 2'}
              className="transition-all duration-150"
            />

            {/* Label Chip */}
            {showLabels && (
              <g transform={`translate(${x1}, ${Math.max(16, y1)})`}>
                <rect
                  x={0}
                  y={-16}
                  width={labelText.length * 7 + (showScores ? 32 : 12)}
                  height={18}
                  rx={3}
                  fill={color}
                  fillOpacity={0.92}
                />
                <text
                  x={4}
                  y={-3}
                  fill="#09090b"
                  fontSize="11"
                  fontFamily="system-ui, sans-serif"
                  fontWeight="600"
                >
                  {labelText}
                  {scoreText}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
};
