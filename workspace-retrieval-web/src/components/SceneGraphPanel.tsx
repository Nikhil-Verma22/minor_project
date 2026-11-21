/**
 * SceneGraphPanel Component.
 * Visualizes the 2D Scene Graph as clean, modern triplet rows and an interactive SVG graph.
 * Uses restrained, unified color chips and subtle separators.
 */

import React, { useState } from 'react';
import { Network, List, Share2, ArrowRight, Filter } from 'lucide-react';
import { SceneGraph } from '../graph/graphTypes';
import { CANONICAL_PREDICATES } from '../config/predicates';

interface SceneGraphPanelProps {
  graph: SceneGraph | null;
  hoveredId: string | null;
  onHoverNode: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
  selectedFrames?: { frameId: string; frameIndex: number }[];
}

export const SceneGraphPanel: React.FC<SceneGraphPanelProps> = ({
  graph,
  hoveredId,
  onHoverNode,
  colorMap,
  isEntityMatch,
  selectedFrames = [],
}) => {
  const [viewMode, setViewMode] = useState<'triplets' | 'graph'>('triplets');
  const [predicateFilter, setPredicateFilter] = useState<string>('all');
  const [frameFilter, setFrameFilter] = useState<string>('all');

  if (!graph || graph.edges.length === 0) {
    return (
      <div className="ai-panel">
        <div className="ai-panel-header">
          <h3 className="ai-panel-title">
            <Network size={16} className="ai-panel-title-icon" />
            2D Scene Graph
          </h3>
        </div>
        <div className="text-center py-8 text-zinc-500 text-xs italic">
          No relationship facts detected in current observation.
        </div>
      </div>
    );
  }

  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));

  const checkMatch = (targetId: string | null | undefined, candidateId: string | null | undefined) => {
    if (!targetId || !candidateId) return false;
    if (isEntityMatch) return isEntityMatch(targetId, candidateId);
    return targetId === candidateId;
  };

  const filteredEdges = graph.edges.filter((e) => {
    if (predicateFilter !== 'all' && e.predicate !== predicateFilter) return false;
    if (frameFilter !== 'all') {
      if (e.frameId && e.frameId !== frameFilter) {
        if (!e.observedFrames || !e.observedFrames.includes(frameFilter)) return false;
      }
    }
    return true;
  });

  return (
    <div className="ai-panel">
      {/* Header with Title, Count, Predicate Filter, and View Mode Toggle */}
      <div className="ai-panel-header">
        <div className="flex items-center gap-2">
          <h3 className="ai-panel-title">
            <Network size={16} className="ai-panel-title-icon" />
            2D Scene Graph
          </h3>
          <span className="badge-counter">{graph.edges.length} triplets</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Frame Filter (if multi-frame) */}
          {selectedFrames.length > 1 && (
            <select
              className="domain-select text-xs py-1 px-2"
              value={frameFilter}
              onChange={(e) => setFrameFilter(e.target.value)}
              title="Filter by observation frame"
            >
              <option value="all">All Frames ({graph.edges.length})</option>
              {selectedFrames.map((f) => {
                const count = graph.edges.filter(
                  (e) => e.frameId === f.frameId || e.observedFrames?.includes(f.frameId)
                ).length;
                return (
                  <option key={f.frameId} value={f.frameId}>
                    Frame {f.frameIndex} ({count})
                  </option>
                );
              })}
            </select>
          )}

          {/* Predicate Filter */}
          <div className="flex items-center gap-1">
            <Filter size={12} className="text-zinc-500" />
            <select
              className="domain-select text-xs py-1 px-2"
              value={predicateFilter}
              onChange={(e) => setPredicateFilter(e.target.value)}
              title="Filter by relationship predicate"
            >
              <option value="all">All ({graph.edges.length})</option>
              {CANONICAL_PREDICATES.map((p) => {
                const count = graph.edges.filter((e) => e.predicate === p.canonical).length;
                if (count === 0) return null;
                return (
                  <option key={p.canonical} value={p.canonical}>
                    {p.displayName} ({count})
                  </option>
                );
              })}
            </select>
          </div>

          {/* View Toggle */}
          <div className="overlay-mode-segmented">
            <button
              type="button"
              className={`overlay-mode-chip ${viewMode === 'triplets' ? 'active' : ''}`}
              onClick={() => setViewMode('triplets')}
              title="Triplets List"
            >
              <List size={12} />
            </button>
            <button
              type="button"
              className={`overlay-mode-chip ${viewMode === 'graph' ? 'active' : ''}`}
              onClick={() => setViewMode('graph')}
              title="Graph View"
            >
              <Share2 size={12} />
            </button>
          </div>
        </div>
      </div>

      {/* Content: Triplets List OR Node-Edge Graph */}
      {viewMode === 'triplets' ? (
        <div className="triplets-scroll-container">
          {filteredEdges.map((edge) => {
            const subjNode = nodeMap.get(edge.sourceId);
            const objNode = nodeMap.get(edge.targetId);

            const isSubjHovered =
              checkMatch(hoveredId, edge.sourceId) ||
              (subjNode?.detectionId ? checkMatch(hoveredId, subjNode.detectionId) : false);

            const isObjHovered =
              checkMatch(hoveredId, edge.targetId) ||
              (objNode?.detectionId ? checkMatch(hoveredId, objNode.detectionId) : false);

            const isRowHovered = isSubjHovered || isObjHovered;

            const subjColor =
              colorMap?.get(edge.sourceId) ||
              (subjNode?.detectionId ? colorMap?.get(subjNode.detectionId) : undefined) ||
              '#38bdf8';

            const isSurface = objNode?.isSurface || edge.targetId.includes('surface');
            const objColor = isSurface
              ? '#34d399'
              : colorMap?.get(edge.targetId) ||
                (objNode?.detectionId ? colorMap?.get(objNode.detectionId) : undefined) ||
                '#a78bfa';

            return (
              <div
                key={edge.id}
                className={`triplet-item ${isRowHovered ? 'active' : ''}`}
                onMouseEnter={() => onHoverNode(edge.sourceId)}
                onMouseLeave={() => onHoverNode(null)}
              >
                <div className="triplet-entities">
                  {/* Subject Chip */}
                  <span
                    className="entity-chip"
                    style={{
                      backgroundColor: `${subjColor}1a`,
                      color: subjColor,
                      border: `1px solid ${isSubjHovered ? subjColor : `${subjColor}40`}`,
                    }}
                    title={subjNode?.label || edge.sourceId}
                  >
                    {subjNode?.label || edge.sourceId}
                  </span>

                  {/* Directed Predicate Badge */}
                  <span className="predicate-badge">
                    <ArrowRight size={10} className="text-cyan-400" />
                    {edge.predicate.replace(/_/g, ' ')}
                  </span>

                  {/* Object Chip */}
                  <span
                    className="entity-chip"
                    style={{
                      backgroundColor: `${objColor}1a`,
                      color: objColor,
                      border: `1px solid ${isObjHovered ? objColor : `${objColor}40`}`,
                    }}
                    title={objNode?.label || edge.targetId}
                  >
                    {objNode?.label || edge.targetId}
                  </span>
                </div>

                {/* Score & Source Metadata */}
                <div className="flex items-center gap-2">
                  {edge.observedFrames && edge.observedFrames.length > 1 ? (
                    <span
                      className="badge-counter text-[10px] font-mono text-cyan-400 bg-cyan-950/70 border border-cyan-800/60"
                      title={`Observed in: ${edge.observedFrames.join(', ')}`}
                    >
                      {edge.observedFrames.length} frames
                    </span>
                  ) : edge.frameId ? (
                    <span
                      className="badge-counter text-[10px] font-mono text-zinc-400"
                      title={`Observed in frame ${edge.frameId}`}
                    >
                      {edge.frameId.replace('seq_001_', '')}
                    </span>
                  ) : null}
                  <span className="triplet-score">{(edge.confidence * 100).toFixed(0)}%</span>
                  <span className="badge-counter text-[10px] uppercase font-mono">
                    {edge.source}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Interactive Node-Edge SVG Graph */
        <div className="bg-[#0B0D12] rounded-lg p-2 border border-zinc-800/80">
          <svg className="w-full h-56" viewBox="0 0 460 220">
            {filteredEdges.slice(0, 16).map((edge, i) => {
              const total = Math.min(filteredEdges.length, 16);
              const angle1 = (i / total) * 2 * Math.PI;
              const angle2 = ((i + 1) / total) * 2 * Math.PI;

              const x1 = 230 + 150 * Math.cos(angle1);
              const y1 = 110 + 75 * Math.sin(angle1);
              const x2 = edge.targetId.includes('surface') ? 230 : 230 + 150 * Math.cos(angle2);
              const y2 = edge.targetId.includes('surface') ? 110 : 110 + 75 * Math.sin(angle2);

              const edgeColor = colorMap?.get(edge.sourceId) || '#38bdf8';

              return (
                <g key={edge.id}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={edgeColor}
                    strokeWidth="1.2"
                    strokeOpacity="0.45"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 3}
                    fill={edgeColor}
                    fontSize="8.5"
                    fontFamily="system-ui, sans-serif"
                    textAnchor="middle"
                    fillOpacity="0.8"
                  >
                    {edge.predicate}
                  </text>
                </g>
              );
            })}

            {graph.nodes.slice(0, 12).map((node, idx) => {
              const isSurface = node.isSurface;
              const angle = (idx / Math.min(graph.nodes.length, 12)) * 2 * Math.PI;
              const cx = isSurface ? 230 : 230 + 150 * Math.cos(angle);
              const cy = isSurface ? 110 : 110 + 75 * Math.sin(angle);

              const isHovered =
                checkMatch(hoveredId, node.id) ||
                (node.detectionId ? checkMatch(hoveredId, node.detectionId) : false);

              const nodeColor = isSurface
                ? '#34d399'
                : colorMap?.get(node.id) ||
                  (node.detectionId ? colorMap?.get(node.detectionId) : undefined) ||
                  '#38bdf8';

              return (
                <g
                  key={node.id}
                  className="cursor-pointer"
                  onMouseEnter={() => onHoverNode(node.id)}
                  onMouseLeave={() => onHoverNode(null)}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSurface ? 20 : 14}
                    fill={`${nodeColor}22`}
                    stroke={isHovered ? '#fbbf24' : nodeColor}
                    strokeWidth={isHovered ? 2.2 : 1.5}
                    style={{
                      filter: isHovered ? `drop-shadow(0 0 6px ${nodeColor})` : 'none',
                    }}
                  />
                  <text
                    x={cx}
                    y={cy + 3.5}
                    fill="#ffffff"
                    fontSize="8.5"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {node.label.slice(0, 6)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
};
