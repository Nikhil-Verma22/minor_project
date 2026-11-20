/**
 * SceneGraphPanel Component.
 * Visualizes 2D Scene Graph as categorized triplets and an interactive node-edge graph.
 * Uses unified object colors matching the BoundingBoxOverlay and ObjectList.
 */

import React, { useState } from 'react';
import { Network, List, Share2, ArrowRight } from 'lucide-react';
import { SceneGraph } from '../graph/graphTypes';
import { CANONICAL_PREDICATES } from '../config/predicates';

interface SceneGraphPanelProps {
  graph: SceneGraph | null;
  hoveredId: string | null;
  onHoverNode: (id: string | null) => void;
  colorMap?: Map<string, string>;
  isEntityMatch?: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
}

export const SceneGraphPanel: React.FC<SceneGraphPanelProps> = ({
  graph,
  hoveredId,
  onHoverNode,
  colorMap,
  isEntityMatch,
}) => {
  const [viewMode, setViewMode] = useState<'triplets' | 'graph'>('triplets');
  const [predicateFilter, setPredicateFilter] = useState<string>('all');

  if (!graph || graph.edges.length === 0) {
    return (
      <div className="panel-card">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Network size={18} className="text-cyan-400" />
            <h2 className="panel-title">2D Scene Graph</h2>
          </div>
        </div>
        <div className="empty-state">
          No relationship facts detected in this observation yet.
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
    if (predicateFilter === 'all') return true;
    return e.predicate === predicateFilter;
  });

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Network size={18} className="text-cyan-400" />
          <h2 className="panel-title">2D Scene Graph</h2>
          <span className="badge-count">{graph.edges.length} triplets</span>
        </div>

        <div className="flex items-center gap-2">
          <select
            className="select-input text-xs"
            value={predicateFilter}
            onChange={(e) => setPredicateFilter(e.target.value)}
          >
            <option value="all">All Predicates ({graph.edges.length})</option>
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

          <div className="view-toggle">
            <button
              type="button"
              className={`view-toggle-btn ${viewMode === 'triplets' ? 'active' : ''}`}
              onClick={() => setViewMode('triplets')}
              title="Triplets List"
            >
              <List size={14} />
            </button>
            <button
              type="button"
              className={`view-toggle-btn ${viewMode === 'graph' ? 'active' : ''}`}
              onClick={() => setViewMode('graph')}
              title="Graph View"
            >
              <Share2 size={14} />
            </button>
          </div>
        </div>
      </div>

      {viewMode === 'triplets' ? (
        <div className="triplets-list">
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
                className={`triplet-row ${isRowHovered ? 'triplet-hovered' : ''}`}
                onMouseEnter={() => onHoverNode(edge.sourceId)}
                onMouseLeave={() => onHoverNode(null)}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span
                    className="node-chip transition-all duration-150"
                    style={{
                      backgroundColor: `${subjColor}22`,
                      color: subjColor,
                      border: `1px solid ${isSubjHovered ? subjColor : `${subjColor}55`}`,
                      boxShadow: isSubjHovered ? `0 0 8px ${subjColor}77` : 'none',
                    }}
                  >
                    {subjNode?.label || edge.sourceId}
                  </span>

                  <span className="predicate-pill flex items-center gap-1">
                    <ArrowRight size={11} className="text-zinc-400" />
                    {edge.predicate.replace(/_/g, ' ')}
                  </span>

                  <span
                    className="node-chip transition-all duration-150"
                    style={{
                      backgroundColor: `${objColor}22`,
                      color: objColor,
                      border: `1px solid ${isObjHovered ? objColor : `${objColor}55`}`,
                      boxShadow: isObjHovered ? `0 0 8px ${objColor}77` : 'none',
                    }}
                  >
                    {objNode?.label || edge.targetId}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <span className="text-zinc-400">{(edge.confidence * 100).toFixed(0)}%</span>
                  <span className={`source-badge ${edge.source === 'vlm' ? 'source-vlm' : 'source-sgg'}`}>
                    {edge.source}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Interactive Node-Edge SVG Graph */
        <div className="graph-container">
          <svg className="w-full h-64" viewBox="0 0 500 240">
            {filteredEdges.slice(0, 15).map((edge, i) => {
              const total = Math.min(filteredEdges.length, 15);
              const angle1 = (i / total) * 2 * Math.PI;
              const angle2 = ((i + 1) / total) * 2 * Math.PI;

              const x1 = 250 + 160 * Math.cos(angle1);
              const y1 = 120 + 80 * Math.sin(angle1);
              const x2 = edge.targetId.includes('surface') ? 250 : 250 + 160 * Math.cos(angle2);
              const y2 = edge.targetId.includes('surface') ? 120 : 120 + 80 * Math.sin(angle2);

              const edgeColor = colorMap?.get(edge.sourceId) || '#38bdf8';

              return (
                <g key={edge.id}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={edgeColor}
                    strokeWidth="1.5"
                    strokeOpacity="0.6"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 4}
                    fill={edgeColor}
                    fontSize="9"
                    textAnchor="middle"
                  >
                    {edge.predicate}
                  </text>
                </g>
              );
            })}

            {graph.nodes.slice(0, 12).map((node, idx) => {
              const isSurface = node.isSurface;
              const angle = (idx / Math.min(graph.nodes.length, 12)) * 2 * Math.PI;
              const cx = isSurface ? 250 : 250 + 160 * Math.cos(angle);
              const cy = isSurface ? 120 : 120 + 80 * Math.sin(angle);

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
                    r={isSurface ? 22 : 16}
                    fill={`${nodeColor}28`}
                    stroke={isHovered ? '#fbbf24' : nodeColor}
                    strokeWidth={isHovered ? 2.5 : 1.8}
                    style={{
                      filter: isHovered ? `drop-shadow(0 0 6px ${nodeColor})` : 'none',
                    }}
                  />
                  <text
                    x={cx}
                    y={cy + 4}
                    fill="#ffffff"
                    fontSize="9"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {node.label.slice(0, 7)}
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
