/**
 * SceneGraphPanel Component.
 * Visualizes 2D Scene Graph as categorized triplets and an interactive node-edge graph.
 */

import React, { useState } from 'react';
import { Network, List, Share2, ArrowRight } from 'lucide-react';
import { SceneGraph } from '../graph/graphTypes';
import { CANONICAL_PREDICATES } from '../config/predicates';

interface SceneGraphPanelProps {
  graph: SceneGraph | null;
  hoveredId: string | null;
  onHoverNode: (id: string | null) => void;
}

export const SceneGraphPanel: React.FC<SceneGraphPanelProps> = ({
  graph,
  hoveredId,
  onHoverNode,
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
            const isHovered = hoveredId === edge.sourceId || hoveredId === edge.targetId;

            return (
              <div
                key={edge.id}
                className={`triplet-row ${isHovered ? 'triplet-hovered' : ''}`}
                onMouseEnter={() => onHoverNode(edge.sourceId)}
                onMouseLeave={() => onHoverNode(null)}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className="node-chip node-subject">
                    {subjNode?.label || edge.sourceId}
                  </span>

                  <span className="predicate-pill flex items-center gap-1">
                    <ArrowRight size={11} className="text-cyan-400" />
                    {edge.predicate.replace(/_/g, ' ')}
                  </span>

                  <span className={`node-chip ${objNode?.isSurface ? 'node-surface' : 'node-object'}`}>
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

              return (
                <g key={edge.id}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke="#38bdf8"
                    strokeWidth="1.5"
                    strokeOpacity="0.6"
                  />
                  <text
                    x={(x1 + x2) / 2}
                    y={(y1 + y2) / 2 - 4}
                    fill="#38bdf8"
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
              const isHovered = hoveredId === node.id;

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
                    fill={isSurface ? '#3b82f6' : isHovered ? '#f59e0b' : '#1e293b'}
                    stroke={isHovered ? '#fbbf24' : '#64748b'}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                  />
                  <text
                    x={cx}
                    y={cy + 4}
                    fill="#ffffff"
                    fontSize="9"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {node.label.slice(0, 5)}
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
