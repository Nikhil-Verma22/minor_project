/**
 * ObjectList Component.
 * Displays detected objects, model labels, mapped workspace categories, and confidence scores.
 */

import React from 'react';
import { Box, Check, HelpCircle } from 'lucide-react';
import { SGGObjectDetection } from '../models/sgg/SGGEngine';

interface ObjectListProps {
  objects: SGGObjectDetection[];
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export const ObjectList: React.FC<ObjectListProps> = ({
  objects,
  hoveredId,
  onHover,
  selectedId,
  onSelect,
}) => {
  if (objects.length === 0) {
    return (
      <div className="panel-card">
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <Box size={18} className="text-cyan-400" />
            <h2 className="panel-title">Detected Objects</h2>
          </div>
        </div>
        <div className="empty-state">No objects detected in current observation.</div>
      </div>
    );
  }

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Box size={18} className="text-cyan-400" />
          <h2 className="panel-title">Detected Objects</h2>
          <span className="badge-count">{objects.length}</span>
        </div>
      </div>

      <div className="overflow-x-auto max-h-60 overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-zinc-900/80 text-zinc-400 uppercase tracking-wider sticky top-0">
            <tr>
              <th className="py-2 px-3">Object / ID</th>
              <th className="py-2 px-3">Model Label</th>
              <th className="py-2 px-3">Workspace Category</th>
              <th className="py-2 px-3">Score</th>
              <th className="py-2 px-3">BBox</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {objects.map((obj) => {
              const isHovered = hoveredId === obj.detectionId;
              const isSelected = selectedId === obj.detectionId;

              return (
                <tr
                  key={obj.detectionId}
                  className={`cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-amber-500/15'
                      : isHovered
                      ? 'bg-zinc-800/50'
                      : 'hover:bg-zinc-900/50'
                  }`}
                  onMouseEnter={() => onHover(obj.detectionId)}
                  onMouseLeave={() => onHover(null)}
                  onClick={() => onSelect(isSelected ? null : obj.detectionId)}
                >
                  <td className="py-2 px-3 font-medium text-zinc-200">
                    {obj.detectionId}
                  </td>
                  <td className="py-2 px-3 text-zinc-400">{obj.modelLabel}</td>
                  <td className="py-2 px-3">
                    {obj.projectCategory ? (
                      <span className="inline-flex items-center gap-1 text-emerald-400">
                        <Check size={12} /> {obj.projectCategory}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-zinc-500">
                        <HelpCircle size={12} /> unmapped
                      </span>
                    )}
                  </td>
                  <td className="py-2 px-3 font-semibold text-cyan-400">
                    {(obj.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-2 px-3 font-mono text-[11px] text-zinc-500">
                    [{obj.bbox.join(', ')}]
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
