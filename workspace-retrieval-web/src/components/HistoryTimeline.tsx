/**
 * HistoryTimeline Component.
 * Compact chronological timeline of workspace observation frames,
 * with JSON export and memory clearing.
 */

import React from 'react';
import { History, Download, Trash2, Clock } from 'lucide-react';
import { FrameRecord } from '../memory/memoryTypes';
import { formatHumanTime } from '../utils/time';
import { ExportImportService } from '../memory/exportImport';

interface HistoryTimelineProps {
  frames: FrameRecord[];
  currentFrameId: string | null;
  onSelectFrame: (frameId: string) => void;
  onClearMemory: () => void;
  onRefreshHistory: () => void;
}

export const HistoryTimeline: React.FC<HistoryTimelineProps> = ({
  frames,
  currentFrameId,
  onSelectFrame,
  onClearMemory,
  onRefreshHistory,
}) => {
  const handleExport = async () => {
    try {
      const json = await ExportImportService.exportAllToJson();
      ExportImportService.downloadJson(json, `workspace-memory-${Date.now()}.json`);
    } catch (err) {
      console.error('Failed to export memory:', err);
    }
  };

  // Display most recent observation first
  const reversedFrames = [...frames].reverse();

  return (
    <div className="ai-panel">
      {/* Header */}
      <div className="ai-panel-header">
        <div className="flex items-center gap-2">
          <History size={16} className="text-indigo-400" />
          <h3 className="ai-panel-title">Workspace History</h3>
          <span className="badge-counter">{frames.length}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="btn-secondary-action text-xs py-1 px-2.5"
            onClick={handleExport}
            disabled={frames.length === 0}
            title="Export all memory records to JSON"
          >
            <Download size={12} /> Export
          </button>
          <button
            type="button"
            className="btn-secondary-action text-xs py-1 px-2 text-rose-400 hover:text-rose-300"
            onClick={onClearMemory}
            disabled={frames.length === 0}
            title="Clear workspace memory history"
          >
            <Trash2 size={12} /> Clear
          </button>
        </div>
      </div>

      {frames.length === 0 ? (
        <div className="text-center py-6 text-zinc-500 text-xs italic">
          No sequential observation frames recorded yet.
        </div>
      ) : (
        <div className="timeline-scroll-container">
          {reversedFrames.map((f, idx) => {
            const isSelected = f.frameId === currentFrameId;
            const observationNumber = frames.length - idx;

            return (
              <div
                key={f.frameId}
                className={`history-item ${isSelected ? 'active' : ''}`}
                onClick={() => onSelectFrame(f.frameId)}
                title="Click to view this observation and its scene graph"
              >
                <div className="history-left">
                  <span className={`history-bullet ${isSelected ? 'active' : ''}`} />
                  <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white text-xs">
                        Observation {observationNumber.toString().padStart(2, '0')}
                      </span>
                      <span className="history-frame-id text-[11px] text-zinc-400">
                        {f.frameId}
                      </span>
                    </div>
                    <div className="history-meta flex items-center gap-2 mt-0.5">
                      <span className="flex items-center gap-1 text-[11px]">
                        <Clock size={10} /> {formatHumanTime(f.timestamp)}
                      </span>
                      <span>&bull;</span>
                      <span className="text-[11px] font-mono">
                        {f.imageWidth} &times; {f.imageHeight}
                      </span>
                    </div>
                  </div>
                </div>

                {isSelected && (
                  <span className="badge-local-first text-[10px] py-0.5 px-2">
                    Active
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
