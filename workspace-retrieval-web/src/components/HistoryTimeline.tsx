/**
 * HistoryTimeline Component.
 * Chronological timeline of workspace observation frames and memory export/clear actions.
 */

import React from 'react';
import { History, Download, Trash2, Calendar, FileText } from 'lucide-react';
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

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <History size={18} className="text-indigo-400" />
          <h2 className="panel-title">Workspace Observation History</h2>
          <span className="badge-count">{frames.length} frames</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className="btn-secondary text-xs flex items-center gap-1"
            onClick={handleExport}
            disabled={frames.length === 0}
            title="Export all memory records to JSON"
          >
            <Download size={13} /> Export JSON
          </button>
          <button
            type="button"
            className="btn-danger text-xs flex items-center gap-1"
            onClick={onClearMemory}
            disabled={frames.length === 0}
            title="Clear all local memory"
          >
            <Trash2 size={13} /> Clear
          </button>
        </div>
      </div>

      {frames.length === 0 ? (
        <div className="empty-state">
          No observation frames recorded yet. Upload an image to start history.
        </div>
      ) : (
        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
          {frames.map((f, idx) => {
            const isSelected = f.frameId === currentFrameId;
            return (
              <div
                key={f.frameId}
                className={`timeline-card ${isSelected ? 'timeline-card-active' : ''}`}
                onClick={() => onSelectFrame(f.frameId)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="timeline-index">#{idx + 1}</span>
                    <span className="font-mono text-xs font-semibold text-zinc-200">
                      {f.frameId}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] text-zinc-400">
                    <Calendar size={12} />
                    {formatHumanTime(f.timestamp)}
                  </div>
                </div>

                <div className="flex items-center gap-3 mt-1 text-[11px] text-zinc-400">
                  <span>Sequence: {f.sequenceId}</span>
                  <span>Dimensions: {f.imageWidth} &times; {f.imageHeight}</span>
                  {f.filename && (
                    <span className="flex items-center gap-1 text-zinc-500">
                      <FileText size={11} /> {f.filename}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
