/**
 * FrameSelector Component.
 * Dedicated timeline / filmstrip for multi-frame workspace observation sessions.
 * Supports multi-selection, frame status indicators, thumbnail previews,
 * and batch / sequential frame additions.
 */

import React, { useRef } from 'react';
import {
  Film,
  CheckSquare,
  Square,
  Plus,
  Upload,
  Trash2,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Box,
} from 'lucide-react';
import { formatHumanTime } from '../utils/time';

export interface FrameItem {
  frameId: string;
  frameIndex: number;
  timestamp: string;
  imageWidth: number;
  imageHeight: number;
  thumbnailDataUrl?: string;
  imageSrc?: string;
  status?: 'pending' | 'processing' | 'completed' | 'failed';
  objectCount?: number;
  relationCount?: number;
}

export interface FrameSelectorProps {
  frames: FrameItem[];
  selectedFrameIds: string[];
  onToggleFrameSelect: (frameId: string) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onSelectOnlyFrame: (frameId: string) => void;
  onAddNextFrame: () => void;
  onBatchUpload: (files: FileList | File[]) => void;
  onDeleteFrame?: (frameId: string) => void;
  isProcessing?: boolean;
}

export const FrameSelector: React.FC<FrameSelectorProps> = ({
  frames,
  selectedFrameIds,
  onToggleFrameSelect,
  onSelectAll,
  onClearSelection,
  onSelectOnlyFrame,
  onAddNextFrame,
  onBatchUpload,
  onDeleteFrame,
  isProcessing = false,
}) => {
  const batchInputRef = useRef<HTMLInputElement>(null);

  const handleBatchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onBatchUpload(e.target.files);
      e.target.value = '';
    }
  };

  const selectedCount = selectedFrameIds.length;
  const totalCount = frames.length;

  return (
    <div className="frame-selector-container">
      {/* Hidden Batch File Input */}
      <input
        ref={batchInputRef}
        type="file"
        multiple
        accept="image/*"
        className="hidden"
        style={{ display: 'none' }}
        onChange={handleBatchChange}
      />

      {/* Header Bar */}
      <div className="frame-selector-header">
        <div className="flex items-center gap-2">
          <Film size={14} className="text-cyan-400" />
          <span className="font-semibold text-xs tracking-wider uppercase text-zinc-300">
            Observation Timeline
          </span>
          <span className="badge-counter text-[11px]">
            {selectedCount} of {totalCount} selected
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {totalCount > 0 && (
            <>
              <button
                type="button"
                className="btn-filmstrip-action"
                onClick={onSelectAll}
                title="Select all frames"
              >
                <CheckSquare size={12} /> Select All
              </button>
              <button
                type="button"
                className="btn-filmstrip-action"
                onClick={onClearSelection}
                title="Clear frame selection"
              >
                <Square size={12} /> Clear
              </button>
              <span className="text-zinc-600 px-1">&bull;</span>
            </>
          )}

          <button
            type="button"
            className="btn-add-frame-primary"
            onClick={onAddNextFrame}
            disabled={isProcessing}
            title="Append next workspace photograph to observation sequence"
          >
            <Plus size={13} />
            <span>Add Next Frame</span>
          </button>

          <button
            type="button"
            className="btn-batch-upload"
            onClick={() => batchInputRef.current?.click()}
            disabled={isProcessing}
            title="Select multiple workspace photographs to process sequentially"
          >
            <Upload size={13} />
            <span>Batch Upload</span>
          </button>
        </div>
      </div>

      {/* Filmstrip Scroll Row */}
      {totalCount === 0 ? (
        <div className="filmstrip-empty">
          <span className="text-xs text-zinc-500 italic">
            No frames in sequence. Click <strong>+ Add Next Frame</strong> or <strong>Batch Upload</strong> to begin.
          </span>
        </div>
      ) : (
        <div className="frame-cards-scroll-strip">
          {frames.map((frame) => {
            const isSelected = selectedFrameIds.includes(frame.frameId);
            const status = frame.status || 'completed';

            return (
              <div
                key={frame.frameId}
                className={`frame-card ${isSelected ? 'selected' : ''}`}
                onClick={(e) => {
                  // If clicking directly on the card, toggle or select only
                  if (e.shiftKey || e.metaKey || e.ctrlKey) {
                    onToggleFrameSelect(frame.frameId);
                  } else {
                    onSelectOnlyFrame(frame.frameId);
                  }
                }}
                title={`Frame ${frame.frameIndex} (${frame.frameId})\nClick to view single frame; click checkbox to multi-select`}
              >
                {/* Thumbnail Header with Checkbox and Delete */}
                <div className="frame-card-thumb-container">
                  {frame.thumbnailDataUrl || frame.imageSrc ? (
                    <img
                      src={frame.thumbnailDataUrl || frame.imageSrc}
                      alt={`Frame ${frame.frameIndex}`}
                      className="frame-card-img"
                    />
                  ) : (
                    <div className="frame-card-img-placeholder">
                      <Box size={24} className="text-zinc-600" />
                    </div>
                  )}

                  {/* Multi-select Checkbox */}
                  <button
                    type="button"
                    className="frame-card-checkbox"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleFrameSelect(frame.frameId);
                    }}
                    title={isSelected ? 'Deselect frame' : 'Select frame'}
                  >
                    {isSelected ? (
                      <CheckSquare size={15} className="text-cyan-400" />
                    ) : (
                      <Square size={15} className="text-zinc-500 hover:text-zinc-300" />
                    )}
                  </button>

                  {/* Delete Button (if provided) */}
                  {onDeleteFrame && (
                    <button
                      type="button"
                      className="frame-card-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteFrame(frame.frameId);
                      }}
                      title="Remove this frame from observation sequence"
                    >
                      <Trash2 size={12} />
                    </button>
                  )}

                  {/* Frame Index Overlay Badge */}
                  <span className="frame-index-badge">
                    Frame {frame.frameIndex}
                  </span>
                </div>

                {/* Card Meta */}
                <div className="frame-card-info">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="flex items-center gap-1 text-zinc-400">
                      <Clock size={10} />
                      {formatHumanTime(frame.timestamp)}
                    </span>

                    {/* Status Badge */}
                    {status === 'completed' && (
                      <span className="text-[10px] text-emerald-400 flex items-center gap-0.5">
                        <CheckCircle2 size={10} /> Done
                      </span>
                    )}
                    {status === 'processing' && (
                      <span className="text-[10px] text-amber-400 flex items-center gap-0.5 animate-pulse">
                        <Loader2 size={10} className="animate-spin" /> Processing
                      </span>
                    )}
                    {status === 'pending' && (
                      <span className="text-[10px] text-cyan-400 flex items-center gap-0.5">
                        <Clock size={10} /> Queued
                      </span>
                    )}
                    {status === 'failed' && (
                      <span className="text-[10px] text-rose-400 flex items-center gap-0.5">
                        <AlertCircle size={10} /> Failed
                      </span>
                    )}
                  </div>

                  {frame.objectCount !== undefined && (
                    <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
                      {frame.objectCount} obj &bull; {frame.relationCount ?? 0} rel
                    </div>
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
