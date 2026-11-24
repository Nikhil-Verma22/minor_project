/**
 * FrameSelector Component.
 * Simplified, intuitive timeline filmstrip for multi-frame workspace observation sessions.
 *
 * Requirements:
 * - One click or tap anywhere on a frame card toggles its selection (select/deselect).
 * - Exactly one primary upload control: "+ Upload Images" (supports 1 or many images).
 * - Safe individual frame deletion with compact confirmation dialog.
 * - Selection changes never trigger SGG inference.
 */

import React, { useRef, useState } from 'react';
import {
  Film,
  CheckSquare,
  Square,
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
  onUploadImages: (files: FileList | File[]) => void;
  onDeleteFrame?: (frameId: string) => void;
  isProcessing?: boolean;
}

export const FrameSelector: React.FC<FrameSelectorProps> = ({
  frames,
  selectedFrameIds,
  onToggleFrameSelect,
  onSelectAll,
  onClearSelection,
  onUploadImages,
  onDeleteFrame,
  isProcessing = false,
}) => {
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [frameToDelete, setFrameToDelete] = useState<FrameItem | null>(null);

  const handleUploadChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadImages(e.target.files);
      e.target.value = '';
    }
  };

  const selectedCount = selectedFrameIds.length;
  const totalCount = frames.length;

  return (
    <div className="frame-selector-container relative">
      {/* Hidden Unified Multi-File Input */}
      <input
        ref={uploadInputRef}
        type="file"
        multiple
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        style={{ display: 'none' }}
        onChange={handleUploadChange}
      />

      {/* Confirmation Modal for Frame Deletion */}
      {frameToDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-fadeIn"
          onClick={() => setFrameToDelete(null)}
        >
          <div
            className="bg-zinc-900 border border-zinc-700 p-5 rounded-lg max-w-sm w-full shadow-2xl space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 text-rose-400 font-semibold text-sm">
              <Trash2 size={16} />
              <h4>Delete Frame {frameToDelete.frameIndex}?</h4>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed">
              This will remove the selected observation, its image, and its frame-specific annotations.
              Other observations in this workspace will remain intact.
            </p>
            <div className="flex justify-end gap-2 pt-2 border-t border-zinc-800">
              <button
                type="button"
                className="px-3 py-1.5 text-xs text-zinc-300 hover:text-white rounded bg-zinc-800 hover:bg-zinc-700 transition"
                onClick={() => setFrameToDelete(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="px-3 py-1.5 text-xs text-white bg-rose-600 hover:bg-rose-500 rounded font-medium transition flex items-center gap-1.5"
                onClick={() => {
                  const id = frameToDelete.frameId;
                  setFrameToDelete(null);
                  onDeleteFrame?.(id);
                }}
              >
                <Trash2 size={12} />
                Delete Frame
              </button>
            </div>
          </div>
        </div>
      )}

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

          {/* Unified Primary Upload Control */}
          <button
            type="button"
            className="btn-add-frame-primary"
            onClick={() => uploadInputRef.current?.click()}
            disabled={isProcessing}
            title="Upload single or multiple images into observation sequence"
          >
            <Upload size={13} />
            <span>+ Upload Images</span>
          </button>
        </div>
      </div>

      {/* Filmstrip Scroll Row */}
      {totalCount === 0 ? (
        <div className="filmstrip-empty">
          <span className="text-xs text-zinc-500 italic">
            No frames in sequence. Click <strong>+ Upload Images</strong> or drag &amp; drop photos anywhere to begin.
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
                className={`frame-card group ${isSelected ? 'selected' : ''}`}
                onClick={() => onToggleFrameSelect(frame.frameId)}
                title={`Frame ${frame.frameIndex} (${frame.frameId})\nClick to ${isSelected ? 'deselect' : 'select'}`}
              >
                {/* Thumbnail Header with Selection Indicator and Delete Button */}
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

                  {/* Top-Left Selection Indicator Pill */}
                  <div
                    className={`frame-card-checkbox ${isSelected ? 'selected' : ''}`}
                    title={isSelected ? 'Selected (click card to deselect)' : 'Unselected (click card to select)'}
                  >
                    {isSelected ? (
                      <CheckSquare size={15} className="text-cyan-400" />
                    ) : (
                      <Square size={15} className="text-zinc-500 hover:text-zinc-300" />
                    )}
                  </div>

                  {/* Top-Right Delete Action Button */}
                  {onDeleteFrame && (
                    <button
                      type="button"
                      className="frame-card-delete-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFrameToDelete(frame);
                      }}
                      title="Permanently delete this observation frame"
                    >
                      <Trash2 size={12} />
                    </button>
                  )}

                  {/* Frame Index Badge */}
                  <span className="frame-index-badge">
                    Frame {frame.frameIndex}
                  </span>
                </div>

                {/* Card Meta Info */}
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
