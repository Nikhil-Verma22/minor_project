/**
 * ImageUploader Component.
 * Supports image file drag/drop, web image drag/drop, local browsing, and sample image selection.
 */

import React, { useRef, useState } from 'react';
import { Upload, Image as ImageIcon, Play, PlusCircle, Sparkles } from 'lucide-react';
import { WORKSPACE_DOMAINS } from '../config/workspaceDomains';

interface ImageUploaderProps {
  onImageSelected: (file: File | Blob, filename: string) => void;
  onRunSgg: () => void;
  onAddNextObservation: () => void;
  selectedDomain: string;
  onDomainChange: (domain: string) => void;
  hasImage: boolean;
  isRunning: boolean;
  hasPriorObservations: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  onRunSgg,
  onAddNextObservation,
  selectedDomain,
  onDomainChange,
  hasImage,
  isRunning,
  hasPriorObservations,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File | Blob, name = 'uploaded_image.jpg') => {
    onImageSelected(file, name);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    // 1. Check local dropped files
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        handleFile(file, file.name);
        return;
      }
    }

    // 2. Check dragged web image URLs (e.g. from Bing, Google Images)
    const uri = e.dataTransfer.getData('text/uri-list');
    const html = e.dataTransfer.getData('text/html');

    let imageUrl = uri;
    if (!imageUrl && html) {
      const match = html.match(/src=["'](https?:\/\/[^"']+)["']/i);
      if (match) imageUrl = match[1];
    }

    if (imageUrl) {
      try {
        const res = await fetch(imageUrl);
        const blob = await res.blob();
        handleFile(blob, 'web_dropped_image.jpg');
      } catch (err) {
        console.warn('Could not fetch dragged web image directly (likely CORS), loading via canvas:', err);
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0);
          canvas.toBlob((b) => {
            if (b) handleFile(b, 'web_image.jpg');
          }, 'image/jpeg');
        };
        img.src = imageUrl;
      }
    }
  };

  const loadSampleImage = async () => {
    try {
      const res = await fetch('/samples/sample_workspace_1.jpg');
      const blob = await res.blob();
      handleFile(blob, 'sample_workspace_1.jpg');
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  return (
    <div className="panel-card">
      <div className="panel-header">
        <div className="flex items-center gap-2">
          <Upload size={18} className="text-cyan-400" />
          <h2 className="panel-title">RGB Workspace Observation</h2>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-400">Domain:</label>
          <select
            className="select-input"
            value={selectedDomain}
            onChange={(e) => onDomainChange(e.target.value)}
          >
            {WORKSPACE_DOMAINS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div
        className={`dropzone ${dragOver ? 'dropzone-active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0], e.target.files[0].name);
            }
          }}
        />
        <ImageIcon size={36} className="text-zinc-500 mb-2" />
        <div className="text-sm font-medium text-zinc-200">
          Drop workspace photo here, or <span className="text-cyan-400 underline">browse</span>
        </div>
        <div className="text-xs text-zinc-400 mt-1">Supports PNG, JPG, WebP (dragged from computer or web search)</div>
      </div>

      <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-800">
        <button
          type="button"
          className="btn-secondary text-xs flex items-center gap-1.5"
          onClick={loadSampleImage}
        >
          <Sparkles size={14} className="text-amber-400" /> Load Sample Workspace Image
        </button>

        <div className="flex items-center gap-2">
          {hasPriorObservations && (
            <button
              type="button"
              className="btn-secondary text-xs flex items-center gap-1.5"
              onClick={onAddNextObservation}
              disabled={!hasImage || isRunning}
            >
              <PlusCircle size={14} /> Add as Next Observation
            </button>
          )}

          <button
            type="button"
            className="btn-primary flex items-center gap-2"
            onClick={onRunSgg}
            disabled={!hasImage || isRunning}
          >
            <Play size={16} fill="currentColor" />
            {isRunning ? 'Processing Scene...' : 'Run Scene Understanding'}
          </button>
        </div>
      </div>
    </div>
  );
};
