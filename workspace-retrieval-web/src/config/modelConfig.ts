/**
 * Model configuration: SGG checkpoints, inference thresholds, and VLM fallback settings.
 */

export interface ModelVariantConfig {
  id: string;
  name: string;
  path: string;
  sizeBytes: number;
  description: string;
}

export const SGG_VARIANTS: Record<string, ModelVariantConfig> = {
  yolo12n: {
    id: 'yolo12n',
    name: 'REACT++ YOLOv12n (PSG)',
    path: '/models/react_pp_yolo12n.onnx',
    sizeBytes: 45362728,
    description: 'Compact 45.4 MB checkpoint optimized for browser execution',
  },
};

export const DEFAULT_THRESHOLDS = {
  boxConfidence: 0.25,
  relationConfidence: 0.05,
  identityAssociationIou: 0.35,
};

export const VLM_CONFIG = {
  modelId: 'executorch-community/SmolVLM-500M-Instruct-ExecuTorch-WebGPU',
  totalPackageSizeMB: 518.2,
  maxContextTokens: 512,
  cacheKeyPrefix: 'smolvlm-execu-v1',
};
