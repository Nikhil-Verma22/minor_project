/**
 * Pipeline Types and Stage Lifecycle.
 */

import { SGGInferenceResult } from '../models/sgg/SGGEngine';
import { SceneGraph } from '../graph/graphTypes';
import { NLQResult } from '../nlq/queryTypes';
import { VLMProgress } from '../models/vlm/VLMEngine';

export type PipelineStage =
  | 'idle'
  | 'image_loaded'
  | 'sgg_running'
  | 'sgg_completed'
  | 'identity_association'
  | 'graph_building'
  | 'memory_update'
  | 'ready_for_query'
  | 'query_parsing'
  | 'graph_memory_reasoning'
  | 'vlm_loading'
  | 'vlm_reasoning'
  | 'answer_ready'
  | 'error';

export interface PipelineCallbacks {
  onStageChange?: (stage: PipelineStage, message?: string) => void;
  onSggResult?: (result: SGGInferenceResult) => void;
  onGraphUpdate?: (graph: SceneGraph) => void;
  onQueryResult?: (result: NLQResult) => void;
  onVlmProgress?: (progress: VLMProgress) => void;
  onError?: (error: string) => void;
}
