/**
 * Session Logging & Audit Data Types.
 * Captures the complete operational lifecycle of workspace observations,
 * model inference, detected objects, scene graphs, frame selections, queries, answers, and errors.
 */

export type SessionLogLevel = 'debug' | 'info' | 'warn' | 'error';

export type SessionLogCategory =
  | 'application'
  | 'upload'
  | 'frame'
  | 'inference'
  | 'detection'
  | 'identity'
  | 'scene_graph'
  | 'memory'
  | 'query'
  | 'vlm'
  | 'deletion'
  | 'performance'
  | 'error';

export interface SessionLogEvent {
  logId: string;
  sessionId: string;
  timestamp: string; // ISO 8601 UTC
  level: SessionLogLevel;
  category: SessionLogCategory;
  eventType: string;

  workspaceId?: string;
  sequenceId?: string;
  frameId?: string;
  queryId?: string;

  selectedFrameIds?: string[];

  message: string;
  details?: Record<string, unknown>;
  durationMs?: number;
}

export interface SessionSummary {
  sessionId: string;
  startTime: string;
  lastActivityTime: string;
  totalEvents: number;
  errorCount: number;
  warnCount: number;
  uploadCount: number;
  inferenceCount: number;
  queryCount: number;
  frameCount: number;
}

export interface LogFilter {
  sessionId?: string;
  category?: SessionLogCategory | 'all';
  level?: SessionLogLevel | 'all';
  eventType?: string;
  frameId?: string;
  queryId?: string;
  searchTerm?: string;
  startTime?: string;
  endTime?: string;
  limit?: number;
}
