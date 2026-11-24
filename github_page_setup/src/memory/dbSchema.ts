/**
 * IndexedDB Schema & Store Names.
 */

export const DB_NAME = 'workspace_retrieval_db';
export const DB_VERSION = 2;

export const STORES = {
  WORKSPACES: 'workspaces',
  SEQUENCES: 'sequences',
  FRAMES: 'frames',
  OBJECT_INSTANCES: 'objectInstances',
  OBJECT_OBSERVATIONS: 'objectObservations',
  RELATIONSHIPS: 'relationships',
  GRAPH_EVENTS: 'graphEvents',
  VLM_CANDIDATES: 'vlmCandidates',
  QUERY_HISTORY: 'queryHistory',
  APP_SETTINGS: 'appSettings',
  SESSION_LOGS: 'sessionLogs',
} as const;
