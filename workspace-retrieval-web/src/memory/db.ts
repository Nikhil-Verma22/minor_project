/**
 * IndexedDB Database Connection via idb.
 */

import { openDB, IDBPDatabase } from 'idb';
import { DB_NAME, DB_VERSION, STORES } from './dbSchema';
import { logMessage } from '../utils/logging';

let dbPromise: Promise<IDBPDatabase> | null = null;

export async function getDatabase(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        logMessage('info', 'IndexedDB', `Upgrading database ${DB_NAME} to v${DB_VERSION}`);

        // 1. Workspaces
        if (!db.objectStoreNames.contains(STORES.WORKSPACES)) {
          db.createObjectStore(STORES.WORKSPACES, { keyPath: 'workspaceId' });
        }

        // 2. Sequences
        if (!db.objectStoreNames.contains(STORES.SEQUENCES)) {
          const s = db.createObjectStore(STORES.SEQUENCES, { keyPath: 'sequenceId' });
          s.createIndex('workspaceId', 'workspaceId', { unique: false });
        }

        // 3. Frames
        if (!db.objectStoreNames.contains(STORES.FRAMES)) {
          const f = db.createObjectStore(STORES.FRAMES, { keyPath: 'frameId' });
          f.createIndex('sequenceId', 'sequenceId', { unique: false });
          f.createIndex('workspaceId', 'workspaceId', { unique: false });
          f.createIndex('timestamp', 'timestamp', { unique: false });
        }

        // 4. Object Instances
        if (!db.objectStoreNames.contains(STORES.OBJECT_INSTANCES)) {
          const oi = db.createObjectStore(STORES.OBJECT_INSTANCES, { keyPath: 'instanceId' });
          oi.createIndex('workspaceId', 'workspaceId', { unique: false });
          oi.createIndex('categoryId', 'categoryId', { unique: false });
          oi.createIndex('workspace_category', ['workspaceId', 'categoryId'], { unique: false });
        }

        // 5. Object Observations
        if (!db.objectStoreNames.contains(STORES.OBJECT_OBSERVATIONS)) {
          const obs = db.createObjectStore(STORES.OBJECT_OBSERVATIONS, { keyPath: 'annotationId' });
          obs.createIndex('frameId', 'frameId', { unique: false });
          obs.createIndex('instanceId', 'instanceId', { unique: false });
          obs.createIndex('categoryId', 'categoryId', { unique: false });
        }

        // 6. Relationships
        if (!db.objectStoreNames.contains(STORES.RELATIONSHIPS)) {
          const r = db.createObjectStore(STORES.RELATIONSHIPS, { keyPath: 'relationshipId' });
          r.createIndex('frameId', 'frameId', { unique: false });
          r.createIndex('workspaceId', 'workspaceId', { unique: false });
          r.createIndex('subjectId', 'subjectId', { unique: false });
          r.createIndex('objectId', 'objectId', { unique: false });
          r.createIndex('frame_predicate', ['frameId', 'predicate'], { unique: false });
        }

        // 7. Graph Events
        if (!db.objectStoreNames.contains(STORES.GRAPH_EVENTS)) {
          const ev = db.createObjectStore(STORES.GRAPH_EVENTS, { keyPath: 'eventId' });
          ev.createIndex('workspaceId', 'workspaceId', { unique: false });
          ev.createIndex('instanceId', 'instanceId', { unique: false });
          ev.createIndex('timestamp', 'timestamp', { unique: false });
        }

        // 8. VLM Candidates
        if (!db.objectStoreNames.contains(STORES.VLM_CANDIDATES)) {
          const v = db.createObjectStore(STORES.VLM_CANDIDATES, { keyPath: 'candidateId' });
          v.createIndex('frameId', 'frameId', { unique: false });
          v.createIndex('workspaceId', 'workspaceId', { unique: false });
          v.createIndex('status', 'status', { unique: false });
        }

        // 9. Query History
        if (!db.objectStoreNames.contains(STORES.QUERY_HISTORY)) {
          const q = db.createObjectStore(STORES.QUERY_HISTORY, { keyPath: 'queryId' });
          q.createIndex('timestamp', 'timestamp', { unique: false });
        }

        // 10. App Settings
        if (!db.objectStoreNames.contains(STORES.APP_SETTINGS)) {
          db.createObjectStore(STORES.APP_SETTINGS, { keyPath: 'settingKey' });
        }
      },
    });
  }
  return dbPromise;
}
