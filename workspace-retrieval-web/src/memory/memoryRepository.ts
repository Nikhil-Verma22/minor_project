/**
 * Memory Repository: Transactional operations and queries over IndexedDB stores.
 */

import { getDatabase } from './db';
import { STORES } from './dbSchema';
import {
  WorkspaceRecord,
  SequenceRecord,
  FrameRecord,
  PersistentObjectInstance,
  ObjectObservation,
  RelationshipFact,
  GraphEvent,
  VLMCandidate,
} from './memoryTypes';
import { logMessage } from '../utils/logging';

export class MemoryRepository {
  /**
   * Transactionally saves an entire observation frame and its derived entities.
   */
  static async saveObservationFrameTransaction(payload: {
    workspace: WorkspaceRecord;
    sequence: SequenceRecord;
    frame: FrameRecord;
    newInstances: PersistentObjectInstance[];
    updatedInstances: PersistentObjectInstance[];
    observations: ObjectObservation[];
    relationships: RelationshipFact[];
    events: GraphEvent[];
  }): Promise<void> {
    const db = await getDatabase();
    const tx = db.transaction(
      [
        STORES.WORKSPACES,
        STORES.SEQUENCES,
        STORES.FRAMES,
        STORES.OBJECT_INSTANCES,
        STORES.OBJECT_OBSERVATIONS,
        STORES.RELATIONSHIPS,
        STORES.GRAPH_EVENTS,
      ],
      'readwrite'
    );

    try {
      await tx.objectStore(STORES.WORKSPACES).put(payload.workspace);
      await tx.objectStore(STORES.SEQUENCES).put(payload.sequence);
      await tx.objectStore(STORES.FRAMES).put(payload.frame);

      const instStore = tx.objectStore(STORES.OBJECT_INSTANCES);
      for (const inst of payload.newInstances) {
        await instStore.put(inst);
      }
      for (const inst of payload.updatedInstances) {
        await instStore.put(inst);
      }

      const obsStore = tx.objectStore(STORES.OBJECT_OBSERVATIONS);
      for (const obs of payload.observations) {
        await obsStore.put(obs);
      }

      const relStore = tx.objectStore(STORES.RELATIONSHIPS);
      for (const rel of payload.relationships) {
        await relStore.put(rel);
      }

      const evStore = tx.objectStore(STORES.GRAPH_EVENTS);
      for (const ev of payload.events) {
        await evStore.put(ev);
      }

      await tx.done;
      logMessage('info', 'MemoryRepository', `Successfully committed frame ${payload.frame.frameId} to IndexedDB`);
    } catch (err) {
      logMessage('error', 'MemoryRepository', 'Transaction failed, aborting', err);
      tx.abort();
      throw err;
    }
  }

  static async getWorkspaces(): Promise<WorkspaceRecord[]> {
    const db = await getDatabase();
    return db.getAll(STORES.WORKSPACES);
  }

  static async getWorkspace(workspaceId: string): Promise<WorkspaceRecord | undefined> {
    const db = await getDatabase();
    return db.get(STORES.WORKSPACES, workspaceId);
  }

  static async getFrames(workspaceId: string): Promise<FrameRecord[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.FRAMES).store.index('workspaceId');
    const frames = await index.getAll(workspaceId);
    return frames.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  static async getFrame(frameId: string): Promise<FrameRecord | undefined> {
    const db = await getDatabase();
    return db.get(STORES.FRAMES, frameId);
  }

  static async getInstances(workspaceId: string): Promise<PersistentObjectInstance[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.OBJECT_INSTANCES).store.index('workspaceId');
    return index.getAll(workspaceId);
  }

  static async getObservationsForFrame(frameId: string): Promise<ObjectObservation[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.OBJECT_OBSERVATIONS).store.index('frameId');
    return index.getAll(frameId);
  }

  static async getObservationsForInstance(instanceId: string): Promise<ObjectObservation[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.OBJECT_OBSERVATIONS).store.index('instanceId');
    const obs = await index.getAll(instanceId);
    return obs.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  static async getRelationshipsForFrame(frameId: string): Promise<RelationshipFact[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.RELATIONSHIPS).store.index('frameId');
    return index.getAll(frameId);
  }

  static async getRelationshipsForInstance(instanceId: string): Promise<RelationshipFact[]> {
    const db = await getDatabase();
    const subjIdx = db.transaction(STORES.RELATIONSHIPS).store.index('subjectId');
    const objIdx = db.transaction(STORES.RELATIONSHIPS).store.index('objectId');

    const [asSubject, asObject] = await Promise.all([
      subjIdx.getAll(instanceId),
      objIdx.getAll(instanceId),
    ]);

    const combined = [...asSubject, ...asObject];
    return combined.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  static async getEvents(workspaceId: string): Promise<GraphEvent[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.GRAPH_EVENTS).store.index('workspaceId');
    const events = await index.getAll(workspaceId);
    return events.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }

  static async saveVlmCandidate(candidate: VLMCandidate): Promise<void> {
    const db = await getDatabase();
    await db.put(STORES.VLM_CANDIDATES, candidate);
  }

  static async getVlmCandidates(frameId: string): Promise<VLMCandidate[]> {
    const db = await getDatabase();
    const index = db.transaction(STORES.VLM_CANDIDATES).store.index('frameId');
    return index.getAll(frameId);
  }

  static async clearWorkspace(workspaceId: string): Promise<void> {
    const db = await getDatabase();
    const tx = db.transaction(
      [
        STORES.WORKSPACES,
        STORES.SEQUENCES,
        STORES.FRAMES,
        STORES.OBJECT_INSTANCES,
        STORES.OBJECT_OBSERVATIONS,
        STORES.RELATIONSHIPS,
        STORES.GRAPH_EVENTS,
        STORES.VLM_CANDIDATES,
      ],
      'readwrite'
    );

    // Delete workspace
    await tx.objectStore(STORES.WORKSPACES).delete(workspaceId);

    // Helper to clear store records matching workspaceId
    const clearByWorkspace = async (storeName: string, indexName: string) => {
      const store = tx.objectStore(storeName as any);
      const index = store.index(indexName);
      let cursor = await index.openCursor(workspaceId);
      while (cursor) {
        await cursor.delete();
        cursor = await cursor.continue();
      }
    };

    await clearByWorkspace(STORES.SEQUENCES, 'workspaceId');
    await clearByWorkspace(STORES.FRAMES, 'workspaceId');
    await clearByWorkspace(STORES.OBJECT_INSTANCES, 'workspaceId');
    await clearByWorkspace(STORES.GRAPH_EVENTS, 'workspaceId');
    await clearByWorkspace(STORES.VLM_CANDIDATES, 'workspaceId');

    await tx.done;
    logMessage('info', 'MemoryRepository', `Cleared workspace ${workspaceId}`);
  }

  static async clearAll(): Promise<void> {
    const db = await getDatabase();
    const stores = Object.values(STORES);
    const tx = db.transaction(stores, 'readwrite');
    for (const store of stores) {
      await tx.objectStore(store).clear();
    }
    await tx.done;
    logMessage('info', 'MemoryRepository', 'Cleared all local memory stores');
  }
}
