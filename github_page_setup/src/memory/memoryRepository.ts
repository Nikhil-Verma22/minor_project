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

  static async saveFrame(frame: FrameRecord): Promise<void> {
    const db = await getDatabase();
    await db.put(STORES.FRAMES, frame);
  }

  static async updateFrameStatus(frameId: string, status: 'pending' | 'processing' | 'completed' | 'failed'): Promise<void> {
    const db = await getDatabase();
    const frame = await db.get(STORES.FRAMES, frameId);
    if (frame) {
      frame.status = status;
      await db.put(STORES.FRAMES, frame);
    }
  }

  static async deleteFrame(frameId: string): Promise<void> {
    const db = await getDatabase();
    const tx = db.transaction(
      [
        STORES.FRAMES,
        STORES.OBJECT_OBSERVATIONS,
        STORES.RELATIONSHIPS,
        STORES.GRAPH_EVENTS,
        STORES.VLM_CANDIDATES,
        STORES.OBJECT_INSTANCES,
      ],
      'readwrite'
    );

    // Delete frame
    await tx.objectStore(STORES.FRAMES).delete(frameId);

    // Delete observations for this frame and track affected instance IDs
    const affectedInstanceIds = new Set<string>();
    const obsStore = tx.objectStore(STORES.OBJECT_OBSERVATIONS);
    const obsIndex = obsStore.index('frameId');
    let obsCursor = await obsIndex.openCursor(frameId);
    while (obsCursor) {
      if (obsCursor.value.instanceId) {
        affectedInstanceIds.add(obsCursor.value.instanceId);
      }
      await obsCursor.delete();
      obsCursor = await obsCursor.continue();
    }

    // Delete relationships for this frame
    const relIndex = tx.objectStore(STORES.RELATIONSHIPS).index('frameId');
    let relCursor = await relIndex.openCursor(frameId);
    while (relCursor) {
      await relCursor.delete();
      relCursor = await relCursor.continue();
    }

    // Delete graph events for this frame (safe fallback if index is missing)
    const evStore = tx.objectStore(STORES.GRAPH_EVENTS);
    if (evStore.indexNames.contains('frameId')) {
      const evIndex = evStore.index('frameId');
      let evCursor = await evIndex.openCursor(frameId);
      while (evCursor) {
        await evCursor.delete();
        evCursor = await evCursor.continue();
      }
    } else {
      let evCursor = await evStore.openCursor();
      while (evCursor) {
        if ((evCursor.value as any).frameId === frameId) {
          await evCursor.delete();
        }
        evCursor = await evCursor.continue();
      }
    }

    // Delete vlm candidates for this frame
    const vlmStore = tx.objectStore(STORES.VLM_CANDIDATES);
    if (vlmStore.indexNames.contains('frameId')) {
      const vlmIndex = vlmStore.index('frameId');
      let vlmCursor = await vlmIndex.openCursor(frameId);
      while (vlmCursor) {
        await vlmCursor.delete();
        vlmCursor = await vlmCursor.continue();
      }
    }

    // Handle affected physical object instances: mark inactive if orphaned (0 observations remaining)
    const instStore = tx.objectStore(STORES.OBJECT_INSTANCES);
    for (const instId of affectedInstanceIds) {
      const remainingObs = await obsStore.index('instanceId').getAll(instId);
      if (remainingObs.length === 0) {
        const inst = await instStore.get(instId);
        if (inst) {
          inst.status = 'inactive';
          await instStore.put(inst);
        }
      }
    }

    await tx.done;
    logMessage('info', 'MemoryRepository', `Successfully deleted frame ${frameId} and its frame-specific observations/relations`);
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

  static async getObservationsForFrames(frameIds: string[]): Promise<ObjectObservation[]> {
    const results: ObjectObservation[] = [];
    for (const fid of frameIds) {
      const obs = await this.getObservationsForFrame(fid);
      results.push(...obs);
    }
    return results;
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

  static async getRelationshipsForFrames(frameIds: string[]): Promise<RelationshipFact[]> {
    const results: RelationshipFact[] = [];
    for (const fid of frameIds) {
      const rels = await this.getRelationshipsForFrame(fid);
      results.push(...rels);
    }
    return results;
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
