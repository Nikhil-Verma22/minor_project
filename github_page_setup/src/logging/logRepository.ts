/**
 * Log Repository.
 * Manages atomic persistence and retrieval of structured session logs in IndexedDB.
 */

import { getDatabase } from '../memory/db';
import { STORES } from '../memory/dbSchema';
import { SessionLogEvent, SessionSummary, LogFilter } from './logTypes';

export class LogRepository {
  /**
   * Persists a single log event to IndexedDB.
   */
  static async addLog(event: SessionLogEvent): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
      const db = await getDatabase();
      await db.put(STORES.SESSION_LOGS, event);
    } catch (err) {
      console.warn('[LogRepository] Failed to persist log event to IndexedDB:', err);
    }
  }

  /**
   * Persists a batch of log events atomically.
   */
  static async addLogsBatch(events: SessionLogEvent[]): Promise<void> {
    if (events.length === 0 || typeof indexedDB === 'undefined') return;
    try {
      const db = await getDatabase();
      const tx = db.transaction(STORES.SESSION_LOGS, 'readwrite');
      for (const ev of events) {
        await tx.store.put(ev);
      }
      await tx.done;
    } catch (err) {
      console.warn('[LogRepository] Failed to persist log batch to IndexedDB:', err);
    }
  }

  /**
   * Retrieves log events with optional filtering.
   */
  static async getLogs(filter?: LogFilter): Promise<SessionLogEvent[]> {
    try {
      const db = await getDatabase();
      let allLogs: SessionLogEvent[] = [];

      if (filter?.sessionId) {
        const index = db.transaction(STORES.SESSION_LOGS).store.index('sessionId');
        allLogs = await index.getAll(filter.sessionId);
      } else {
        allLogs = await db.getAll(STORES.SESSION_LOGS);
      }

      // Sort descending by timestamp (newest first)
      allLogs.sort((a, b) => b.timestamp.localeCompare(a.timestamp));

      // Apply in-memory filters
      let filtered = allLogs;

      if (filter?.category && filter.category !== 'all') {
        filtered = filtered.filter((l) => l.category === filter.category);
      }

      if (filter?.level && filter.level !== 'all') {
        filtered = filtered.filter((l) => l.level === filter.level);
      }

      if (filter?.frameId) {
        filtered = filtered.filter((l) => l.frameId === filter.frameId || l.selectedFrameIds?.includes(filter.frameId!));
      }

      if (filter?.queryId) {
        filtered = filtered.filter((l) => l.queryId === filter.queryId);
      }

      if (filter?.eventType) {
        filtered = filtered.filter((l) => l.eventType === filter.eventType);
      }

      if (filter?.searchTerm) {
        const q = filter.searchTerm.toLowerCase();
        filtered = filtered.filter(
          (l) =>
            l.message.toLowerCase().includes(q) ||
            l.eventType.toLowerCase().includes(q) ||
            (l.frameId && l.frameId.toLowerCase().includes(q)) ||
            (l.queryId && l.queryId.toLowerCase().includes(q))
        );
      }

      if (filter?.limit && filter.limit > 0) {
        filtered = filtered.slice(0, filter.limit);
      }

      return filtered;
    } catch (err) {
      console.warn('[LogRepository] Failed to retrieve logs:', err);
      return [];
    }
  }

  /**
   * Computes an aggregated session summary for telemetry.
   */
  static async getSessionSummary(sessionId: string): Promise<SessionSummary> {
    const logs = await this.getLogs({ sessionId });
    const summary: SessionSummary = {
      sessionId,
      startTime: logs.length > 0 ? logs[logs.length - 1].timestamp : new Date().toISOString(),
      lastActivityTime: logs.length > 0 ? logs[0].timestamp : new Date().toISOString(),
      totalEvents: logs.length,
      errorCount: 0,
      warnCount: 0,
      uploadCount: 0,
      inferenceCount: 0,
      queryCount: 0,
      frameCount: 0,
    };

    const countedFrames = new Set<string>();

    for (const log of logs) {
      if (log.level === 'error') summary.errorCount++;
      if (log.level === 'warn') summary.warnCount++;
      if (log.category === 'upload') summary.uploadCount++;
      if (log.category === 'inference' && log.eventType === 'INFERENCE_COMPLETED') summary.inferenceCount++;
      if (log.category === 'query' && log.eventType === 'QUERY_COMPLETED') summary.queryCount++;
      if (log.frameId && !countedFrames.has(log.frameId)) {
        countedFrames.add(log.frameId);
        summary.frameCount++;
      }
    }

    return summary;
  }

  /**
   * Clears logs, optionally restricted to a specific session.
   * Does NOT touch workspaces, frames, or observations.
   */
  static async clearLogs(sessionId?: string): Promise<void> {
    try {
      const db = await getDatabase();
      if (!sessionId) {
        await db.clear(STORES.SESSION_LOGS);
      } else {
        const tx = db.transaction(STORES.SESSION_LOGS, 'readwrite');
        const index = tx.store.index('sessionId');
        let cursor = await index.openCursor(sessionId);
        while (cursor) {
          await cursor.delete();
          cursor = await cursor.continue();
        }
        await tx.done;
      }
    } catch (err) {
      console.warn('[LogRepository] Failed to clear logs:', err);
    }
  }
}

