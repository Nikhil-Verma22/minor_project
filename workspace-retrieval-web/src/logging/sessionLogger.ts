/**
 * Session Logger Subsystem.
 * Provides high-throughput, non-blocking, structured session event logging.
 * Decoupled from application components and fully persistent in IndexedDB.
 */

import { generateId } from '../utils/ids';
import { SessionLogEvent, SessionLogLevel, SessionLogCategory } from './logTypes';
import { LogRepository } from './logRepository';

type LogSubscriber = (event: SessionLogEvent) => void;

class SessionLoggerImpl {
  private sessionId: string;
  private subscribers: Set<LogSubscriber> = new Set();
  private buffer: SessionLogEvent[] = [];
  private flushTimer: any = null;
  private currentWorkspaceId: string = 'workspace_default';
  private currentSequenceId: string = 'seq_001';
  private activeSelectedFrameIds: string[] = [];

  constructor() {
    this.sessionId = this.initSessionId();
  }

  private initSessionId(): string {
    const STORAGE_KEY = 'workspace_active_session_id';
    try {
      let id = localStorage.getItem(STORAGE_KEY);
      if (!id) {
        id = `sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
        localStorage.setItem(STORAGE_KEY, id);
      }
      return id;
    } catch {
      return `sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
    }
  }

  public getSessionId(): string {
    return this.sessionId;
  }

  public startNewSession(): string {
    this.flush();
    this.sessionId = `sess_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 6)}`;
    try {
      localStorage.setItem('workspace_active_session_id', this.sessionId);
    } catch {}
    this.info('application', 'SESSION_STARTED', `Started new session ${this.sessionId}`, {
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node/Unknown',
    });
    return this.sessionId;
  }

  public setContext(context: {
    workspaceId?: string;
    sequenceId?: string;
    selectedFrameIds?: string[];
  }): void {
    if (context.workspaceId) this.currentWorkspaceId = context.workspaceId;
    if (context.sequenceId) this.currentSequenceId = context.sequenceId;
    if (context.selectedFrameIds) this.activeSelectedFrameIds = [...context.selectedFrameIds];
  }

  public subscribe(cb: LogSubscriber): () => void {
    this.subscribers.add(cb);
    return () => this.subscribers.delete(cb);
  }

  public log(event: {
    level: SessionLogLevel;
    category: SessionLogCategory;
    eventType: string;
    message: string;
    workspaceId?: string;
    sequenceId?: string;
    frameId?: string;
    queryId?: string;
    selectedFrameIds?: string[];
    details?: Record<string, unknown>;
    durationMs?: number;
  }): SessionLogEvent {
    const fullEvent: SessionLogEvent = {
      logId: generateId('log'),
      sessionId: this.sessionId,
      timestamp: new Date().toISOString(),
      level: event.level,
      category: event.category,
      eventType: event.eventType,
      workspaceId: event.workspaceId || this.currentWorkspaceId,
      sequenceId: event.sequenceId || this.currentSequenceId,
      frameId: event.frameId,
      queryId: event.queryId,
      selectedFrameIds: event.selectedFrameIds || (this.activeSelectedFrameIds.length > 0 ? this.activeSelectedFrameIds : undefined),
      message: event.message,
      details: event.details,
      durationMs: event.durationMs,
    };

    // Buffer event
    this.buffer.push(fullEvent);

    // Notify in-memory subscribers (e.g. live telemetry)
    for (const sub of this.subscribers) {
      try {
        sub(fullEvent);
      } catch (err) {
        console.warn('[SessionLogger] Error in subscriber callback:', err);
      }
    }

    // Flush immediately for error/warn or critical events
    if (
      event.level === 'error' ||
      event.level === 'warn' ||
      event.eventType.includes('COMPLETED') ||
      event.eventType.includes('DELETED') ||
      this.buffer.length >= 10
    ) {
      this.flush();
    } else {
      this.scheduleFlush();
    }

    return fullEvent;
  }

  public info(
    category: SessionLogCategory,
    eventType: string,
    message: string,
    details?: Record<string, unknown>,
    meta?: { frameId?: string; queryId?: string; selectedFrameIds?: string[]; durationMs?: number }
  ): SessionLogEvent {
    return this.log({
      level: 'info',
      category,
      eventType,
      message,
      details,
      ...meta,
    });
  }

  public warn(
    category: SessionLogCategory,
    eventType: string,
    message: string,
    details?: Record<string, unknown>,
    meta?: { frameId?: string; queryId?: string; selectedFrameIds?: string[]; durationMs?: number }
  ): SessionLogEvent {
    return this.log({
      level: 'warn',
      category,
      eventType,
      message,
      details,
      ...meta,
    });
  }

  public error(
    category: SessionLogCategory,
    eventType: string,
    message: string,
    details?: Record<string, unknown>,
    meta?: { frameId?: string; queryId?: string; selectedFrameIds?: string[]; durationMs?: number }
  ): SessionLogEvent {
    return this.log({
      level: 'error',
      category,
      eventType,
      message,
      details,
      ...meta,
    });
  }

  public debug(
    category: SessionLogCategory,
    eventType: string,
    message: string,
    details?: Record<string, unknown>,
    meta?: { frameId?: string; queryId?: string; selectedFrameIds?: string[]; durationMs?: number }
  ): SessionLogEvent {
    return this.log({
      level: 'debug',
      category,
      eventType,
      message,
      details,
      ...meta,
    });
  }

  /**
   * High-precision duration timer helper.
   */
  public startTimer(
    category: SessionLogCategory,
    eventType: string,
    message: string,
    details?: Record<string, unknown>,
    meta?: { frameId?: string; queryId?: string; selectedFrameIds?: string[] }
  ): (detailsUpdate?: Record<string, unknown>) => number {
    const startTime = performance.now();
    return (detailsUpdate?: Record<string, unknown>) => {
      const durationMs = Math.round((performance.now() - startTime) * 10) / 10;
      this.info(
        category,
        eventType,
        `${message} (${durationMs} ms)`,
        { ...details, ...detailsUpdate },
        { ...meta, durationMs }
      );
      return durationMs;
    };
  }

  private scheduleFlush(): void {
    if (this.flushTimer) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.flush();
    }, 500);
  }

  public flush(): void {
    if (this.buffer.length === 0) return;
    const batch = [...this.buffer];
    this.buffer = [];
    LogRepository.addLogsBatch(batch);
  }
}

export const sessionLogger = new SessionLoggerImpl();
