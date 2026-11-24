/**
 * Log Export Utilities.
 * Exports structured session logs locally to JSON or CSV for inspection and analysis.
 */

import { LogRepository } from './logRepository';
import { SessionLogEvent } from './logTypes';

export async function exportLogsAsJson(sessionId?: string): Promise<void> {
  const logs = await LogRepository.getLogs(sessionId ? { sessionId } : undefined);
  const summary = sessionId ? await LogRepository.getSessionSummary(sessionId) : null;

  const exportPayload = {
    exportDate: new Date().toISOString(),
    sessionId: sessionId || 'all_sessions',
    summary,
    totalRecords: logs.length,
    events: logs,
  };

  const jsonStr = JSON.stringify(exportPayload, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `session_logs_${sessionId || 'export'}_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function exportLogsAsCsv(sessionId?: string): Promise<void> {
  const logs = await LogRepository.getLogs(sessionId ? { sessionId } : undefined);
  if (logs.length === 0) return;

  const headers = [
    'logId',
    'sessionId',
    'timestamp',
    'level',
    'category',
    'eventType',
    'workspaceId',
    'frameId',
    'queryId',
    'selectedFrameIds',
    'durationMs',
    'message',
  ];

  const escapeCsv = (val: any) => {
    if (val === null || val === undefined) return '';
    const str = Array.isArray(val) ? val.join(';') : String(val);
    return `"${str.replace(/"/g, '""')}"`;
  };

  const rows = logs.map((l) =>
    [
      l.logId,
      l.sessionId,
      l.timestamp,
      l.level,
      l.category,
      l.eventType,
      l.workspaceId || '',
      l.frameId || '',
      l.queryId || '',
      l.selectedFrameIds ? l.selectedFrameIds.join(';') : '',
      l.durationMs !== undefined ? l.durationMs : '',
      l.message,
    ]
      .map(escapeCsv)
      .join(',')
  );

  const csvContent = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `session_logs_${sessionId || 'export'}_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
