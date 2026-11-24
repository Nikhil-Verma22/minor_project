/**
 * Time utility functions.
 * Adheres to ISO 8601 formatting with timezone offset.
 */

export function getCurrentIsoTimestamp(): string {
  const date = new Date();
  const tzOffsetMinutes = -date.getTimezoneOffset();
  const diffHours = Math.floor(Math.abs(tzOffsetMinutes) / 60);
  const diffMins = Math.abs(tzOffsetMinutes) % 60;
  const sign = tzOffsetMinutes >= 0 ? '+' : '-';
  const tzString = `${sign}${String(diffHours).padStart(2, '0')}:${String(diffMins).padStart(2, '0')}`;

  const iso = date.toISOString().slice(0, 19);
  return `${iso}${tzString}`;
}

export function formatHumanTime(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return isoString;
  }
}
