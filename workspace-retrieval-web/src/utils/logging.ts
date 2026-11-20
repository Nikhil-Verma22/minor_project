/**
 * Structured logger utility.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export function logMessage(level: LogLevel, module: string, message: string, data?: unknown): void {
  const timestamp = new Date().toISOString().slice(11, 23);
  const prefix = `[${timestamp}] [${module}]`;

  switch (level) {
    case 'debug':
      console.debug(prefix, message, data !== undefined ? data : '');
      break;
    case 'info':
      console.info(prefix, message, data !== undefined ? data : '');
      break;
    case 'warn':
      console.warn(prefix, message, data !== undefined ? data : '');
      break;
    case 'error':
      console.error(prefix, message, data !== undefined ? data : '');
      break;
  }
}
