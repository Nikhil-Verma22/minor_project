/**
 * ID generator utilities conforming to project schema.
 */

let counter = 1000;

export function generateId(prefix: string): string {
  counter += 1;
  const rand = Math.random().toString(36).substring(2, 6);
  return `${prefix}_${Date.now().toString(36)}_${rand}`;
}

export function formatFrameId(sequenceId: string, frameNumber: number): string {
  const pad = String(frameNumber).padStart(3, '0');
  return `${sequenceId}_frame_${pad}`;
}

export function generateInstanceId(categoryId: string, sequenceNumber: number): string {
  const pad = String(sequenceNumber).padStart(3, '0');
  return `${categoryId}_${pad}`;
}
