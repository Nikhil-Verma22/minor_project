/**
 * Export and Import utility for Memory records.
 */

import { getDatabase } from './db';
import { STORES } from './dbSchema';
import { logMessage } from '../utils/logging';

export interface MemoryBackup {
  version: number;
  exportedAt: string;
  stores: Record<string, any[]>;
}

export class ExportImportService {
  /**
   * Export all database records to a serializable JSON object.
   */
  static async exportAllToJson(): Promise<string> {
    const db = await getDatabase();
    const backup: MemoryBackup = {
      version: 1,
      exportedAt: new Date().toISOString(),
      stores: {},
    };

    const storeNames = Object.values(STORES);
    for (const name of storeNames) {
      backup.stores[name] = await db.getAll(name);
    }

    return JSON.stringify(backup, null, 2);
  }

  /**
   * Import database records from JSON backup string.
   */
  static async importFromJson(jsonString: string): Promise<{ success: boolean; importedCounts: Record<string, number> }> {
    const data: MemoryBackup = JSON.parse(jsonString);
    if (!data.stores) {
      throw new Error('Invalid backup file: missing stores object');
    }

    const db = await getDatabase();
    const importedCounts: Record<string, number> = {};

    for (const [storeName, records] of Object.entries(data.stores)) {
      if (!Array.isArray(records)) continue;
      const tx = db.transaction(storeName as any, 'readwrite');
      const store = tx.objectStore(storeName as any);
      for (const record of records) {
        await store.put(record);
      }
      await tx.done;
      importedCounts[storeName] = records.length;
    }

    logMessage('info', 'ExportImportService', 'Import completed successfully', importedCounts);
    return { success: true, importedCounts };
  }

  /**
   * Trigger browser file download for backup.
   */
  static downloadJson(jsonString: string, filename = 'workspace-memory-backup.json'): void {
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}
