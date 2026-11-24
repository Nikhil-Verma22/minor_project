/**
 * Entity Resolver.
 * Resolves natural language object mentions to registered categories and persistent instances
 * using exact alias matching and Fuse.js fuzzy fallback.
 */

import Fuse from 'fuse.js';
import { WORKSPACE_CATEGORIES, ObjectCategoryDefinition } from '../config/categories';
import { PersistentObjectInstance } from '../memory/memoryTypes';
import { ATTRIBUTE_COLORS } from './aliases';

export interface ResolvedEntity {
  category?: string;
  instanceId?: string;
  mention: string;
  attributes: Record<string, string | null>;
  isAmbiguous: boolean;
  candidateInstances?: PersistentObjectInstance[];
}

// Common physical scene object mappings for OpenPSG / COCO classes
const COMMON_SCENE_SYNONYMS: Record<string, string> = {
  chair: 'chair',
  seat: 'chair',
  armchair: 'chair',
  couch: 'chair',
  sofa: 'chair',
  stool: 'chair',
  bench: 'chair',
  plant: 'potted plant',
  'potted plant': 'potted plant',
  pot: 'potted plant',
  planter: 'potted plant',
  'flower pot': 'potted plant',
  'plant pot': 'potted plant',
  'pot of plant': 'potted plant',
  person: 'person',
  human: 'person',
  man: 'person',
  woman: 'person',
  people: 'person',
  table: 'dining table',
  'dining table': 'dining table',
  desk: 'workspace_surface',
  surface: 'workspace_surface',
  'workspace surface': 'workspace_surface',
  wall: 'wall',
  floor: 'floor',
  ceiling: 'ceiling',
  shelf: 'shelf',
  window: 'window',
  cabinet: 'cabinet',
};

export class EntityResolver {
  private categoryFuse: Fuse<ObjectCategoryDefinition>;

  constructor() {
    this.categoryFuse = new Fuse(WORKSPACE_CATEGORIES, {
      keys: ['id', 'name', 'aliases'],
      threshold: 0.2, // Strict threshold to prevent false matches (e.g. chair -> charger)
    });
  }

  /**
   * Resolve an object mention from user text against workspace categories and instances.
   */
  resolveMention(
    mentionText: string,
    activeInstances: PersistentObjectInstance[]
  ): ResolvedEntity {
    const cleanText = mentionText.trim().toLowerCase();
    const attributes: Record<string, string> = {};

    // Extract color attribute if present
    for (const color of ATTRIBUTE_COLORS) {
      if (new RegExp(`\\b${color}\\b`, 'i').test(cleanText)) {
        attributes['color'] = color;
      }
    }

    // Check workspace surface explicitly
    if (/\b(workspace surface|desk|table|surface|bench|workstation)\b/i.test(cleanText)) {
      return {
        category: 'workspace_surface',
        instanceId: 'workspace_surface_001',
        mention: mentionText,
        attributes,
        isAmbiguous: false,
      };
    }

    // Check exact instanceId match (e.g. "laptop_001")
    const exactInst = activeInstances.find((inst) => inst.instanceId.toLowerCase() === cleanText);
    if (exactInst) {
      return {
        category: exactInst.categoryId,
        instanceId: exactInst.instanceId,
        mention: mentionText,
        attributes: { ...exactInst.attributes, ...attributes },
        isAmbiguous: false,
      };
    }

    // Match category via Common Scene Objects, active instances, aliases, or Fuse.js
    let matchedCategory: string | undefined;

    // 1. Direct check in Common Scene Synonyms (chair, pot, plant, person, etc.)
    if (COMMON_SCENE_SYNONYMS[cleanText]) {
      matchedCategory = COMMON_SCENE_SYNONYMS[cleanText];
    }

    // 2. Check active instances directly by category, displayName, or ID prefix
    if (!matchedCategory && activeInstances.length > 0) {
      const instMatch = activeInstances.find(
        (inst) =>
          inst.categoryId.toLowerCase() === cleanText ||
          inst.displayName.toLowerCase() === cleanText ||
          inst.instanceId.toLowerCase().startsWith(cleanText + '_')
      );
      if (instMatch) {
        matchedCategory = instMatch.categoryId;
      }
    }

    // 3. Direct check in category list
    if (!matchedCategory) {
      for (const cat of WORKSPACE_CATEGORIES) {
        const allNames = [cat.id, cat.name.toLowerCase(), ...cat.aliases.map((a) => a.toLowerCase())];
        for (const alias of allNames) {
          if (cleanText === alias || (alias.length > 3 && cleanText.includes(alias))) {
            matchedCategory = cat.id;
            break;
          }
        }
        if (matchedCategory) break;
      }
    }

    // 4. Strict Fuse.js fallback
    if (!matchedCategory) {
      const searchRes = this.categoryFuse.search(cleanText);
      if (searchRes.length > 0) {
        matchedCategory = searchRes[0].item.id;
      }
    }

    if (!matchedCategory) {
      return {
        mention: mentionText,
        attributes,
        isAmbiguous: false,
      };
    }

    // Filter matching active instances in workspace
    const candidates = activeInstances.filter((inst) => inst.categoryId === matchedCategory);

    // Apply attribute filter if specified
    const filteredCandidates = attributes['color']
      ? candidates.filter((c) => c.attributes.color === attributes['color'])
      : candidates;

    if (filteredCandidates.length === 1) {
      return {
        category: matchedCategory,
        instanceId: filteredCandidates[0].instanceId,
        mention: mentionText,
        attributes: { ...filteredCandidates[0].attributes, ...attributes },
        isAmbiguous: false,
      };
    }

    if (filteredCandidates.length > 1) {
      return {
        category: matchedCategory,
        mention: mentionText,
        attributes,
        isAmbiguous: true,
        candidateInstances: filteredCandidates,
      };
    }

    return {
      category: matchedCategory,
      mention: mentionText,
      attributes,
      isAmbiguous: false,
    };
  }
}
