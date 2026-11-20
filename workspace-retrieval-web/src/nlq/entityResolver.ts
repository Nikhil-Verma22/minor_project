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

export class EntityResolver {
  private categoryFuse: Fuse<ObjectCategoryDefinition>;

  constructor() {
    this.categoryFuse = new Fuse(WORKSPACE_CATEGORIES, {
      keys: ['id', 'name', 'aliases'],
      threshold: 0.35,
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

    // Match category via Fuse.js or aliases
    let matchedCategory: string | undefined;

    // Direct check in category list
    for (const cat of WORKSPACE_CATEGORIES) {
      const allNames = [cat.id, cat.name.toLowerCase(), ...cat.aliases.map((a) => a.toLowerCase())];
      for (const alias of allNames) {
        if (cleanText.includes(alias) || alias.includes(cleanText)) {
          matchedCategory = cat.id;
          break;
        }
      }
      if (matchedCategory) break;
    }

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
