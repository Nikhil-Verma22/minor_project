/**
 * Unified Object Palette & Color Mapping Utility.
 * Ensures consistent colors across BoundingBoxOverlay, ObjectList, and SceneGraphPanel.
 */

export const OBJECT_PALETTE = [
  '#38bdf8', // Sky Blue
  '#34d399', // Emerald Green
  '#f472b6', // Coral Pink
  '#fbbf24', // Amber Gold
  '#a78bfa', // Purple / Violet
  '#f87171', // Ruby Red
  '#2dd4bf', // Teal
  '#fb923c', // Tangerine Orange
  '#818cf8', // Indigo
  '#a3e635', // Lime Green
  '#e879f9', // Fuchsia
  '#06b6d4', // Cyan
  '#facc15', // Yellow
  '#4ade80', // Mint
];

export function getPaletteColor(index: number): string {
  return OBJECT_PALETTE[Math.abs(index) % OBJECT_PALETTE.length];
}

export interface EntityColorMapping {
  colorMap: Map<string, string>;
  resolveColor: (id: string | null | undefined, fallbackIndex?: number) => string;
  isEntityMatch: (id1: string | null | undefined, id2: string | null | undefined) => boolean;
}

/**
 * Builds a bidirectional ID map and color map for all detected objects and scene graph nodes.
 */
export function buildEntityColorMap(
  objects: { detectionId: string; instanceId?: string }[],
  nodes?: { id: string; detectionId?: string; isSurface?: boolean }[]
): EntityColorMapping {
  const colorMap = new Map<string, string>();
  const idAliases = new Map<string, Set<string>>();

  const registerAlias = (a: string, b: string) => {
    if (!a || !b) return;
    if (!idAliases.has(a)) idAliases.set(a, new Set([a]));
    if (!idAliases.has(b)) idAliases.set(b, new Set([b]));
    idAliases.get(a)!.add(b);
    idAliases.get(b)!.add(a);
  };

  // 1. Assign colors to objects by detection order
  objects.forEach((obj, idx) => {
    const color = getPaletteColor(idx);
    colorMap.set(obj.detectionId, color);
    if (obj.instanceId) {
      colorMap.set(obj.instanceId, color);
      registerAlias(obj.detectionId, obj.instanceId);
    }
  });

  // 2. Link scene graph nodes to their corresponding object colors
  if (nodes) {
    nodes.forEach((node, idx) => {
      if (node.isSurface || node.id.includes('surface')) {
        colorMap.set(node.id, '#10b981'); // Emerald for workspace surface
        return;
      }
      if (node.detectionId && colorMap.has(node.detectionId)) {
        const c = colorMap.get(node.detectionId)!;
        colorMap.set(node.id, c);
        registerAlias(node.id, node.detectionId);
      } else if (!colorMap.has(node.id)) {
        const c = getPaletteColor(objects.length + idx);
        colorMap.set(node.id, c);
      }
    });
  }

  const resolveColor = (id: string | null | undefined, fallbackIndex: number = 0): string => {
    if (id && colorMap.has(id)) {
      return colorMap.get(id)!;
    }
    return getPaletteColor(fallbackIndex);
  };

  const isEntityMatch = (id1: string | null | undefined, id2: string | null | undefined): boolean => {
    if (!id1 || !id2) return false;
    if (id1 === id2) return true;
    const aliases = idAliases.get(id1);
    return aliases ? aliases.has(id2) : false;
  };

  return { colorMap, resolveColor, isEntityMatch };
}
