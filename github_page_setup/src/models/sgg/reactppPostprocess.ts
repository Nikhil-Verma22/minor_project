/**
 * REACT++ Postprocessing.
 * Un-letterboxes coordinates, filters by confidence, applies VocabularyAdapter.
 */

import { SGGObjectDetection, SGGRelation } from './SGGEngine';
import { VocabularyAdapter } from './vocabularyAdapter';
import { LetterboxResult } from '../../utils/image';

export interface PostprocessOutput {
  objects: SGGObjectDetection[];
  relationships: SGGRelation[];
  postprocessTimeMs: number;
}

export function postprocessReactppOutput(
  boxesData: Float32Array,
  relsData: Float32Array,
  letterbox: LetterboxResult,
  objClasses: string[],
  relClasses: string[],
  options?: { boxConf?: number; relConf?: number }
): PostprocessOutput {
  const startTime = performance.now();
  const boxConfThresh = options?.boxConf ?? 0.25;
  const relConfThresh = options?.relConf ?? 0.05;

  const { r, padX, padY, origW, origH } = letterbox;

  const numBoxes = Math.floor(boxesData.length / 6);
  const objects: SGGObjectDetection[] = [];
  const validBoxIndicesMap = new Map<number, string>(); // modelIndex -> detectionId

  for (let i = 0; i < numBoxes; i++) {
    const offset = i * 6;
    const rawX1 = boxesData[offset];
    const rawY1 = boxesData[offset + 1];
    const rawX2 = boxesData[offset + 2];
    const rawY2 = boxesData[offset + 3];
    const cls1Based = Math.round(boxesData[offset + 4]);
    const score = boxesData[offset + 5];

    if (score < boxConfThresh) continue;

    // Un-letterbox coordinates to original image coordinates
    const x1 = Math.max(0, Math.min(origW, (rawX1 - padX) / r));
    const y1 = Math.max(0, Math.min(origH, (rawY1 - padY) / r));
    const x2 = Math.max(0, Math.min(origW, (rawX2 - padX) / r));
    const y2 = Math.max(0, Math.min(origH, (rawY2 - padY) / r));

    // OpenPSG class mapping (cls1Based is 1-based, 0 is background in some conventions, metadata array starts at 0)
    const clsIdx = cls1Based - 1;
    const modelLabel = objClasses[clsIdx] || `unknown_cls_${cls1Based}`;

    const mapping = VocabularyAdapter.mapObjectCategory(modelLabel);
    const detectionId = `det_${i}_${clsIdx}`;

    validBoxIndicesMap.set(i, detectionId);

    objects.push({
      detectionId,
      modelLabel,
      projectCategory: mapping.projectCategory,
      bbox: [
        Math.round(x1 * 10) / 10,
        Math.round(y1 * 10) / 10,
        Math.round(x2 * 10) / 10,
        Math.round(y2 * 10) / 10,
      ],
      confidence: Math.round(score * 1000) / 1000,
      attributes: {
        color: null,
        type: null,
      },
    });
  }

  const numRels = Math.floor(relsData.length / 5);
  const relationships: SGGRelation[] = [];

  for (let j = 0; j < numRels; j++) {
    const offset = j * 5;
    const subjIdx = Math.round(relsData[offset]);
    const objIdx = Math.round(relsData[offset + 1]);
    const relCls = Math.round(relsData[offset + 2]);
    const tripletScore = relsData[offset + 3];

    // Only include relation if both subject and object passed detection threshold
    const subjectDetectionId = validBoxIndicesMap.get(subjIdx);
    const objectDetectionId = validBoxIndicesMap.get(objIdx);

    if (!subjectDetectionId || !objectDetectionId) continue;
    if (tripletScore < relConfThresh) continue;

    const predicateModelLabel = relClasses[relCls] || `rel_${relCls}`;
    if (predicateModelLabel === '__background__') continue;

    const mapping = VocabularyAdapter.mapPredicate(predicateModelLabel);

    relationships.push({
      relationId: `rel_${j}_${subjIdx}_${objIdx}`,
      subjectDetectionId,
      objectDetectionId,
      predicateModelLabel,
      canonicalPredicate: mapping.canonicalPredicate ?? undefined,
      confidence: Math.round(tripletScore * 1000) / 1000,
      source: 'reactpp',
      provisional: false,
    });
  }

  const postprocessTimeMs = performance.now() - startTime;
  return {
    objects,
    relationships,
    postprocessTimeMs,
  };
}
