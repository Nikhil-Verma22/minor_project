/**
 * Tests for REACT++ Postprocessing.
 * Verifies un-letterbox coordinate math, confidence filtering, and triplet unpacking.
 */

import { postprocessReactppOutput } from '../../src/models/sgg/reactppPostprocess';

export function testPostprocessing() {
  console.log('--- Testing Postprocessing ---');

  // Simulated letterbox: original 640x480, resized to 640x480, padX=0, padY=80, r=1.0
  const letterbox = {
    canvas: {} as any,
    r: 1.0,
    padX: 0,
    padY: 80,
    newW: 640,
    newH: 480,
    origW: 640,
    origH: 480,
  };

  const objClasses = ['person', 'bicycle', 'car', 'laptop', 'cell phone'];
  const relClasses = ['__background__', 'on', 'connected to', 'beside'];

  // Box 0: raw [10, 90, 110, 190], cls 4 (laptop, 1-based index 4 -> 3), score 0.85
  // Box 1: raw [200, 200, 300, 300], cls 5 (cell phone, 1-based index 5 -> 4), score 0.15 (below default 0.25 conf)
  // Box 2: raw [300, 100, 400, 200], cls 1 (person), score 0.90
  const boxesData = new Float32Array([
    10, 90, 110, 190, 4, 0.85,
    200, 200, 300, 300, 5, 0.15,
    300, 100, 400, 200, 1, 0.90,
  ]);

  // Relation 0: subj 0, obj 2, rel 3 (beside), triplet score 0.45, rel score 0.50
  // Relation 1: subj 0, obj 1 (obj 1 was filtered out), rel 1 (on), triplet score 0.60
  const relsData = new Float32Array([
    0, 2, 3, 0.45, 0.50,
    0, 1, 1, 0.60, 0.70,
  ]);

  const output = postprocessReactppOutput(
    boxesData,
    relsData,
    letterbox,
    objClasses,
    relClasses,
    { boxConf: 0.25, relConf: 0.05 }
  );

  // Box 1 must be filtered out
  if (output.objects.length !== 2) {
    throw new Error(`Expected 2 surviving objects, got ${output.objects.length}`);
  }

  // Check unpadding on Box 0: y1 was 90, padY=80, r=1.0 -> y1 should be 10.0
  const laptopObj = output.objects[0];
  if (laptopObj.bbox[1] !== 10) {
    throw new Error(`Expected unpadded y1=10, got ${laptopObj.bbox[1]}`);
  }
  if (laptopObj.projectCategory !== 'laptop') {
    throw new Error(`Expected projectCategory='laptop', got ${laptopObj.projectCategory}`);
  }

  // Relation 1 should be omitted because obj 1 was filtered out
  if (output.relationships.length !== 1) {
    throw new Error(`Expected 1 surviving relation, got ${output.relationships.length}`);
  }

  const rel = output.relationships[0];
  if (rel.canonicalPredicate !== 'next_to') {
    throw new Error(`Expected mapped canonicalPredicate='next_to', got ${rel.canonicalPredicate}`);
  }

  console.log('✓ Postprocessing tests passed!');
}
