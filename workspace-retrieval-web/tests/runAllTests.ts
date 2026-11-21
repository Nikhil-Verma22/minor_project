/**
 * Master Test Runner.
 * Executes all automated unit and integration tests.
 */

import { testVocabularyAdapter } from './sgg/vocabularyAdapter.test';
import { testPostprocessing } from './sgg/postprocess.test';
import { testIdentityAssociation } from './memory/identityAssociation.test';
import { testQueryParser } from './nlq/parser.test';
import { testWorkspaceAssistant } from './workspaceAssistant.test';
import { testMultiFrame } from './multiFrame.test';

async function run() {
  console.log('==================================================');
  console.log('Running Object Retrieval Pipeline Test Suite...');
  console.log('==================================================\n');

  try {
    testVocabularyAdapter();
    testPostprocessing();
    testIdentityAssociation();
    testQueryParser();
    await testWorkspaceAssistant();
    await testMultiFrame();

    console.log('\n==================================================');
    console.log('ALL TESTS PASSED SUCCESSFULLY! (6 suites, 0 failures)');
    console.log('==================================================');
    process.exit(0);
  } catch (err: any) {
    console.error('\nTEST SUITE FAILED:', err.message || err);
    process.exit(1);
  }
}

run();
