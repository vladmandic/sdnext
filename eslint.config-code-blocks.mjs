import markdown from '@eslint/markdown';
import { defineConfig, globalIgnores } from 'eslint/config';

import base from './eslint.config.mjs';

// Fenced code blocks in markdown files; eslint.config.mjs lints the markdown itself.
// A processor replaces the document with its code blocks, so the two need separate runs.
export default defineConfig([
  ...base,
  ...markdown.configs.processor,
  globalIgnores(['modules/control/proc/**']), // vendored
]);
