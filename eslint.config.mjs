import path from 'node:path';

import { includeIgnoreFile } from '@eslint/compat';
import css from '@eslint/css';
import js from '@eslint/js';
import json from '@eslint/json';
import markdown from '@eslint/markdown';
import html from '@html-eslint/eslint-plugin';
import { configs, helpers, plugins, rules } from 'eslint-config-airbnb-extended';
import pluginPromise from 'eslint-plugin-promise';
import { defineConfig, globalIgnores } from 'eslint/config';
import globals from 'globals';

const gitignorePath = path.resolve('.', '.gitignore');

const jsConfig = defineConfig([
  // ESLint recommended config
  {
    name: 'js/config',
    files: helpers.extensions.allFiles,
    ...js.configs.recommended,
    languageOptions: {
      ecmaVersion: 'latest',
      parserOptions: {
        ecmaVersion: 'latest',
      },
      globals: { // Set per project
        ...globals.builtin,
        ...globals.browser,
        ...globals.jquery,
        panzoom: 'readonly',
        authFetch: 'readonly',
        log: 'readonly',
        debug: 'readonly',
        error: 'readonly',
        xhrGet: 'readonly',
        xhrPost: 'readonly',
        gradioApp: 'readonly',
        executeCallbacks: 'readonly',
        onAfterUiUpdate: 'readonly',
        onOptionsChanged: 'readonly',
        optionsChangedCallbacks: 'readonly',
        onUiLoaded: 'readonly',
        onUiUpdate: 'readonly',
        onUiTabChange: 'readonly',
        onUiReady: 'readonly',
        uiCurrentTab: 'writable',
        uiElementIsVisible: 'readonly',
        uiElementInSight: 'readonly',
        getUICurrentTabContent: 'readonly',
        waitForFlag: 'readonly',
        logFn: 'readonly',
        generateForever: 'readonly',
        showContributors: 'readonly',
        opts: 'writable',
        monitorOption: 'readonly',
        sortUIElements: 'readonly',
        all_gallery_buttons: 'readonly',
        selected_gallery_button: 'readonly',
        selected_gallery_index: 'readonly',
        switch_to_txt2img: 'readonly',
        switch_to_img2img_tab: 'readonly',
        switch_to_img2img: 'readonly',
        switch_to_sketch: 'readonly',
        switch_to_inpaint: 'readonly',
        witch_to_inpaint_sketch: 'readonly',
        switch_to_extras: 'readonly',
        get_tab_index: 'readonly',
        create_submit_args: 'readonly',
        restartReload: 'readonly',
        markSelectedCards: 'readonly',
        updateInput: 'readonly',
        toggleCompact: 'readonly',
        setFontSize: 'readonly',
        setTheme: 'readonly',
        registerDragDrop: 'readonly',
        getToken: 'readonly',
        getENActiveTab: 'readonly',
        quickApplyStyle: 'readonly',
        quickSaveStyle: 'readonly',
        setupExtraNetworks: 'readonly',
        showNetworks: 'readonly',
        localization: 'readonly',
        randomId: 'readonly',
        requestProgress: 'readonly',
        setRefreshInterval: 'readonly',
        modalPrevImage: 'readonly',
        modalNextImage: 'readonly',
        galleryClickEventHandler: 'readonly',
        getExif: 'readonly',
        jobStatusEl: 'readonly',
        removeSplash: 'readonly',
        initGPU: 'readonly',
        startGPU: 'readonly',
        disableNVML: 'readonly',
        idbGet: 'readonly',
        idbPut: 'readonly',
        idbDel: 'readonly',
        idbAdd: 'readonly',
        idbCount: 'readonly',
        idbFolderCleanup: 'readonly',
        idbClearAll: 'readonly',
        idbIsReady: 'readonly',
        initChangelog: 'readonly',
        sendNotification: 'readonly',
        monitorConnection: 'readonly',
      },
    },
  },
  pluginPromise.configs['flat/recommended'],
  // Stylistic plugin
  plugins.stylistic,
  // Import X plugin
  plugins.importX,
  // Airbnb base recommended config
  ...configs.base.recommended,
  {
    name: 'sdnext/js',
    files: helpers.extensions.allFiles,
    languageOptions: {
      ecmaVersion: 'latest',
      parserOptions: {
        ecmaVersion: 'latest',
      },
    },
    rules: {
      camelcase: 'off',
      'default-case': 'off',
      'max-classes-per-file': 'warn',
      'no-await-in-loop': 'off',
      'no-bitwise': 'off',
      'no-continue': 'off',
      'no-console': 'off',
      'no-loop-func': 'off',
      'no-param-reassign': 'off',
      'no-plusplus': 'off',
      'no-redeclare': 'off',
      'no-restricted-globals': 'off',
      'no-restricted-syntax': 'off',
      'no-unused-vars': 'off',
      'no-use-before-define': 'warn',
      'no-useless-escape': 'warn',
      'prefer-destructuring': 'off',
      'prefer-rest-params': 'off',
      'prefer-template': 'warn',
      'promise/no-nesting': 'off',
      radix: 'off',
      '@stylistic/brace-style': [
        'error',
        '1tbs',
        {
          allowSingleLine: true,
        },
      ],
      '@stylistic/indent': ['error', 2],
      '@stylistic/lines-between-class-members': [
        'error',
        'always',
        {
          exceptAfterSingleLine: true,
        },
      ],
      '@stylistic/max-len': [
        'warn',
        {
          code: 275,
          tabWidth: 2,
        },
      ],
      '@stylistic/max-statements-per-line': 'off',
      '@stylistic/no-mixed-operators': 'off',
      '@stylistic/object-curly-newline': [
        'error',
        {
          multiline: true,
          consistent: true,
        },
      ],
      '@stylistic/quotes': [
        'error',
        'single',
        {
          avoidEscape: true,
        },
      ],
      '@stylistic/semi': [
        'error',
        'always',
        {
          omitLastInOneLineBlock: false,
        },
      ],
      'promise/always-return': 'off',
      'promise/catch-or-return': 'off',
    },
  },
]);

// const typescriptConfig = defineConfig([
//   // TypeScript ESLint plugin
//   plugins.typescriptEslint,
//   // Airbnb base TypeScript config
//   ...configs.base.typescript,
//   {
//     name: 'sdnext/typescript',
//     files: helpers.extensions.tsFiles,
//     rules: {
//       '@typescript-eslint/ban-ts-comment': 'off',
//       '@typescript-eslint/explicit-module-boundary-types': 'off',
//       '@typescript-eslint/no-shadow': 'error',
//       '@typescript-eslint/no-var-requires': 'off',
//     },
//   },
// ]);

const nodeConfig = defineConfig([
  // Node plugin
  plugins.node,
  {
    name: 'sdnext/node',
    files: ['**/cli/*.js'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
    rules: {
      // Import as rule sets to override the `files` setting from default config
      ...rules.node.base.rules,
      ...rules.node.globals.rules,
      ...rules.node.noUnsupportedFeatures.rules,
      ...rules.node.promises.rules,
      'n/no-sync': 'off',
      'n/no-process-exit': 'off',
      'n/hashbang': 'off',
    },
  },
]);

const jsonConfig = defineConfig([
  {
    files: ['**/*.json'],
    ignores: ['package-lock.json'],
    plugins: { json },
    language: 'json/json',
    extends: ['json/recommended'],
    rules: {
      'json/no-empty-keys': 'off',
    },
  },
]);

const markdownConfig = defineConfig([
  {
    files: ['**/*.md'],
    plugins: { markdown },
    language: 'markdown/gfm',
    processor: 'markdown/markdown',
    extends: ['markdown/recommended'],
  },
]);

const cssConfig = defineConfig([
  {
    files: ['**/*.css'],
    language: 'css/css',
    plugins: { css },
    extends: ['css/recommended'],
    // languageOptions: {
    //   tolerant: true,
    // },
    rules: {
      'css/font-family-fallbacks': 'off',
      'css/no-invalid-properties': [
        'error',
        {
          allowUnknownVariables: true,
        },
      ],
      'css/no-important': 'off',
      'css/use-baseline': 'off',
    },
  },
]);

const htmlConfig = defineConfig([
  {
    files: ['**/*.html'],
    plugins: {
      html,
    },
    extends: ['html/recommended'],
    language: 'html/html',
    rules: {
      'html/attrs-newline': 'off',
      'html/element-newline': 'off',
      'html/indent': [
        'warn',
        2,
      ],
      'html/no-duplicate-class': 'error',
      'html/no-extra-spacing-attrs': [
        'error',
        {
          enforceBeforeSelfClose: true,
          disallowMissing: true,
          disallowTabs: true,
          disallowInAssignment: true,
        },
      ],
      'html/require-closing-tags': [
        'error',
        {
          selfClosing: 'always',
        },
      ],
      'html/use-baseline': 'off',
    },
  },
]);

export default defineConfig([
  // Ignore files and folders listed in .gitignore
  includeIgnoreFile(gitignorePath),
  globalIgnores([
    '**/node_modules',
    '**/extensions',
    '**/extensions-builtin',
    '**/repositories',
    '**/venv',
    '**/panZoom.js',
    '**/split.js',
    '**/exifr.js',
    '**/jquery.js',
    '**/sparkline.js',
    '**/iframeResizer.min.js',
  ]),
  ...jsConfig,
  // ...typescriptConfig,
  ...nodeConfig,
  ...jsonConfig,
  ...markdownConfig,
  ...cssConfig,
  ...htmlConfig,
]);
