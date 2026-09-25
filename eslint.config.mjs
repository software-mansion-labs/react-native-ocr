import cspell from '@cspell/eslint-plugin';
import { fixupConfigRules } from '@eslint/compat';
import { FlatCompat } from '@eslint/eslintrc';
import js from '@eslint/js';
import jsdoc from 'eslint-plugin-jsdoc';
import prettier from 'eslint-plugin-prettier';
import { defineConfig } from 'eslint/config';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const typedocConfig = require('./typedoc.json');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

// `@category` tags must name one of the categories typedoc renders, so the
// generated API reference never grows an unsorted bucket.
const VALID_CATEGORIES = typedocConfig.categoryOrder.filter((category) => category !== '*');
const CATEGORY_TAG_MATCH = `^(${VALID_CATEGORIES.join('|')})$`;

export default defineConfig([
  {
    extends: [
      ...fixupConfigRules(compat.extends('@react-native', 'prettier')),
      jsdoc.configs['flat/recommended-typescript'],
    ],
    plugins: { prettier, '@cspell': cspell },
    rules: {
      'react/react-in-jsx-scope': 'off',
      'prettier/prettier': 'error',
      '@cspell/spellchecker': [
        'warn',
        {
          customWordListFile: path.resolve(__dirname, '.cspell-wordlist.txt'),
          autoFix: false,
        },
      ],
      'camelcase': 'error',
      'jsdoc/require-jsdoc': 'off',
      'jsdoc/require-param': ['error', { checkDestructured: false }],
      'jsdoc/check-param-names': ['error', { checkDestructured: false }],
      'jsdoc/require-yields-type': 'off',
      'jsdoc/require-yields-description': 'warn',
      'jsdoc/check-tag-names': ['error', { definedTags: ['property'] }],
      'jsdoc/match-description': [
        'error',
        {
          contexts: ['any'],
          mainDescription: false,
          tags: {
            category: {
              message: '@category must be one of the categories listed in typedoc.json',
              match: CATEGORY_TAG_MATCH,
            },
          },
        },
      ],
    },
    settings: {
      jsdoc: {
        tagNamePreference: {
          typeParam: 'typeParam',
        },
      },
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    rules: {
      'no-console': 'warn',
    },
  },
  {
    // Alphabets and label maps are data, not prose.
    files: ['src/charsets.ts', 'src/constants.ts'],
    rules: {
      '@cspell/spellchecker': 'off',
    },
  },
  {
    // The JSDoc rules keep the generated API reference complete. Test helpers
    // and the example app are not part of that surface.
    files: ['__tests__/**/*.{ts,tsx}', 'example/**/*.{ts,tsx}', '*.{js,mjs}'],
    rules: {
      'jsdoc/require-param': 'off',
      'jsdoc/require-param-description': 'off',
      'jsdoc/require-returns': 'off',
      'jsdoc/require-returns-description': 'off',
    },
  },
  {
    ignores: [
      // The flat config uses import.meta, which the React Native parser preset rejects.
      'eslint.config.mjs',
      'node_modules/',
      'lib/',
      'docs/',
      'coverage/',
      // CMake writes *.ts dependency files into the host test build tree.
      'cpp/tests/build/',
      '.yarn/',
      'example/ios/',
      'example/android/',
      'example/.expo/',
    ],
  },
]);
