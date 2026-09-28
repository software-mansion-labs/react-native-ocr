/**
 * Jest configuration for the TypeScript suites under `__tests__/`.
 *
 * The native boundary is faked, not stubbed: `__tests__/support/fakeRnOcr.ts`
 * implements the package's own JSI ops in plain JavaScript, and
 * `__tests__/support/rneMock.ts` (with `cvMock.ts` and `mathMock.ts` behind
 * the `/cv` and `/math` subpaths) stands in for react-native-executorch with
 * a typed-array-backed tensor. react-native-executorch's `schema` module is
 * plain TypeScript, so the real one is used through its `/schema` subpath.
 */
module.exports = {
  preset: '@react-native/jest-preset',
  rootDir: '.',
  roots: ['<rootDir>/__tests__'],
  testMatch: ['**/*.test.ts', '**/*.test.tsx'],

  // The fake JSI global has to exist before any `src/` module is imported:
  // `src/native/bridge.ts` reaches for the TurboModule when it is missing.
  setupFilesAfterEnv: ['<rootDir>/__tests__/support/setup.ts'],

  moduleNameMapper: {
    '^react-native-executorch$': '<rootDir>/__tests__/support/rneMock.ts',
    '^react-native-executorch/cv$': '<rootDir>/__tests__/support/cvMock.ts',
    '^react-native-executorch/math$': '<rootDir>/__tests__/support/mathMock.ts',
    '^react-native-executorch/schema$':
      '<rootDir>/node_modules/react-native-executorch/src/core/schema.ts',
    '^react-native-worklets$': '<rootDir>/__tests__/support/workletsMock.ts',
  },

  // The preset transforms only react-native itself; the schema module above is
  // TypeScript source inside react-native-executorch.
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?|react-native-executorch)/)',
  ],

  modulePathIgnorePatterns: ['<rootDir>/example/node_modules', '<rootDir>/lib/'],

  clearMocks: true,
  restoreMocks: true,
};
