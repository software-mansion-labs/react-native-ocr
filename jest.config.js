/**
 * Jest configuration for the TypeScript suites under `__tests__/`.
 *
 * The native boundary is faked, not stubbed: `__tests__/support/fakeRnOcr.ts`
 * implements the package's own JSI ops in plain JavaScript, and
 * `__tests__/support/rneMock.ts` stands in for react-native-executorch with a
 * typed-array-backed tensor.
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
    '^react-native-worklets$': '<rootDir>/__tests__/support/workletsMock.ts',
  },

  modulePathIgnorePatterns: ['<rootDir>/example/node_modules', '<rootDir>/lib/'],

  clearMocks: true,
  restoreMocks: true,
};
