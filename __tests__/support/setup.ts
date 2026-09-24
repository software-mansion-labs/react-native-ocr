/**
 * Jest setup: installs the fake native object and checks for leaked tensors.
 *
 * Runs before each test file is evaluated, which matters because
 * `src/native/bridge.ts` reaches for the TurboModule when `__rnocr_jsi__` is
 * missing, and the TurboModule does not exist under Jest.
 */
import { installFakeRnOcr } from './fakeRnOcr';
import { tensorTracker } from './fakeTensor';

installFakeRnOcr();

beforeEach(() => {
  tensorTracker.reset();
});

// Native memory is not garbage collected, so anything a test allocates through
// the fake and does not dispose is a leak in the code under test.
afterEach(() => {
  const leaked = tensorTracker.liveDescriptions();
  if (leaked.length === 0) return;
  throw new Error(`Tensors were left undisposed by this test: ${leaked.join(', ')}`);
});
