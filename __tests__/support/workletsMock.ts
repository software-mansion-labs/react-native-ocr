/**
 * Mock of `react-native-worklets`: a worklet is an ordinary function marked for
 * a second runtime, so running it inline exercises the same code minus the
 * thread hop.
 */
export type WorkletRuntime = { readonly name: string };

export const createWorkletRuntime = (config?: { name?: string }): WorkletRuntime => ({
  name: config?.name ?? 'FakeWorkletRuntime',
});

export const runOnRuntimeAsync = async <Args, R>(
  _runtime: WorkletRuntime,
  fn: (args: Args) => R,
  args: Args
): Promise<R> => fn(args);

export const scheduleOnRN = <Args extends unknown[]>(
  fn: (...args: Args) => unknown,
  ...args: Args
): void => {
  setTimeout(() => fn(...args), 0);
};
