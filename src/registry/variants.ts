/**
 * `DEFAULT` variant resolution for the model registry, mirroring
 * react-native-executorch's own (which it does not export).
 *
 * `DEFAULT` is resolved once, when this module is first imported, from the
 * platform, the backends the binary was linked with, and the order the
 * variants are declared in. A group whose best export does not follow from
 * that order pins one per platform through the second argument of
 * {@link variants}.
 */
import { Platform } from 'react-native';
import { getRegisteredBackends } from 'react-native-executorch';
import { isEmulator } from '../native/ops';

/** Every backend the registry publishes for, spelled as the variant keys spell it. */
export const ALL_BACKENDS = ['xnnpack', 'coreml', 'mlx', 'vulkan'] as const;

/** The backend prefix a variant key starts with. */
export type BackendTag = (typeof ALL_BACKENDS)[number];

/** The platforms the registry resolves defaults for. */
export type TargetPlatform = 'ios' | 'android';

/**
 * The backends a platform prefers, best first.
 *
 * Accelerators lead and XNNPACK trails: a model is only exported to Core ML,
 * MLX or Vulkan once it has been shown to run better there, and XNNPACK is the
 * one backend every model exports to. Core ML sits above MLX only to make the
 * order deterministic.
 *
 * The iOS simulator links the Core ML backend but cannot run it: no Neural
 * Engine, and MPSGraph refuses the compiled models. MLX only ever ships a
 * device slice, so it has nothing to run there either.
 * @param platform The platform.
 * @param emulator Whether the app runs on a simulator or emulator.
 * @returns The preferred backends, best first.
 */
export function backendOrder(platform: TargetPlatform, emulator: boolean): readonly BackendTag[] {
  if (platform === 'ios') {
    return emulator ? ['xnnpack'] : ['coreml', 'mlx', 'xnnpack'];
  }
  return ['vulkan', 'xnnpack'];
}

/**
 * The backends this build may default to, best first.
 * @param order The platform's preference order.
 * @param registered The backend names the native runtime reports as linked.
 * @returns `order` less every backend the binary was not linked with, or
 * `order` untouched when the runtime reported nothing, so that a missing
 * answer widens the choice rather than emptying it.
 */
export function resolveCandidateBackends(
  order: readonly BackendTag[],
  registered: readonly string[]
): readonly BackendTag[] {
  if (registered.length === 0) return order;
  const names = registered.map((name) => name.toLowerCase());
  return order.filter((tag) => names.some((name) => name.startsWith(tag)));
}

/**
 * Picks the variant key a build should default to.
 * @param keys The group's variant keys, in declaration order.
 * @param candidates The backends the build may default to, best first.
 * @param pin The key pinned for this platform, if any.
 * @returns The chosen key.
 */
export function pickVariant(
  keys: readonly string[],
  candidates: readonly BackendTag[],
  pin?: string
): string {
  const backendOf = (key: string) => key.toLowerCase().split('_')[0] as BackendTag;

  if (pin !== undefined && keys.includes(pin) && candidates.includes(backendOf(pin))) {
    return pin;
  }

  for (const tag of candidates) {
    const match = keys.find((key) => backendOf(key) === tag);
    if (match !== undefined) return match;
  }

  // No preferred backend is both published for this model and linked into the
  // build: an app that opted out of the backends its models need. Hand back
  // the first variant so the registry still names a model and the failure
  // surfaces at load, where the error says which backend is missing.
  return keys[0]!;
}

const PLATFORM: TargetPlatform = Platform.OS === 'ios' ? 'ios' : 'android';

/**
 * The backends the native runtime reports, or nothing when it cannot be asked.
 * @returns The registered backend names.
 */
function readRegisteredBackends(): readonly string[] {
  try {
    return getRegisteredBackends();
  } catch {
    return [];
  }
}

const CANDIDATE_BACKENDS = resolveCandidateBackends(
  backendOrder(PLATFORM, isEmulator()),
  readRegisteredBackends()
);

/**
 * Adds a platform-resolved `DEFAULT` to a group of backend variants.
 *
 * Declare the variants best-first within each backend: with several exports
 * from the same backend, the earliest one wins.
 * @typeParam V The variant map.
 * @param map The group's variants, keyed by backend and precision.
 * @param pinned Variant keys to prefer on a given platform, for groups whose
 * best export does not follow from the declaration order. Ignored when the
 * pinned variant's backend is not linked into the build.
 * @returns The variants, plus the `DEFAULT` alias for this platform.
 */
export function variants<V extends Record<string, object>>(
  map: V,
  pinned?: Partial<Record<TargetPlatform, Extract<keyof V, string>>>
): V & { readonly DEFAULT: V[keyof V] } {
  const key = pickVariant(Object.keys(map), CANDIDATE_BACKENDS, pinned?.[PLATFORM]);
  return { ...map, DEFAULT: map[key] as V[keyof V] };
}

/**
 * Adds a `DEFAULT` to a group of sub-groups (a model family split by scale or
 * input size), mirroring the `DEFAULT` of the first sub-group declared, which
 * resolved itself per platform.
 * @typeParam V The sub-group map.
 * @param map The family's sub-groups, most representative first.
 * @returns The sub-groups, plus the inherited `DEFAULT`.
 */
export function family<V extends Record<string, { readonly DEFAULT: unknown }>>(
  map: V
): V & { readonly DEFAULT: V[keyof V]['DEFAULT'] } {
  const first = Object.keys(map)[0]!;
  return { ...map, DEFAULT: map[first]!.DEFAULT as V[keyof V]['DEFAULT'] };
}
