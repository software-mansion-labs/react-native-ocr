import {
  backendOrder,
  family,
  pickVariant,
  resolveCandidateBackends,
  variants,
} from '../../src/registry/variants';

describe('backendOrder', () => {
  it('prefers accelerators on devices and only XNNPACK on the iOS simulator', () => {
    expect(backendOrder('ios', false)).toEqual(['coreml', 'mlx', 'xnnpack']);
    expect(backendOrder('ios', true)).toEqual(['xnnpack']);
    expect(backendOrder('android', false)).toEqual(['vulkan', 'xnnpack']);
    expect(backendOrder('android', true)).toEqual(['vulkan', 'xnnpack']);
  });
});

describe('resolveCandidateBackends', () => {
  const ios = backendOrder('ios', false);

  it('keeps the whole order when the runtime reports nothing', () => {
    expect(resolveCandidateBackends(ios, [])).toEqual(ios);
  });

  it('drops backends the binary was not linked with, keeping the order', () => {
    expect(resolveCandidateBackends(ios, ['XnnpackBackend', 'CoreMLBackend'])).toEqual([
      'coreml',
      'xnnpack',
    ]);
    expect(resolveCandidateBackends(ios, ['XnnpackBackend'])).toEqual(['xnnpack']);
    expect(resolveCandidateBackends(backendOrder('android', false), ['XnnpackBackend'])).toEqual([
      'xnnpack',
    ]);
  });
});

describe('pickVariant', () => {
  const keys = ['XNNPACK', 'COREML', 'VULKAN'];

  it('takes the first candidate backend the group publishes', () => {
    expect(pickVariant(keys, ['coreml', 'mlx', 'xnnpack'])).toBe('COREML');
    expect(pickVariant(keys, ['vulkan', 'xnnpack'])).toBe('VULKAN');
    expect(pickVariant(keys, ['xnnpack'])).toBe('XNNPACK');
    expect(pickVariant(keys, ['mlx', 'xnnpack'])).toBe('XNNPACK');
  });

  it('matches keys by their backend prefix and prefers the first declared', () => {
    expect(pickVariant(['XNNPACK_FP32', 'XNNPACK_INT8'], ['xnnpack'])).toBe('XNNPACK_FP32');
    expect(pickVariant(['COREML_FP16', 'XNNPACK'], ['xnnpack'])).toBe('XNNPACK');
  });

  it('honors a pin only when its backend is a candidate', () => {
    expect(pickVariant(keys, ['coreml', 'xnnpack'], 'XNNPACK')).toBe('XNNPACK');
    expect(pickVariant(keys, ['coreml', 'xnnpack'], 'VULKAN')).toBe('COREML');
    expect(pickVariant(keys, ['coreml', 'xnnpack'], 'MISSING')).toBe('COREML');
  });

  it('falls back to the first key when nothing published is linked', () => {
    expect(pickVariant(['COREML', 'VULKAN'], ['xnnpack'])).toBe('COREML');
  });
});

describe('variants and family', () => {
  it('adds a DEFAULT resolved for this platform (iOS device under Jest)', () => {
    const group = variants({ XNNPACK: { id: 'x' }, COREML: { id: 'c' } });
    expect(group.DEFAULT).toBe(group.COREML);
    expect(Object.keys(group)).toEqual(['XNNPACK', 'COREML', 'DEFAULT']);
  });

  it('applies a platform pin', () => {
    const group = variants({ XNNPACK: { id: 'x' }, COREML: { id: 'c' } }, { ios: 'XNNPACK' });
    expect(group.DEFAULT).toBe(group.XNNPACK);
  });

  it('inherits the first sub-group DEFAULT in a family', () => {
    const small = variants({ XNNPACK: { id: 'sx' }, COREML: { id: 'sc' } });
    const large = variants({ XNNPACK: { id: 'lx' }, COREML: { id: 'lc' } });
    expect(family({ SMALL: small, LARGE: large }).DEFAULT).toBe(small.DEFAULT);
  });

  it('reads the linked backends once at import', () => {
    jest.isolateModules(() => {
      const rne = require('react-native-executorch');
      rne.setFakeRegisteredBackends(['XnnpackBackend']);
      const fresh = require('../../src/registry/variants');
      const group = fresh.variants({ COREML: { id: 'c' }, XNNPACK: { id: 'x' } });
      expect(group.DEFAULT).toBe(group.XNNPACK);
    });
  });
});
