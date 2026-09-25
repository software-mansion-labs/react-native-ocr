import { tensor } from 'react-native-executorch';
import { greedyCtcDecode } from '../../src/ocr/ctc';

const codedError = (code: string) => expect.objectContaining({ name: 'RnExecuTorchError', code });

/** Builds `[1, T, V]` probs from one row per timestep. */
function probsOf(rows: number[][]) {
  const vocab = rows[0]!.length;
  return tensor('float32', [1, rows.length, vocab], Float32Array.from(rows.flat()));
}

const CHARSET = ['[blank]', 'a', 'b', 'c'];

describe('greedyCtcDecode', () => {
  it('collapses repeats, drops blanks and averages the kept probabilities', () => {
    const probs = probsOf([
      [0.1, 0.0, 0.9, 0.0], // b
      [0.2, 0.0, 0.8, 0.0], // b again: repeat, no new glyph
      [1.0, 0.0, 0.0, 0.0], // blank
      [0.3, 0.0, 0.7, 0.0], // b after a blank: new glyph
      [0.4, 0.0, 0.0, 0.6], // c
    ]);
    const result = greedyCtcDecode(probs, CHARSET);
    probs.dispose();
    expect(result.text).toBe('bbc');
    expect(result.confidence).toBeCloseTo((0.9 + 0.8 + 0.7 + 0.6) / 4);
  });

  it('returns empty text with zero confidence when every timestep is blank', () => {
    const probs = probsOf([
      [1, 0, 0, 0],
      [1, 0, 0, 0],
    ]);
    expect(greedyCtcDecode(probs, CHARSET)).toEqual({ text: '', confidence: 0 });
    probs.dispose();
  });

  it('counts classes beyond the charset toward the confidence but emits nothing for them', () => {
    const probs = probsOf([
      [0, 0.9, 0, 0, 0], // a
      [0, 0, 0, 0, 0.5], // class 4, past the 4-entry charset
    ]);
    const result = greedyCtcDecode(probs, CHARSET);
    probs.dispose();
    expect(result.text).toBe('a');
    expect(result.confidence).toBeCloseTo(0.7);
  });

  it('accepts a plain [T, V] tensor', () => {
    const probs = tensor('float32', [2, 4], Float32Array.from([0, 0, 0, 1, 0, 1, 0, 0]));
    expect(greedyCtcDecode(probs, CHARSET).text).toBe('ca');
    probs.dispose();
  });

  it('rejects tensors that are not float32 of rank 2 or 3', () => {
    const wrongType = tensor('uint8', [1, 2, 4]);
    expect(() => greedyCtcDecode(wrongType, CHARSET)).toThrow(codedError('INVALID_ARGUMENT'));
    wrongType.dispose();
    const wrongRank = tensor('float32', [4]);
    expect(() => greedyCtcDecode(wrongRank, CHARSET)).toThrow(codedError('INVALID_ARGUMENT'));
    wrongRank.dispose();
  });
});
