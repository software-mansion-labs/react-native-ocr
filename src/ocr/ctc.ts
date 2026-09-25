/**
 * Greedy CTC decoding of a recognizer's probability tensor.
 */
import { RnExecuTorchError, tensor, type Tensor } from 'react-native-executorch';
import { argmax, gather } from 'react-native-executorch/math';

/**
 * Greedily decodes softmaxed CTC probabilities: takes each timestep's most
 * probable class natively (`math.argmax` + `math.gather`), drops the blank
 * (index 0) and consecutive repeats, and averages the kept timesteps'
 * probabilities into a confidence.
 *
 * This is the default recognizer decode; a custom `decode` option on an OCR
 * model replaces it and may use it as a building block.
 * @category Functions
 * @param probs Softmaxed recognizer output, `float32` of shape `[1, T, V]` (a
 * plain `[T, V]` is accepted too). Index 0 of `V` is the CTC blank.
 * @param charset The CTC lookup: `charset[i]` labels class `i`, with the blank
 * placeholder at index 0. Classes beyond its length count toward the
 * confidence but emit no text.
 * @returns The decoded text and the mean probability of the non-blank
 * timesteps (0 when there are none).
 * @throws {RnExecuTorchError} With code `INVALID_ARGUMENT` when `probs` is not
 * a `float32` tensor of rank 2 or 3.
 */
export function greedyCtcDecode(
  probs: Tensor,
  charset: readonly string[]
): { readonly text: string; readonly confidence: number } {
  'worklet';
  const shape = probs.shape;
  const rank = shape.length;
  if (probs.dtype !== 'float32' || (rank !== 2 && rank !== 3)) {
    throw RnExecuTorchError(
      'INVALID_ARGUMENT',
      `greedyCtcDecode: expected a float32 [1, T, V] tensor, got ${probs.dtype}[${shape.join(', ')}]`
    );
  }
  const timesteps = shape[rank - 2]!;
  const pickedShape = [...shape.slice(0, rank - 1), 1];
  const tIndices = tensor('int32', pickedShape);
  const tMaxima = tensor('float32', pickedShape);
  const indices = new Int32Array(timesteps);
  const maxima = new Float32Array(timesteps);
  try {
    // Both read from `probs`: argmax writes the indices, gather then reads the
    // value at each of them. Chaining would feed the indices back in as `src`.
    probs.through(argmax, tIndices);
    probs.through(gather, tIndices, tMaxima);
    tIndices.getData(indices);
    tMaxima.getData(maxima);
  } finally {
    tIndices.dispose();
    tMaxima.dispose();
  }

  let text = '';
  let last = -1;
  let probabilitySum = 0;
  let charCounter = 0;
  for (let t = 0; t < timesteps; t++) {
    const idx = indices[t]!;
    if (idx >= 1) {
      probabilitySum += maxima[t]!;
      charCounter++;
      if (idx !== last && idx < charset.length) {
        text += charset[idx]!;
      }
    }
    last = idx;
  }
  return { text, confidence: charCounter === 0 ? 0 : probabilitySum / charCounter };
}
