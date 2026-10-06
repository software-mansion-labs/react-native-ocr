/**
 * Writes a real CRAFT detector output to the app's documents directory, as the
 * fixture `cpp/tests/CraftParityTest.cpp` reads: the page letterboxed to a
 * legal detector size, run through `detect`, and the `[H, W, 2]` float32
 * heatmap stored behind a small header. `scripts/pull-craft-fixtures.sh`
 * copies the files from the simulator or emulator into `cpp/tests/fixtures`.
 *
 * The detector is run here directly, not through the package's engine, so the
 * heatmap is captured before any box extraction touches it.
 */
import ReactNativeBlobUtil from 'react-native-blob-util';
import { Platform } from 'react-native';
import { IMAGENET_NORM, loadModel, tensor, type Model } from 'react-native-executorch';
import {
  cvtColor,
  normalize,
  resize,
  toChannelsFirst,
  type ImageBuffer,
} from 'react-native-executorch/cv';
import type { ConcreteDim, TensorSpec } from 'react-native-executorch/schema';

/** The header `CraftParityTest.cpp` expects: magic, version, height, width. */
const MAGIC = 'CRFT';
const VERSION = 1;

/** Letterbox target: keeps the fixtures a few hundred KB each. */
const TARGET_WIDTH = 400;

/**
 * A legal even detector size at or above `desired` within `dim`'s domain
 * (CRAFT's heatmap is half-resolution, so the input must be even).
 */
function legalEvenDim(dim: ConcreteDim, desired: number): number {
  const even = (value: number) => value + (value % 2);
  switch (dim.kind) {
    case 'constant':
      return dim.value;
    case 'enum': {
      const evens = dim.choices.filter((c) => c % 2 === 0).sort((a, b) => a - b);
      return evens.find((c) => c >= desired) ?? evens[evens.length - 1]!;
    }
    case 'range': {
      // Snap up onto min + k * step (the step is even for every CRAFT export).
      const { min, max, step } = dim.range;
      const steps = Math.ceil((Math.max(desired, min) - min) / step);
      return even(Math.min(min + steps * step, max));
    }
  }
}

function detectorInputDims(model: Model): { h: ConcreteDim; w: ConcreteDim } {
  const input = model.schema.detect?.inputs[0] as TensorSpec<ConcreteDim> | undefined;
  if (!input || input.kind !== 'Tensor' || input.shape.length !== 4) {
    throw new Error('the model has no detect method with a [1, 3, H, W] input');
  }
  return { h: input.shape[2]!, w: input.shape[3]! };
}

/** Runs `detect` on the letterboxed page and returns the raw heatmap. */
function runDetect(model: Model, page: ImageBuffer, detW: number, detH: number): Float32Array {
  const heatH = detH / 2;
  const heatW = detW / 2;
  const tPage = tensor('uint8', [page.height, page.width, 4]);
  const tResize = tensor('uint8', [detH, detW, 4]);
  const tRgb = tensor('uint8', [detH, detW, 3]);
  const tCF = tensor('uint8', [3, detH, detW]);
  const tNorm = tensor('float32', [3, detH, detW]);
  const tInput = tensor('float32', [1, 3, detH, detW]);
  const tHeat = tensor('float32', [1, heatH, heatW, 2]);
  try {
    tPage.setData(page.data);
    tPage
      .through(resize, tResize, { mode: 'letterbox', interpolation: 'area', padValue: 0 })
      .through(cvtColor, tRgb, 'RGBA2RGB')
      .through(toChannelsFirst, tCF)
      .through(normalize, tNorm, IMAGENET_NORM)
      .copyTo(tInput);
    model.execute('detect', [tInput], [tHeat]);
    return tHeat.getData(new Float32Array(heatH * heatW * 2));
  } finally {
    for (const t of [tPage, tResize, tRgb, tCF, tNorm, tInput, tHeat]) t.dispose();
  }
}

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  }
  return btoa(binary);
}

/** Packs the header and the little-endian float32 heatmap into one buffer. */
function encodeFixture(heat: Float32Array, heatH: number, heatW: number): Uint8Array {
  const out = new Uint8Array(16 + heat.byteLength);
  const view = new DataView(out.buffer);
  for (let i = 0; i < 4; i++) out[i] = MAGIC.charCodeAt(i);
  view.setUint32(4, VERSION, true);
  view.setInt32(8, heatH, true);
  view.setInt32(12, heatW, true);
  const data = new Uint8Array(heat.buffer, heat.byteOffset, heat.byteLength);
  out.set(data, 16);
  return out;
}

export type CraftFixtureInfo = {
  /** Where the `.heat` file was written. */
  readonly path: string;
  readonly detectorSize: { width: number; height: number };
  readonly heatmapSize: { width: number; height: number };
};

/**
 * Runs the model's detector on `page` and writes `<name>.heat` plus a
 * `<name>.json` sidecar describing it into the documents directory.
 * @param modelPath The local path of the fused EasyOCR model (already downloaded).
 * @param page The page to detect on (RGBA).
 * @param name The fixture name (`sample`, `vertical`).
 * @param provenance Free-form facts to record in the sidecar.
 */
export async function saveCraftFixture(
  modelPath: string,
  page: ImageBuffer,
  name: string,
  provenance: Record<string, string>
): Promise<CraftFixtureInfo> {
  const model = await loadModel(modelPath);
  try {
    const dims = detectorInputDims(model);
    const scale = Math.min(1, TARGET_WIDTH / page.width);
    const detW = legalEvenDim(dims.w, Math.round(page.width * scale));
    const detH = legalEvenDim(dims.h, Math.round(page.height * scale));
    const heat = runDetect(model, page, detW, detH);
    const heatH = detH / 2;
    const heatW = detW / 2;

    const dir = ReactNativeBlobUtil.fs.dirs.DocumentDir;
    const path = `${dir}/${name}.heat`;
    await ReactNativeBlobUtil.fs.writeFile(
      path,
      toBase64(encodeFixture(heat, heatH, heatW)),
      'base64'
    );
    const sidecar = {
      ...provenance,
      platform: Platform.OS,
      imageSize: { width: page.width, height: page.height },
      detectorSize: { width: detW, height: detH },
      heatmapSize: { width: heatW, height: heatH },
      letterbox: { interpolation: 'area', padValue: 0, norm: 'IMAGENET_NORM' },
      createdAt: new Date().toISOString(),
    };
    await ReactNativeBlobUtil.fs.writeFile(
      `${dir}/${name}.json`,
      JSON.stringify(sidecar, null, 2) + '\n',
      'utf8'
    );
    return {
      path,
      detectorSize: { width: detW, height: detH },
      heatmapSize: { width: heatW, height: heatH },
    };
  } finally {
    model.dispose();
  }
}

/**
 * Writes `last-run.json` to the documents directory: whatever the screen
 * passes (detections, grade, timings), so a run can be inspected on the
 * development machine, where iOS gives no console output.
 */
export async function saveLastRun(run: unknown): Promise<void> {
  await ReactNativeBlobUtil.fs.writeFile(
    `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/last-run.json`,
    JSON.stringify(run, null, 2) + '\n',
    'utf8'
  );
}
