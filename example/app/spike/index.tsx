/**
 * Native spike: proves the package's JSI object is installed, that a host
 * function captured in a worklet closure is callable from
 * react-native-executorch's background worklet runtime, and that OpenCV is
 * linked (`rotate` runs `cv::rotate`). The self-test also runs every Phase 2
 * op once on the worklet runtime (`opsSelfTest`), plus the Phase 3 CRAFT
 * extractor strategy (geometry on top of react-native-executorch's `cv`
 * helpers), so a device log proves the whole native surface links and answers
 * and that the pure-TypeScript layer runs on the worklet runtime. It also logs
 * which variant the model registry resolved `DEFAULT` to.
 *
 * Every button rotates the *current* image by its angle, so clicks accumulate;
 * "Reset" returns to the source pattern. The red square in the source's
 * top-left corner makes the accumulated rotation visible and checkable.
 */
import { AlphaType, Canvas, ColorType, Image as SkiaImage, Skia } from '@shopify/react-native-skia';
import { useEffect, useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, View } from 'react-native';
import { tensor, wrapAsync } from 'react-native-executorch';
import {
  craftExtractBoxes,
  cropTensor,
  extractCraftTextBoxes,
  isEmulator,
  models,
  padCornerMean,
  rotateTensor,
  warpByGridTensor,
  type RotationDegrees,
} from '@swmansion/react-native-ocr';
import { Button } from '../../components/Button';
import { commonStyles, theme } from '../../theme';

const WIDTH = 96;
const HEIGHT = 64;
const CHANNELS = 4;

type Frame = { width: number; height: number; data: Uint8Array };
type Where = 'worklet' | 'js';
type TotalDegrees = 0 | 90 | 180 | 270;

/** An RGBA gradient with a solid red square in the top-left corner. */
function makeSource(): Frame {
  const data = new Uint8Array(WIDTH * HEIGHT * CHANNELS);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const i = (y * WIDTH + x) * CHANNELS;
      const corner = x < 20 && y < 20;
      data[i] = corner ? 220 : Math.round((255 * x) / (WIDTH - 1));
      data[i + 1] = corner ? 30 : Math.round((255 * y) / (HEIGHT - 1));
      data[i + 2] = corner ? 30 : 160;
      data[i + 3] = 255;
    }
  }
  return { width: WIDTH, height: HEIGHT, data };
}

/**
 * Builds a tensor from the frame, rotates it natively and reads the result
 * back. Runs wherever it is called: inline on the JS thread, or on
 * react-native-executorch's worklet runtime through `wrapAsync`.
 */
function rotateFrame(
  data: Uint8Array,
  height: number,
  width: number,
  degrees: RotationDegrees
): Frame {
  'worklet';
  const input = tensor('uint8', [height, width, CHANNELS], data);
  try {
    const output = rotateTensor(input, degrees);
    try {
      const outHeight = output.shape[0]!;
      const outWidth = output.shape[1]!;
      return {
        width: outWidth,
        height: outHeight,
        data: output.getData(new Uint8Array(outHeight * outWidth * CHANNELS)),
      };
    } finally {
      output.dispose();
    }
  } finally {
    input.dispose();
  }
}

const rotateFrameOnWorklet = wrapAsync(rotateFrame);

function rotateWhere(frame: Frame, degrees: RotationDegrees, where: Where): Promise<Frame> | Frame {
  return where === 'worklet'
    ? rotateFrameOnWorklet(frame.data, frame.height, frame.width, degrees)
    : rotateFrame(frame.data, frame.height, frame.width, degrees);
}

/**
 * Checks the frame against the accumulated clockwise rotation of the source:
 * the axes must match, and the red corner square must sit in the top-left
 * (0), top-right (90), bottom-right (180) or bottom-left (270) corner.
 */
function check(frame: Frame, total: TotalDegrees): { shapeOk: boolean; cornerOk: boolean } {
  const swap = total === 90 || total === 270;
  const shapeOk =
    frame.width === (swap ? HEIGHT : WIDTH) && frame.height === (swap ? WIDTH : HEIGHT);

  const { width, height, data } = frame;
  const at = (x: number, y: number) => (y * width + x) * CHANNELS;
  const isRed = (i: number) => data[i] === 220 && data[i + 1] === 30 && data[i + 2] === 30;
  const corners: Record<TotalDegrees, number> = {
    0: at(0, 0),
    90: at(width - 1, 0),
    180: at(width - 1, height - 1),
    270: at(0, height - 1),
  };
  const others = ([0, 90, 180, 270] as TotalDegrees[]).filter((k) => k !== total);
  const cornerOk = isRed(corners[total]) && others.every((k) => !isRed(corners[k]));
  return { shapeOk, cornerOk };
}

function toSkiaImage(frame: Frame) {
  const data = Skia.Data.fromBytes(frame.data);
  return Skia.Image.MakeImage(
    {
      width: frame.width,
      height: frame.height,
      colorType: ColorType.RGBA_8888,
      alphaType: AlphaType.Unpremul,
    },
    data,
    frame.width * CHANNELS
  );
}

function FramePreview({ label, frame }: { label: string; frame: Frame }) {
  const image = useMemo(() => toSkiaImage(frame), [frame]);
  const scale = 2;
  return (
    <View style={styles.preview}>
      <Text style={styles.previewLabel}>
        {label} ({frame.width}x{frame.height})
      </Text>
      <Canvas style={{ width: frame.width * scale, height: frame.height * scale }}>
        {image && (
          <SkiaImage
            image={image}
            x={0}
            y={0}
            width={frame.width * scale}
            height={frame.height * scale}
            fit="fill"
          />
        )}
      </Canvas>
    </View>
  );
}

/**
 * Rotates the source through both call paths, chains two rotations, and logs
 * a PASS/FAIL line per step so a device log alone proves the call path works.
 * Touches no screen state.
 */
async function selfTest(source: Frame): Promise<string> {
  const steps: { where: Where; degrees: RotationDegrees }[] = [
    { where: 'worklet', degrees: 90 },
    { where: 'worklet', degrees: 180 },
    { where: 'worklet', degrees: 270 },
    { where: 'js', degrees: 90 },
  ];
  let passed = 0;
  for (const { where, degrees } of steps) {
    try {
      const started = Date.now();
      const frame = await rotateWhere(source, degrees, where);
      const { shapeOk, cornerOk } = check(frame, degrees);
      const ok = shapeOk && cornerOk;
      passed += ok ? 1 : 0;
      console.log(
        `[rnocr-spike] ${Platform.OS} ${ok ? 'PASS' : 'FAIL'} ${degrees} deg on ${where}: ${frame.width}x${frame.height} shape=${shapeOk} corner=${cornerOk} in ${Date.now() - started} ms`
      );
    } catch (e) {
      console.log(`[rnocr-spike] ${Platform.OS} FAILED ${degrees} deg on ${where}: ${String(e)}`);
    }
  }
  // Accumulation: 90 on the worklet runtime followed by 90 on the JS thread
  // must equal a single 180.
  try {
    const first = await rotateWhere(source, 90, 'worklet');
    const chained = rotateFrame(first.data, first.height, first.width, 90);
    const { shapeOk, cornerOk } = check(chained, 180);
    const ok = shapeOk && cornerOk;
    passed += ok ? 1 : 0;
    console.log(
      `[rnocr-spike] ${Platform.OS} ${ok ? 'PASS' : 'FAIL'} 90 worklet + 90 js = 180: ${chained.width}x${chained.height} shape=${shapeOk} corner=${cornerOk}`
    );
  } catch (e) {
    console.log(`[rnocr-spike] ${Platform.OS} FAILED chained rotation: ${String(e)}`);
  }
  const opsPassed = await opsSelfTest(source);
  const summary = `self-test: ${passed + opsPassed}/${steps.length + 1 + OPS_CHECKS} passed`;
  console.log(`[rnocr-spike] ${Platform.OS} ${summary}`);
  return summary;
}

const OPS_CHECKS = 5;

/**
 * Runs crop, padCornerMean, warpByGrid, extractCraftTextBoxes and the CRAFT
 * extractor strategy once each on the worklet runtime against inputs whose
 * results are known in closed form, and returns how many of the five matched.
 * Everything native is disposed inside the worklet.
 */
function runOps(data: Uint8Array, height: number, width: number): boolean[] {
  'worklet';
  const results: boolean[] = [];
  const page = tensor('uint8', [height, width, CHANNELS], data);
  try {
    // 1. Crop the red corner square: every pixel of the 20x20 result is red.
    const corner = cropTensor(page, { format: 'xyxy', xmin: 0, ymin: 0, xmax: 20, ymax: 20 });
    try {
      const px = corner.getData(new Uint8Array(20 * 20 * CHANNELS));
      let allRed = corner.shape[0] === 20 && corner.shape[1] === 20;
      for (let i = 0; i < px.length && allRed; i += CHANNELS) {
        allRed = px[i] === 220 && px[i + 1] === 30 && px[i + 2] === 30;
      }
      results.push(allRed);
    } finally {
      corner.dispose();
    }

    // 2. Pad a uniform strip: the new columns take the corner color.
    const strip = new Uint8Array(4 * 6 * CHANNELS);
    for (let i = 0; i < strip.length; i += CHANNELS) {
      strip[i] = 10;
      strip[i + 1] = 20;
      strip[i + 2] = 30;
      strip[i + 3] = 255;
    }
    const padded = padCornerMean(strip, 4, 6, CHANNELS, 9);
    let padOk = padded.length === 4 * 9 * CHANNELS;
    for (let y = 0; y < 4 && padOk; y++) {
      const i = (y * 9 + 8) * CHANNELS;
      padOk = padded[i] === 10 && padded[i + 1] === 20 && padded[i + 2] === 30;
    }
    results.push(padOk);

    // 3. Warp through an identity grid: the page comes back unchanged.
    const grid = tensor('float32', [2, 2, 2], Float32Array.from([-1, 1, -1, 1, -1, -1, 1, 1]));
    try {
      const warped = warpByGridTensor(page, grid, { interpolation: 'nearest' });
      try {
        const out = warped.getData(new Uint8Array(height * width * CHANNELS));
        let same = true;
        for (let i = 0; i < out.length && same; i++) same = out[i] === data[i];
        results.push(same);
      } finally {
        warped.dispose();
      }
    } finally {
      grid.dispose();
    }

    // 4. Decode a synthetic CRAFT heatmap with one 8x8 text blob: one box that
    // covers it, in doubled (detector input) coordinates.
    const heat = new Float32Array(32 * 32 * 2);
    for (let y = 8; y <= 15; y++) for (let x = 8; x <= 15; x++) heat[(y * 32 + x) * 2] = 1;
    const heatmap = tensor('float32', [1, 32, 32, 2], heat);
    try {
      const boxes = extractCraftTextBoxes(heatmap, {
        textThreshold: 0.4,
        linkThreshold: 0.4,
        lowTextThreshold: 0.7,
        minArea: 10,
        charLevel: false,
        scale: 2,
      });
      const box = boxes[0];
      results.push(
        boxes.length === 1 &&
          box !== undefined &&
          box.x0 <= 16 &&
          box.y0 <= 16 &&
          box.x1 >= 30 &&
          box.y1 >= 30 &&
          box.x1 <= 64
      );
    } finally {
      heatmap.dispose();
    }

    // 5. Phase 3: the CRAFT extractor strategy turns a 20x8 blob into one line
    // quad, grouping and de-skewing in TypeScript on top of
    // react-native-executorch's cv helpers, all on the worklet runtime.
    const lineHeat = new Float32Array(32 * 32 * 2);
    for (let y = 8; y <= 15; y++) for (let x = 8; x <= 27; x++) lineHeat[(y * 32 + x) * 2] = 1;
    const lineHeatmap = tensor('float32', [1, 32, 32, 2], lineHeat);
    try {
      const quads = craftExtractBoxes.extract([lineHeatmap], { width: 64, height: 64 }, false);
      const quad = quads[0];
      results.push(
        quads.length === 1 &&
          quad !== undefined &&
          quad[0].x <= 16 &&
          quad[0].y <= 16 &&
          quad[2].x >= 54 &&
          quad[2].y >= 30 &&
          quad[2].x <= 64
      );
    } finally {
      lineHeatmap.dispose();
    }
  } finally {
    page.dispose();
  }
  return results;
}

const runOpsOnWorklet = wrapAsync(runOps);

async function opsSelfTest(source: Frame): Promise<number> {
  const names = [
    'crop',
    'padCornerMean',
    'warpByGrid',
    'extractCraftTextBoxes',
    'craftExtractBoxes (phase 3)',
  ];
  try {
    const started = Date.now();
    const results = await runOpsOnWorklet(source.data, source.height, source.width);
    const elapsed = Date.now() - started;
    results.forEach((ok, i) => {
      console.log(`[rnocr-spike] ${Platform.OS} ${ok ? 'PASS' : 'FAIL'} ${names[i]} on worklet`);
    });
    console.log(`[rnocr-spike] ${Platform.OS} phase 2 + 3 ops ran in ${elapsed} ms`);
    return results.filter(Boolean).length;
  } catch (e) {
    const err = e as { name?: string; code?: string; message?: string };
    console.log(
      `[rnocr-spike] ${Platform.OS} FAILED phase 2 ops: ${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`
    );
    return 0;
  }
}

export default function NativeSpikeScreen() {
  const source = useMemo(makeSource, []);
  const [current, setCurrent] = useState<Frame>(source);
  const [total, setTotal] = useState<TotalDegrees>(0);
  const [status, setStatus] = useState<string>('idle');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const english = models.ocr.EASYOCR.ENGLISH;
    const variant = (['XNNPACK', 'COREML', 'VULKAN'] as const).find(
      (key) => english[key] === english.DEFAULT
    );
    console.log(
      `[rnocr-spike] ${Platform.OS} isEmulator=${isEmulator()} registry DEFAULT=${variant ?? 'unknown'} ${english.DEFAULT.modelPath}`
    );
    selfTest(source).then(setStatus);
  }, [source]);

  const rotateBy = async (where: Where, degrees: RotationDegrees) => {
    setBusy(true);
    const started = Date.now();
    try {
      const frame = await rotateWhere(current, degrees, where);
      const next = ((total + degrees) % 360) as TotalDegrees;
      const { shapeOk, cornerOk } = check(frame, next);
      const verdict = shapeOk && cornerOk ? 'PASS' : 'FAIL';
      const message = `${verdict} +${degrees}° on ${where} → ${next}° total in ${Date.now() - started} ms`;
      console.log(`[rnocr-spike] ${Platform.OS} ${message}`);
      setCurrent(frame);
      setTotal(next);
      setStatus(message);
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      const message = `FAILED +${degrees}° on ${where}: ${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`;
      console.log(`[rnocr-spike] ${Platform.OS} ${message}`);
      setStatus(message);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setCurrent(source);
    setTotal(0);
    setStatus('reset to source');
  };

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <Text style={commonStyles.description}>
        platform: {Platform.OS} · isEmulator: {String(isEmulator())}
      </Text>
      <Text style={commonStyles.description}>
        Each button rotates the current image by its angle. The "worklet" buttons run on
        react-native-executorch's background runtime; the "JS thread" button runs inline.
      </Text>

      <FramePreview label={`current · ${total}° total`} frame={current} />

      <View style={commonStyles.buttonRow}>
        <Button title="+90° worklet" onPress={() => rotateBy('worklet', 90)} disabled={busy} />
        <Button title="+180° worklet" onPress={() => rotateBy('worklet', 180)} disabled={busy} />
        <Button title="+270° worklet" onPress={() => rotateBy('worklet', 270)} disabled={busy} />
      </View>
      <View style={commonStyles.buttonRow}>
        <Button
          title="+90° JS thread"
          variant="secondary"
          onPress={() => rotateBy('js', 90)}
          disabled={busy}
        />
        <Button title="Reset" variant="accent" onPress={reset} disabled={busy} />
      </View>

      <Text style={styles.status}>{status}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  preview: {
    alignItems: 'center',
    marginBottom: theme.spacing.large,
  },
  previewLabel: {
    fontSize: 13,
    color: theme.colors.textMuted,
    marginBottom: theme.spacing.small,
  },
  status: {
    fontSize: 14,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.large,
    textAlign: 'center',
  },
});
