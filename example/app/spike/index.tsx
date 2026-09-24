/**
 * Native spike: proves the package's JSI object is installed, that a host
 * function captured in a worklet closure is callable from
 * react-native-executorch's background worklet runtime, and that OpenCV is
 * linked (`rotate` runs `cv::rotate`).
 *
 * Every button rotates the *current* image by its angle, so clicks accumulate;
 * "Reset" returns to the source pattern. The red square in the source's
 * top-left corner makes the accumulated rotation visible and checkable.
 */
import { AlphaType, Canvas, ColorType, Image as SkiaImage, Skia } from '@shopify/react-native-skia';
import { useEffect, useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, View } from 'react-native';
import { tensor, wrapAsync } from 'react-native-executorch';
import { isEmulator, rotateTensor, type RotationDegrees } from '@swmansion/react-native-ocr';
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
  const summary = `self-test: ${passed}/${steps.length + 1} passed`;
  console.log(`[rnocr-spike] ${Platform.OS} ${summary}`);
  return summary;
}

export default function NativeSpikeScreen() {
  const source = useMemo(makeSource, []);
  const [current, setCurrent] = useState<Frame>(source);
  const [total, setTotal] = useState<TotalDegrees>(0);
  const [status, setStatus] = useState<string>('idle');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    console.log(`[rnocr-spike] ${Platform.OS} isEmulator=${isEmulator()}`);
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
      <Text style={commonStyles.title}>Native spike</Text>
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
