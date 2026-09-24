/**
 * Native spike: proves the package's JSI object is installed, that a host
 * function captured in a worklet closure is callable from
 * react-native-executorch's background worklet runtime, and that OpenCV is
 * linked (`rotate` runs `cv::rotate`).
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

/** An RGBA gradient with a solid red square in the top-left corner. */
function makePattern(): Uint8Array {
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
  return data;
}

type Frame = { width: number; height: number; data: Uint8Array };

/**
 * Builds a tensor from the pattern, rotates it natively and reads the result
 * back. Runs wherever it is called: inline on the JS thread, or on
 * react-native-executorch's worklet runtime through `wrapAsync`.
 */
function rotatePattern(src: Uint8Array, degrees: RotationDegrees): Frame {
  'worklet';
  const input = tensor('uint8', [HEIGHT, WIDTH, CHANNELS], src);
  try {
    const output = rotateTensor(input, degrees);
    try {
      const height = output.shape[0]!;
      const width = output.shape[1]!;
      return {
        width,
        height,
        data: output.getData(new Uint8Array(height * width * CHANNELS)),
      };
    } finally {
      output.dispose();
    }
  } finally {
    input.dispose();
  }
}

const rotatePatternOnWorklet = wrapAsync(rotatePattern);

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

/**
 * Checks where the red corner square landed. For a clockwise rotation the
 * top-left square moves to the top-right (90), bottom-right (180) or
 * bottom-left (270) corner.
 */
function checkCorner(frame: Frame, degrees: RotationDegrees): boolean {
  const { width, height, data } = frame;
  const at = (x: number, y: number) => (y * width + x) * CHANNELS;
  const isRed = (i: number) => data[i] === 220 && data[i + 1] === 30 && data[i + 2] === 30;
  const expected =
    degrees === 90
      ? at(width - 1, 0)
      : degrees === 180
        ? at(width - 1, height - 1)
        : at(0, height - 1);
  return isRed(expected) && !isRed(at(0, 0));
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

export default function NativeSpikeScreen() {
  const source = useMemo<Frame>(() => ({ width: WIDTH, height: HEIGHT, data: makePattern() }), []);
  const [result, setResult] = useState<{ label: string; frame: Frame } | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [busy, setBusy] = useState(false);

  const run = async (where: 'worklet' | 'js', degrees: RotationDegrees) => {
    setBusy(true);
    setStatus('running...');
    const started = Date.now();
    try {
      const frame =
        where === 'worklet'
          ? await rotatePatternOnWorklet(source.data, degrees)
          : rotatePattern(source.data, degrees);
      const swap = degrees !== 180;
      const shapeOk =
        frame.width === (swap ? HEIGHT : WIDTH) && frame.height === (swap ? WIDTH : HEIGHT);
      const cornerOk = checkCorner(frame, degrees);
      const verdict = shapeOk && cornerOk ? 'PASS' : 'FAIL';
      const message = `${verdict} ${degrees} deg on ${where}: ${frame.width}x${frame.height} shape=${shapeOk} corner=${cornerOk} in ${Date.now() - started} ms`;
      console.log(`[rnocr-spike] ${Platform.OS} ${message}`);
      setResult({ label: `${degrees} deg on ${where}`, frame });
      setStatus(message);
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      const message = `FAILED ${degrees} deg on ${where}: ${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`;
      console.log(`[rnocr-spike] ${Platform.OS} ${message}`);
      setStatus(message);
    } finally {
      setBusy(false);
    }
  };

  // Self-run once on mount so a device log alone proves the call path works.
  useEffect(() => {
    console.log(`[rnocr-spike] ${Platform.OS} isEmulator=${isEmulator()}`);
    (async () => {
      await run('worklet', 90);
      await run('worklet', 180);
      await run('worklet', 270);
      await run('js', 90);
      console.log(`[rnocr-spike] ${Platform.OS} done`);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <Text style={commonStyles.title}>Native spike</Text>
      <Text style={commonStyles.description}>
        platform: {Platform.OS} · isEmulator: {String(isEmulator())}
      </Text>

      <FramePreview label="source" frame={source} />

      <View style={commonStyles.buttonRow}>
        <Button title="90° worklet" onPress={() => run('worklet', 90)} disabled={busy} />
        <Button title="180° worklet" onPress={() => run('worklet', 180)} disabled={busy} />
        <Button title="270° worklet" onPress={() => run('worklet', 270)} disabled={busy} />
      </View>
      <View style={commonStyles.buttonRow}>
        <Button
          title="90° JS thread"
          variant="secondary"
          onPress={() => run('js', 90)}
          disabled={busy}
        />
      </View>

      <Text style={styles.status}>{status}</Text>
      {result && <FramePreview label={result.label} frame={result.frame} />}
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
