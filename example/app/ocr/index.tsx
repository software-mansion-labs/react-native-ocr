/**
 * PaddleOCR parity: reads the same page with this package's `useOcr` and with
 * react-native-executorch's `useOpticalCharacterRecognizer`, both on the same
 * PP-OCRv6 export, and compares the two results line by line. The verdict is
 * the first thing on the screen so a screenshot alone records it.
 */
import { Canvas, Image as SkiaImage, Path, Skia, useImage } from '@shopify/react-native-skia';
import { useEffect, useMemo, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import {
  models as rneModels,
  useOpticalCharacterRecognizer,
  type PaddleOcrModel,
} from 'react-native-executorch';
import {
  isEmulator,
  models,
  useOcr,
  type OcrDetection,
  type Quad,
} from '@swmansion/react-native-ocr';
import { Button } from '../../components/Button';
import { ModelStatus } from '../../components/ModelStatus';
import { commonStyles, theme } from '../../theme';
import { skImageToBuffer } from '../../utils';

// On Android `DEFAULT` is the Vulkan export, which crashes inside ExecuTorch's
// Vulkan backend on the emulator's software GPU, so emulators run on the CPU.
const PPOCRV6 = models.ocr.PADDLE.PPOCRV6_SMALL;
const OURS = isEmulator() ? PPOCRV6.XNNPACK : PPOCRV6.DEFAULT;
const BACKEND = (['XNNPACK', 'COREML', 'VULKAN'] as const).find((key) => PPOCRV6[key] === OURS);

// The same export and thresholds on both sides, so any difference comes from
// the pipelines and not from the model.
const THEIRS: PaddleOcrModel = {
  modelPath: OURS.modelPath,
  charsetPath: rneModels.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT.charsetPath,
  modelOpts: { defaultConfidenceThreshold: OURS.modelOpts.minConfidence ?? 0 },
};

type Line = Pick<OcrDetection, 'text' | 'confidence' | 'quad'>;

type Comparison = {
  readonly ours: Line[];
  readonly theirs: Line[];
  readonly sameText: number;
  readonly maxQuadDelta: number;
  readonly maxConfidenceDelta: number;
  readonly differences: string[];
  readonly oursMs: number;
  readonly theirsMs: number;
};

function quadDelta(a: Quad, b: Quad): number {
  let max = 0;
  for (let i = 0; i < 4; i++) {
    max = Math.max(max, Math.hypot(a[i]!.x - b[i]!.x, a[i]!.y - b[i]!.y));
  }
  return max;
}

/** Compares the two results position by position; both are in reading order. */
function compare(
  ours: Line[],
  theirs: Line[]
): Omit<Comparison, 'ours' | 'theirs' | 'oursMs' | 'theirsMs'> {
  const differences: string[] = [];
  let sameText = 0;
  let maxQuadDelta = 0;
  let maxConfidenceDelta = 0;
  const count = Math.max(ours.length, theirs.length);
  for (let i = 0; i < count; i++) {
    const a = ours[i];
    const b = theirs[i];
    if (!a || !b) {
      differences.push(`#${i + 1} only in ${a ? 'ours' : 'RNE'}: "${(a ?? b)!.text}"`);
      continue;
    }
    if (a.text === b.text) {
      sameText++;
    } else {
      differences.push(`#${i + 1} ours "${a.text}" / RNE "${b.text}"`);
    }
    maxQuadDelta = Math.max(maxQuadDelta, quadDelta(a.quad, b.quad));
    maxConfidenceDelta = Math.max(maxConfidenceDelta, Math.abs(a.confidence - b.confidence));
  }
  return { sameText, maxQuadDelta, maxConfidenceDelta, differences };
}

function quadPath(quad: Quad, scale: number) {
  const path = Skia.Path.Make();
  path.moveTo(quad[0].x * scale, quad[0].y * scale);
  for (let i = 1; i < 4; i++) path.lineTo(quad[i]!.x * scale, quad[i]!.y * scale);
  path.close();
  return path;
}

export default function OcrParityScreen() {
  const image = useImage(require('../../assets/ocr/sample.png'));
  const ours = useOcr(OURS);
  const theirs = useOpticalCharacterRecognizer(THEIRS);
  const [comparison, setComparison] = useState<Comparison>();
  const [failure, setFailure] = useState<string>();
  const [busy, setBusy] = useState(false);
  const { width: windowWidth } = useWindowDimensions();

  const { runOcr } = ours;
  const { recognizeCharacters } = theirs;
  const ready = !!image && !!runOcr && !!recognizeCharacters;

  const run = async () => {
    if (!image || !runOcr || !recognizeCharacters) return;
    setBusy(true);
    setFailure(undefined);
    try {
      const page = skImageToBuffer(image);
      let started = Date.now();
      const oursResult = await runOcr(page);
      const oursMs = Date.now() - started;
      started = Date.now();
      const theirsResult = await recognizeCharacters(page);
      const theirsMs = Date.now() - started;
      const result = {
        ours: oursResult.detections,
        theirs: theirsResult,
        oursMs,
        theirsMs,
        ...compare(oursResult.detections, theirsResult),
      };
      console.log(
        `[rnocr-parity] ${Platform.OS} ${result.sameText}/${result.theirs.length} lines identical, ` +
          `ours ${result.ours.length} lines, max quad delta ${result.maxQuadDelta.toFixed(2)} px`
      );
      setComparison(result);
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      setFailure(`${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Run once as soon as both pipelines and the page are ready.
  useEffect(() => {
    if (ready) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const canvasWidth = windowWidth - 2 * theme.spacing.large;
  const scale = image ? canvasWidth / image.width() : 1;
  const paths = useMemo(
    () => ({
      ours: comparison?.ours.map((line) => quadPath(line.quad, scale)) ?? [],
      theirs: comparison?.theirs.map((line) => quadPath(line.quad, scale)) ?? [],
    }),
    [comparison, scale]
  );

  const identical =
    comparison !== undefined &&
    comparison.ours.length === comparison.theirs.length &&
    comparison.sameText === comparison.theirs.length;

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <ModelStatus
        isReady={ours.isReady && theirs.isReady}
        downloadProgress={Math.min(ours.downloadProgress, theirs.downloadProgress)}
        error={(ours.error ?? theirs.error)?.message}
        modelTypeLabel="PP-OCRv6"
      />
      {failure && <Text style={styles.failure}>{failure}</Text>}
      {comparison && (
        <View style={styles.verdict}>
          <Text style={[styles.verdictTitle, identical ? styles.pass : styles.fail]}>
            parity: {comparison.sameText}/{comparison.theirs.length} lines identical
          </Text>
          <Text style={styles.verdictLine}>
            lines: ours {comparison.ours.length} · RNE {comparison.theirs.length}
          </Text>
          <Text style={styles.verdictLine}>
            max quad delta {comparison.maxQuadDelta.toFixed(2)} px · max confidence delta{' '}
            {comparison.maxConfidenceDelta.toFixed(4)}
          </Text>
          <Text style={styles.verdictLine}>
            ours {comparison.oursMs} ms · RNE {comparison.theirsMs} ms · {Platform.OS} · {BACKEND}
          </Text>
          {comparison.differences.map((difference) => (
            <Text key={difference} style={styles.difference}>
              {difference}
            </Text>
          ))}
        </View>
      )}

      {image && (
        <Canvas style={{ width: canvasWidth, height: image.height() * scale }}>
          <SkiaImage
            image={image}
            x={0}
            y={0}
            width={canvasWidth}
            height={image.height() * scale}
            fit="fill"
          />
          {paths.theirs.map((path, i) => (
            <Path key={`t${i}`} path={path} style="stroke" strokeWidth={3} color="#ff3b30" />
          ))}
          {paths.ours.map((path, i) => (
            <Path key={`o${i}`} path={path} style="stroke" strokeWidth={1} color="#39FF14" />
          ))}
        </Canvas>
      )}
      <Text style={styles.legend}>green: this package · red underneath: RNE</Text>

      <Button title="Run again" onPress={run} disabled={!ready || busy} loading={busy} />

      {comparison?.ours.map((line, i) => (
        <Text key={i} style={styles.line}>
          {i + 1}. {line.text} ({line.confidence.toFixed(3)})
        </Text>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  verdict: {
    alignSelf: 'stretch',
    backgroundColor: theme.colors.cardBackground,
    borderRadius: theme.radius.small,
    padding: theme.spacing.medium,
    marginBottom: theme.spacing.medium,
  },
  verdictTitle: { fontSize: 18, fontWeight: '700', marginBottom: 4 },
  verdictLine: { fontSize: 13, color: theme.colors.textSecondary },
  pass: { color: '#1e8e3e' },
  fail: { color: theme.colors.errorText },
  difference: { fontSize: 12, color: theme.colors.errorText, marginTop: 4 },
  failure: { color: theme.colors.errorText, marginBottom: theme.spacing.medium },
  legend: { fontSize: 12, color: theme.colors.textMuted, marginVertical: theme.spacing.small },
  line: { alignSelf: 'stretch', fontSize: 12, color: theme.colors.textSecondary },
});
