/**
 * Layout on device: reads the paper-like `layout.png` with PP-OCRv6 composed
 * with the PP-DocLayout detector (`layout` on the config), draws the regions
 * the detector found and lists the blocks in the order the pipeline returns
 * them, each with its label and the lines it read. The `layout` switch
 * reruns the same page as whole-page OCR for comparison, and the lines read
 * are graded against `assets/ocr/expected.json` either way.
 */
import { Canvas, Image as SkiaImage, Path, Rect, Skia, useImage } from '@shopify/react-native-skia';
import { useLocalSearchParams } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import {
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import {
  models,
  useOcr,
  type DocLayoutLabel,
  type DocumentBlock,
  type OcrDetection,
  type OcrModel,
  type OcrResult,
  type Quad,
} from '@swmansion/react-native-ocr';
import expected from '../../assets/ocr/expected.json';
import { Button } from '../../components/Button';
import { ModelStatus } from '../../components/ModelStatus';
import { saveLastRun } from '../../craftFixture';
import { commonStyles, theme } from '../../theme';
import { skImageToBuffer } from '../../utils';

type Backend = 'XNNPACK' | 'COREML' | 'VULKAN';
const BACKENDS: Backend[] = ['XNNPACK', Platform.OS === 'ios' ? 'COREML' : 'VULKAN'];

const EXPECTED = expected.layout;

// One color per family of PP-DocLayout labels, so the regions read at a glance.
const REGION_COLORS = new Map<string, string>([
  ['doc_title', '#d81b60'],
  ['paragraph_title', '#8e24aa'],
  ['text', '#1e88e5'],
  ['abstract', '#00897b'],
  ['image', '#fb8c00'],
  ['chart', '#fb8c00'],
  ['figure_title', '#6d4c41'],
  ['table', '#43a047'],
  ['footnote', '#757575'],
]);
const regionColor = (label: string) => REGION_COLORS.get(label) ?? '#546e7a';

type Grade = {
  readonly found: number;
  readonly missing: string[];
  readonly extra: string[];
};

/** Each expected line matches one read line exactly; order is not graded. */
function grade(read: readonly string[], lines: readonly string[]): Grade {
  const pool = [...read];
  const missing: string[] = [];
  for (const line of lines) {
    const at = pool.indexOf(line);
    if (at === -1) missing.push(line);
    else pool.splice(at, 1);
  }
  return { found: lines.length - missing.length, missing, extra: pool };
}

type Outcome = {
  readonly backend: Backend;
  readonly layout: boolean;
  readonly layoutBackend: Backend;
  readonly result: OcrResult<DocLayoutLabel>;
  readonly grade: Grade;
  readonly ms: number;
};

function quadPath(quad: Quad, scale: number) {
  const path = Skia.PathBuilder.Make().moveTo(quad[0].x * scale, quad[0].y * scale);
  for (let i = 1; i < 4; i++) path.lineTo(quad[i]!.x * scale, quad[i]!.y * scale);
  return path.close().build();
}

function describeBlock(block: DocumentBlock<DocLayoutLabel>): string {
  const { xmin, ymin, xmax, ymax } = block.bbox;
  const bounds = `${Math.round(xmin)},${Math.round(ymin)} ${Math.round(xmax - xmin)}x${Math.round(ymax - ymin)}`;
  return `${block.regionType} ${block.layoutConfidence.toFixed(2)} at ${bounds}, ${block.lines.length} lines`;
}

export default function LayoutScreen() {
  const image = useImage(require('../../assets/ocr/layout.png'));
  // `/layout?backend=COREML&layoutBackend=XNNPACK&layout=0` opens the screen
  // in that state. The layout model has its own backend so its accelerated
  // export can be tried while the OCR model stays on the CPU (PP-OCRv6's
  // Vulkan export crashes at load, see the parity screen).
  const params = useLocalSearchParams<{
    backend?: string;
    layoutBackend?: string;
    layout?: string;
  }>();
  const [backend, setBackend] = useState<Backend>(
    BACKENDS.find((b) => b === params.backend) ?? 'XNNPACK'
  );
  const [layoutBackend, setLayoutBackend] = useState<Backend>(
    BACKENDS.find((b) => b === params.layoutBackend) ?? 'XNNPACK'
  );
  const [layoutOn, setLayoutOn] = useState(params.layout !== '0');
  const config = useMemo<OcrModel<DocLayoutLabel>>(
    () => ({
      ...models.ocr.PADDLE.PPOCRV6_SMALL[backend],
      ...(layoutOn ? { layout: models.layoutDetection.PP_DOCLAYOUT[layoutBackend] } : {}),
    }),
    [backend, layoutBackend, layoutOn]
  );
  const { isReady, error, downloadProgress, runOcr } = useOcr<DocLayoutLabel>(config);

  const setup = `${backend} layout=${layoutOn ? layoutBackend : 'off'}`;
  useEffect(() => {
    console.log(`[rnocr-layout] ${Platform.OS} ${setup} loading`);
  }, [setup]);
  useEffect(() => {
    if (isReady) console.log(`[rnocr-layout] ${Platform.OS} ${setup} loaded`);
    if (error) console.log(`[rnocr-layout] ${Platform.OS} ${setup} failed: ${error.message}`);
  }, [isReady, error, setup]);

  const [outcome, setOutcome] = useState<Outcome>();
  const [failure, setFailure] = useState<string>();
  const [busy, setBusy] = useState(false);
  const { width: windowWidth } = useWindowDimensions();

  const ready = !!image && !!runOcr;

  const run = async () => {
    if (!image || !runOcr) return;
    setBusy(true);
    setFailure(undefined);
    try {
      const started = Date.now();
      const result = await runOcr(skImageToBuffer(image));
      const ms = Date.now() - started;
      const texts = result.detections.map((d) => d.text);
      const next: Outcome = {
        backend,
        layout: layoutOn,
        layoutBackend,
        result,
        grade: grade(texts, EXPECTED),
        ms,
      };
      const regions = result.regions.map((r) => `${r.label}:${r.confidence.toFixed(2)}`);
      console.log(
        `[rnocr-layout] ${Platform.OS} ${setup} ` +
          `${next.grade.found}/${EXPECTED.length} expected lines read, ` +
          `${texts.length} detections, ${result.regions.length} regions, ` +
          `${result.blocks.length} blocks, ${ms} ms; regions ${JSON.stringify(regions)}; ` +
          `blocks ${JSON.stringify(result.blocks.map(describeBlock))}`
      );
      setOutcome(next);
      // Best effort: the run as a file, for inspection off the device.
      saveLastRun({
        screen: 'layout',
        backend,
        layout: layoutOn,
        layoutBackend,
        ms,
        grade: next.grade,
        regions: result.regions,
        blocks: result.blocks,
        detections: result.detections,
      }).catch(() => {});
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      const line = `${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`;
      console.log(`[rnocr-layout] ${Platform.OS} ${setup} run failed: ${line}`);
      setFailure(line);
    } finally {
      setBusy(false);
    }
  };

  // Run as soon as the models and the page are ready, again on every reload.
  useEffect(() => {
    setOutcome(undefined);
    if (ready) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const canvasWidth = windowWidth - 2 * theme.spacing.large;
  const scale = image ? canvasWidth / image.width() : 1;
  const linePaths = useMemo(
    () => outcome?.result.detections.map((line: OcrDetection) => quadPath(line.quad, scale)) ?? [],
    [outcome, scale]
  );

  const perfect =
    outcome !== undefined &&
    outcome.grade.found === EXPECTED.length &&
    outcome.grade.extra.length === 0;

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <Text style={styles.rowLabel}>OCR model</Text>
      <View style={commonStyles.buttonRow}>
        {BACKENDS.map((key) => (
          <Button
            key={key}
            title={key}
            variant={key === backend ? 'primary' : 'secondary'}
            onPress={() => setBackend(key)}
            disabled={busy}
          />
        ))}
      </View>
      <View style={styles.labelRow}>
        <Text style={styles.switchLabel}>layout model</Text>
        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>layout</Text>
          <Switch value={layoutOn} onValueChange={setLayoutOn} disabled={busy} />
        </View>
      </View>
      <View style={commonStyles.buttonRow}>
        {BACKENDS.map((key) => (
          <Button
            key={key}
            title={key}
            variant={key === layoutBackend ? 'primary' : 'secondary'}
            onPress={() => setLayoutBackend(key)}
            disabled={busy || !layoutOn}
          />
        ))}
      </View>
      <ModelStatus
        isReady={isReady}
        downloadProgress={downloadProgress}
        error={error?.message}
        modelTypeLabel={
          layoutOn ? `PP-OCRv6 ${backend} + PP-DocLayout ${layoutBackend}` : `PP-OCRv6 ${backend}`
        }
      />
      {failure && <Text style={styles.failure}>{failure}</Text>}
      {outcome && (
        <View style={styles.verdict}>
          <Text style={[styles.verdictTitle, perfect ? styles.pass : styles.fail]}>
            read: {outcome.grade.found}/{EXPECTED.length} expected lines
          </Text>
          <Text style={styles.verdictLine}>
            {outcome.result.detections.length} detections · {outcome.result.regions.length} regions
            · {outcome.result.blocks.length} blocks · {outcome.ms} ms · {Platform.OS} ·{' '}
            {outcome.backend} · layout {outcome.layout ? outcome.layoutBackend : 'off'}
          </Text>
          {outcome.grade.missing.map((line) => (
            <Text key={`m${line}`} style={styles.difference}>
              missing "{line}"
            </Text>
          ))}
          {outcome.grade.extra.map((line, i) => (
            <Text key={`e${i}`} style={styles.difference}>
              extra "{line}"
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
          {outcome?.result.regions.map((region, i) => (
            <Rect
              key={`r${i}`}
              x={region.box.xmin * scale}
              y={region.box.ymin * scale}
              width={(region.box.xmax - region.box.xmin) * scale}
              height={(region.box.ymax - region.box.ymin) * scale}
              style="stroke"
              strokeWidth={3}
              color={regionColor(region.label)}
            />
          ))}
          {linePaths.map((path, i) => (
            <Path key={`l${i}`} path={path} style="stroke" strokeWidth={1.5} color="#39FF14" />
          ))}
        </Canvas>
      )}

      <View style={commonStyles.buttonRow}>
        <Button title="Run again" onPress={run} disabled={!ready || busy} loading={busy} />
      </View>

      {outcome?.result.blocks.map((block, i) => (
        <View key={i} style={styles.block}>
          <Text style={[styles.blockTitle, { color: regionColor(block.regionType) }]}>
            {i + 1}. {describeBlock(block)}
          </Text>
          <Text style={styles.blockText}>{block.text}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  switchRow: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing.small },
  switchLabel: { fontSize: 13, color: theme.colors.textSecondary },
  rowLabel: {
    alignSelf: 'stretch',
    fontSize: 13,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.small,
  },
  labelRow: {
    alignSelf: 'stretch',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.small,
  },
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
  block: { alignSelf: 'stretch', marginTop: theme.spacing.medium },
  blockTitle: { fontSize: 13, fontWeight: '600' },
  blockText: { fontSize: 12, color: theme.colors.textSecondary, marginTop: 2 },
});
