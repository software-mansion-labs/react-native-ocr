/**
 * Document helpers on device: reads a page with PP-OCRv6 composed with the
 * PaddleHelpers model (`documentModels` on the config) and, optionally, the
 * PP-DocLayout detector, with the orientation, dewarp and tables stages
 * toggled per run. Three bundled pages exercise one stage each (`rotated.png`
 * is the parity page turned 90 degrees, `curved.png` is the same page curled
 * on a table, `table.png` carries a ruled table) and are graded against
 * `assets/ocr/expected.json`; a photo from the library or the camera can be
 * read too, ungraded. The canvas shows the corrected frame the result is
 * relative to, so a rotation or a dewarp is visible.
 */
import {
  AlphaType,
  Canvas,
  ColorType,
  Image as SkiaImage,
  Path,
  Rect,
  Skia,
  useImage,
  type SkImage,
} from '@shopify/react-native-skia';
import { useLocalSearchParams } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
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
  type ImageBuffer,
  type OcrModel,
  type OcrResult,
  type Quad,
} from '@swmansion/react-native-ocr';
import expected from '../../assets/ocr/expected.json';
import { Button } from '../../components/Button';
import { ModelStatus } from '../../components/ModelStatus';
import { saveLastRun } from '../../craftFixture';
import { commonStyles, theme } from '../../theme';
import { getImage, skImageToBuffer } from '../../utils';

type Backend = 'XNNPACK' | 'COREML' | 'VULKAN';
const BACKENDS: Backend[] = ['XNNPACK', Platform.OS === 'ios' ? 'COREML' : 'VULKAN'];

type SampleKey = 'rotated' | 'curved' | 'table' | 'photo';
type Stages = {
  readonly orientation: boolean;
  readonly dewarp: boolean;
  readonly layout: boolean;
  readonly tables: boolean;
};
/** The bundled pages, each opening with the stage it exercises switched on. */
const SAMPLES: { key: SampleKey; title: string; stages: Stages }[] = [
  {
    key: 'rotated',
    title: 'Rotated',
    stages: { orientation: true, dewarp: false, layout: false, tables: false },
  },
  {
    key: 'curved',
    title: 'Curved',
    stages: { orientation: false, dewarp: true, layout: false, tables: false },
  },
  {
    key: 'table',
    title: 'Table',
    stages: { orientation: false, dewarp: false, layout: true, tables: true },
  },
];
const PHOTO_STAGES: Stages = { orientation: true, dewarp: true, layout: true, tables: true };
/** `dewarpMinVarianceRatio` for photos: the flattened page keeps about a third. */
const PHOTO_DEWARP_RATIO = 0.1;

const REGION_COLORS = new Map<string, string>([
  ['doc_title', '#d81b60'],
  ['paragraph_title', '#8e24aa'],
  ['text', '#1e88e5'],
  ['table', '#43a047'],
  ['image', '#fb8c00'],
]);
const regionColor = (label: string) => REGION_COLORS.get(label) ?? '#546e7a';

type Grade = {
  readonly found: number;
  readonly total: number;
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
  return { found: lines.length - missing.length, total: lines.length, missing, extra: pool };
}

/** The rows of a filled table skeleton, each as its cell texts. */
function tableRows(html: string): string[][] {
  return (html.match(/<tr>[\s\S]*?<\/tr>/g) ?? []).map((row) =>
    (row.match(/<td[^>]*>([\s\S]*?)<\/td>/g) ?? []).map((cell) =>
      cell.replace(/<td[^>]*>|<\/td>/g, '')
    )
  );
}

type Outcome = {
  readonly backend: Backend;
  readonly helpersBackend: Backend;
  readonly sample: SampleKey;
  readonly stages: Stages;
  readonly result: OcrResult<DocLayoutLabel>;
  /** Whether orientation or dewarp produced a new frame. */
  readonly corrected: boolean;
  readonly grade?: Grade;
  readonly ms: number;
};

function quadPath(quad: Quad, scale: number) {
  const path = Skia.PathBuilder.Make().moveTo(quad[0].x * scale, quad[0].y * scale);
  for (let i = 1; i < 4; i++) path.lineTo(quad[i]!.x * scale, quad[i]!.y * scale);
  return path.close().build();
}

/** Wraps a result frame (RGBA, as `skImageToBuffer` produces it) for the canvas. */
function bufferToSkImage(image: ImageBuffer): SkImage | null {
  if (image.format !== 'rgba') return null;
  return Skia.Image.MakeImage(
    {
      width: image.width,
      height: image.height,
      colorType: ColorType.RGBA_8888,
      alphaType: AlphaType.Unpremul,
    },
    Skia.Data.fromBytes(image.data),
    image.width * 4
  );
}

/**
 * Whether the result frame differs from the input. The result crosses the
 * worklet runtime by copy, so identity says nothing; the bytes do.
 */
function isCorrected(input: ImageBuffer, image: ImageBuffer): boolean {
  if (image.width !== input.width || image.height !== input.height) return true;
  if (image.data.length !== input.data.length) return true;
  for (let i = 0; i < image.data.length; i++) {
    if (image.data[i] !== input.data[i]) return true;
  }
  return false;
}

function describeBlock(block: DocumentBlock<DocLayoutLabel>): string {
  const { xmin, ymin, xmax, ymax } = block.bbox;
  const bounds = `${Math.round(xmin)},${Math.round(ymin)} ${Math.round(xmax - xmin)}x${Math.round(ymax - ymin)}`;
  return `${block.regionType} ${block.layoutConfidence.toFixed(2)} at ${bounds}, ${block.lines.length} lines`;
}

function describeStages(stages: Stages): string {
  return (Object.keys(stages) as (keyof Stages)[]).map((k) => `${k}=${+stages[k]}`).join(' ');
}

export default function DocumentScreen() {
  const pages: Record<Exclude<SampleKey, 'photo'>, SkImage | null> = {
    rotated: useImage(require('../../assets/ocr/rotated.png')),
    curved: useImage(require('../../assets/ocr/curved.png')),
    table: useImage(require('../../assets/ocr/table.png')),
  };
  const [photoUri, setPhotoUri] = useState<string>();
  const photo = useImage(photoUri ?? null);

  // `/document?sample=table&backend=COREML&orientation=1&dewarp=0&layout=1&tables=1`
  // opens the screen in that state; a stage left out keeps the sample's default.
  // The helper model has its own backend (`helpersBackend`) so its accelerated
  // export can be tried while the OCR model stays on the CPU (PP-OCRv6's
  // Vulkan export crashes at load, see the parity screen).
  // `dewarpRatio` overrides `dewarpMinVarianceRatio` (0 accepts every warp);
  // the curved page defaults to 0.1, since the flattened page keeps only about
  // a third of the variance of the photo once the dark table is cropped away.
  const params = useLocalSearchParams<{
    sample?: string;
    backend?: string;
    orientation?: string;
    dewarp?: string;
    layout?: string;
    tables?: string;
    dewarpRatio?: string;
    helpersBackend?: string;
  }>();
  const initialSample = SAMPLES.find((s) => s.key === params.sample) ?? SAMPLES[0]!;
  const flag = (value: string | undefined, fallback: boolean) =>
    value === undefined ? fallback : value === '1';
  const [backend, setBackend] = useState<Backend>(
    BACKENDS.find((b) => b === params.backend) ?? 'XNNPACK'
  );
  const [helpersBackend, setHelpersBackend] = useState<Backend>(
    BACKENDS.find((b) => b === params.helpersBackend) ?? 'XNNPACK'
  );
  const [sample, setSample] = useState<SampleKey>(initialSample.key);
  const [stages, setStages] = useState<Stages>({
    orientation: flag(params.orientation, initialSample.stages.orientation),
    dewarp: flag(params.dewarp, initialSample.stages.dewarp),
    layout: flag(params.layout, initialSample.stages.layout),
    tables: flag(params.tables, initialSample.stages.tables),
  });
  const setStage = (key: keyof Stages) => (value: boolean) =>
    setStages((s) => ({
      ...s,
      [key]: value,
      ...(key === 'layout' && !value ? { tables: false } : {}),
    }));
  const pickSample = (key: SampleKey) => {
    setSample(key);
    setStages(SAMPLES.find((s) => s.key === key)?.stages ?? PHOTO_STAGES);
  };

  // Orientation, dewarp and tables are per-run options, so toggling them does
  // not reload; the layout model is part of the config, so it does.
  const dewarpRatio =
    params.dewarpRatio !== undefined
      ? Number(params.dewarpRatio)
      : sample === 'curved' || sample === 'photo'
        ? PHOTO_DEWARP_RATIO
        : undefined;
  const config = useMemo<OcrModel<DocLayoutLabel>>(
    () => ({
      ...models.ocr.PADDLE.PPOCRV6_SMALL[backend],
      documentModels: {
        ...models.documentModels.PP_HELPERS[helpersBackend],
        ...(dewarpRatio === undefined ? {} : { dewarpMinVarianceRatio: dewarpRatio }),
      },
      ...(stages.layout ? { layout: models.layoutDetection.PP_DOCLAYOUT[backend] } : {}),
    }),
    [backend, helpersBackend, stages.layout, dewarpRatio]
  );
  const { isReady, error, downloadProgress, runOcr } = useOcr<DocLayoutLabel>(config);

  const setup =
    `${backend} helpers=${helpersBackend} ${sample} ${describeStages(stages)} ` +
    `dewarpRatio=${dewarpRatio ?? 'default'}`;
  useEffect(() => {
    console.log(`[rnocr-document] ${Platform.OS} ${setup} loading`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config]);
  useEffect(() => {
    if (isReady) console.log(`[rnocr-document] ${Platform.OS} ${setup} loaded`);
    if (error) console.log(`[rnocr-document] ${Platform.OS} ${setup} failed: ${error.message}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isReady, error]);

  const [outcome, setOutcome] = useState<Outcome>();
  const [failure, setFailure] = useState<string>();
  const [busy, setBusy] = useState(false);
  const { width: windowWidth } = useWindowDimensions();

  const image = sample === 'photo' ? photo : pages[sample];
  const ready = !!image && !!runOcr;

  const run = async () => {
    if (!image || !runOcr) return;
    setBusy(true);
    setFailure(undefined);
    try {
      const started = Date.now();
      const input = skImageToBuffer(image);
      const result = await runOcr(input, {
        orientation: stages.orientation,
        dewarp: stages.dewarp,
        tables: stages.tables,
      });
      const ms = Date.now() - started;
      const texts = result.detections.map((d) => d.text);
      const next: Outcome = {
        backend,
        helpersBackend,
        sample,
        stages,
        result,
        corrected: isCorrected(input, result.image),
        grade: sample === 'photo' ? undefined : grade(texts, expected[sample]),
        ms,
      };
      const tables = result.blocks
        .filter((b) => b.tableHtml !== undefined)
        .map((b) => tableRows(b.tableHtml!));
      console.log(
        `[rnocr-document] ${Platform.OS} ${setup} ` +
          (next.grade ? `${next.grade.found}/${next.grade.total} expected lines read, ` : '') +
          `${texts.length} detections, ${result.regions.length} regions, ` +
          `${result.blocks.length} blocks, frame ${result.image.width}x${result.image.height} ` +
          `${next.corrected ? 'corrected' : 'unchanged'}, ` +
          `${ms} ms; tables ${JSON.stringify(tables)}; ` +
          `blocks ${JSON.stringify(result.blocks.map(describeBlock))}`
      );
      setOutcome(next);
      // Best effort: the run as a file, for inspection off the device.
      saveLastRun({
        screen: 'document',
        backend,
        helpersBackend,
        sample,
        stages,
        ms,
        frame: {
          width: result.image.width,
          height: result.image.height,
          corrected: next.corrected,
        },
        grade: next.grade,
        regions: result.regions,
        blocks: result.blocks,
        detections: result.detections,
      }).catch(() => {});
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      const line = `${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`;
      console.log(`[rnocr-document] ${Platform.OS} ${setup} run failed: ${line}`);
      setFailure(line);
    } finally {
      setBusy(false);
    }
  };

  // Run as soon as the models and the page are ready, again on every reload
  // and on every change of page or stage. A change of page or layout also
  // changes the config, and the runner lags that by one load: a run on the
  // previous runner would race its disposal, so the run waits for the runner
  // of this config.
  const loadedFor = useRef<OcrModel<DocLayoutLabel> | undefined>(undefined);
  useEffect(() => {
    loadedFor.current = runOcr ? config : undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runOcr]);
  useEffect(() => {
    setOutcome(undefined);
    if (ready && loadedFor.current === config) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, image, stages, runOcr, config]);

  const pickPhoto = async (useCamera: boolean) => {
    const uri = await getImage(useCamera, 1200);
    if (!uri) return;
    setPhotoUri(uri);
    pickSample('photo');
  };

  // The canvas shows the corrected frame once there is a result for the
  // current page, else the page itself.
  const shown = useMemo(
    () => (outcome ? bufferToSkImage(outcome.result.image) : null) ?? image,
    [outcome, image]
  );
  const canvasWidth = windowWidth - 2 * theme.spacing.large;
  const scale = shown ? canvasWidth / shown.width() : 1;
  const linePaths = useMemo(
    () => outcome?.result.detections.map((line) => quadPath(line.quad, scale)) ?? [],
    [outcome, scale]
  );

  const tableBlocks = outcome?.result.blocks.filter((b) => b.tableHtml !== undefined) ?? [];
  const expectedShape = expected.tableShape;
  const shapeOf = (rows: string[][]) => [rows.length, Math.max(0, ...rows.map((r) => r.length))];
  const perfect =
    outcome?.grade !== undefined &&
    outcome.grade.found === outcome.grade.total &&
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
      <Text style={styles.rowLabel}>helper model</Text>
      <View style={commonStyles.buttonRow}>
        {BACKENDS.map((key) => (
          <Button
            key={key}
            title={key}
            variant={key === helpersBackend ? 'primary' : 'secondary'}
            onPress={() => setHelpersBackend(key)}
            disabled={busy}
          />
        ))}
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.samples}>
        {SAMPLES.map((s) => (
          <Button
            key={s.key}
            title={s.title}
            variant={s.key === sample ? 'primary' : 'secondary'}
            onPress={() => pickSample(s.key)}
            disabled={busy}
            style={styles.sample}
          />
        ))}
        <Button
          title="Photo…"
          variant={sample === 'photo' ? 'primary' : 'secondary'}
          onPress={() => pickPhoto(false)}
          disabled={busy}
          style={styles.sample}
        />
        <Button
          title="Camera…"
          variant="secondary"
          onPress={() => pickPhoto(true)}
          disabled={busy}
          style={styles.sample}
        />
      </ScrollView>
      <View style={styles.switches}>
        {(['orientation', 'dewarp', 'layout', 'tables'] as const).map((key) => (
          <View key={key} style={styles.switchRow}>
            <Text style={styles.switchLabel}>{key}</Text>
            <Switch
              value={stages[key]}
              onValueChange={setStage(key)}
              disabled={busy || (key === 'tables' && !stages.layout)}
            />
          </View>
        ))}
      </View>
      <ModelStatus
        isReady={isReady}
        downloadProgress={downloadProgress}
        error={error?.message}
        modelTypeLabel={
          `PP-OCRv6 ${backend} + PaddleHelpers ${helpersBackend}` +
          (stages.layout ? ` + PP-DocLayout ${backend}` : '')
        }
      />
      {failure && <Text style={styles.failure}>{failure}</Text>}
      {outcome && (
        <View style={styles.verdict}>
          {outcome.grade ? (
            <Text style={[styles.verdictTitle, perfect ? styles.pass : styles.fail]}>
              read: {outcome.grade.found}/{outcome.grade.total} expected lines
            </Text>
          ) : (
            <Text style={styles.verdictTitle}>read: {outcome.result.detections.length} lines</Text>
          )}
          <Text style={styles.verdictLine}>
            {outcome.result.detections.length} detections · {outcome.result.regions.length} regions
            · {outcome.result.blocks.length} blocks · frame {outcome.result.image.width}x
            {outcome.result.image.height} {outcome.corrected ? 'corrected' : 'unchanged'} ·{' '}
            {outcome.ms} ms · {Platform.OS} · {outcome.backend} · helpers {outcome.helpersBackend}
          </Text>
          <Text style={styles.verdictLine}>{describeStages(outcome.stages)}</Text>
          {outcome.sample === 'table' && (
            <Text
              style={[
                styles.verdictLine,
                tableBlocks.length === 1 &&
                String(shapeOf(tableRows(tableBlocks[0]!.tableHtml!))) === String(expectedShape)
                  ? styles.pass
                  : styles.fail,
              ]}
            >
              table structure: {tableBlocks.length} table(s)
              {tableBlocks.map((b) => ` ${shapeOf(tableRows(b.tableHtml!)).join('x')}`)}, expected{' '}
              {expectedShape.join('x')}
            </Text>
          )}
          {outcome.grade?.missing.map((line) => (
            <Text key={`m${line}`} style={styles.difference}>
              missing "{line}"
            </Text>
          ))}
          {outcome.grade?.extra.map((line, i) => (
            <Text key={`e${i}`} style={styles.difference}>
              extra "{line}"
            </Text>
          ))}
        </View>
      )}

      {shown && (
        <Canvas style={{ width: canvasWidth, height: shown.height() * scale }}>
          <SkiaImage
            image={shown}
            x={0}
            y={0}
            width={canvasWidth}
            height={shown.height() * scale}
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
          {block.tableHtml !== undefined ? (
            <View style={styles.table}>
              {tableRows(block.tableHtml).map((row, r) => (
                <Text key={r} style={styles.tableRow}>
                  {row.join(' | ')}
                </Text>
              ))}
              <Text style={styles.html}>{block.tableHtml}</Text>
            </View>
          ) : (
            <Text style={styles.blockText}>{block.text}</Text>
          )}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  samples: { alignSelf: 'stretch', flexGrow: 0, marginBottom: theme.spacing.medium },
  sample: { marginRight: theme.spacing.small },
  switches: {
    alignSelf: 'stretch',
    flexDirection: 'row',
    flexWrap: 'wrap',
    columnGap: theme.spacing.large,
    rowGap: theme.spacing.small,
    marginBottom: theme.spacing.large,
  },
  switchRow: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing.small },
  switchLabel: { fontSize: 13, color: theme.colors.textSecondary },
  rowLabel: {
    alignSelf: 'stretch',
    fontSize: 13,
    color: theme.colors.textSecondary,
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
  table: { marginTop: 2 },
  tableRow: { fontSize: 12, fontFamily: Platform.select({ ios: 'Menlo', default: 'monospace' }) },
  html: { fontSize: 10, color: theme.colors.textSecondary, marginTop: 4 },
});
