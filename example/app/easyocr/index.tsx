/**
 * EasyOCR (CRAFT + CRNN) on device: reads one of the sample pages (the two
 * English ones, then one per other alphabet) with this package's `useOcr`,
 * optionally with the `vertical` pass, and grades the result against the text
 * the page was rendered with
 * (`assets/ocr/expected.json`). The verdict is the first thing on the screen
 * so a screenshot alone records it. The "Save detector heatmap" action writes
 * the raw CRAFT output as a fixture for the native parity test.
 */
import { Canvas, Image as SkiaImage, Path, Skia, useImage } from '@shopify/react-native-skia';
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
import { models, useOcr, type OcrDetection, type Quad } from '@swmansion/react-native-ocr';
import expected from '../../assets/ocr/expected.json';
import { Button } from '../../components/Button';
import { ModelStatus } from '../../components/ModelStatus';
import { saveCraftFixture, saveLastRun } from '../../craftFixture';
import { commonStyles, theme } from '../../theme';
import { resolveModelOverride, skImageToBuffer } from '../../utils';

type Language = keyof typeof models.ocr.EASYOCR;
const LANGUAGES = Object.keys(models.ocr.EASYOCR) as Language[];

// The screen opens on the CPU export, which runs everywhere; the accelerated
// export is an explicit choice (see the parity screen for why).
type Backend = 'XNNPACK' | 'COREML' | 'VULKAN';
const BACKENDS: Backend[] = ['XNNPACK', Platform.OS === 'ios' ? 'COREML' : 'VULKAN'];

// The English pages, then one page per other alphabet; picking a script page
// also picks its model. The CJK pages carry vertical columns next to their
// horizontal lines, so they open with the vertical pass on.
type SampleKey =
  | 'sample'
  | 'vertical'
  | 'cyrillic'
  | 'latin'
  | 'japanese'
  | 'zh_sim'
  | 'korean'
  | 'telugu'
  | 'kannada';
const SAMPLES: { key: SampleKey; title: string; vertical: boolean; language?: Language }[] = [
  { key: 'sample', title: 'Page', vertical: false },
  { key: 'vertical', title: 'Vertical page', vertical: true },
  { key: 'cyrillic', title: 'Cyrillic', vertical: false, language: 'CYRILLIC' },
  { key: 'latin', title: 'Latin', vertical: false, language: 'LATIN' },
  { key: 'japanese', title: 'Japanese', vertical: true, language: 'JAPANESE' },
  { key: 'zh_sim', title: 'Chinese', vertical: true, language: 'ZH_SIM' },
  { key: 'korean', title: 'Korean', vertical: true, language: 'KOREAN' },
  { key: 'telugu', title: 'Telugu', vertical: false, language: 'TELUGU' },
  { key: 'kannada', title: 'Kannada', vertical: false, language: 'KANNADA' },
];

type Grade = {
  /** Expected lines read exactly (each counted once). */
  readonly found: number;
  readonly missing: string[];
  readonly extra: string[];
  /** Character accuracy: 1 - edit distance / expected length, over the joined lines. */
  readonly charAccuracy: number;
};

function editDistance(a: string, b: string): number {
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(
        previous[j]! + 1,
        current[j - 1]! + 1,
        previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    previous = current;
  }
  return previous[b.length]!;
}

/**
 * Grades order-independently: each expected line matches one read line
 * exactly, the remaining expected lines pair with the closest remaining read
 * lines (smallest edit distance first), and a read line left over joins the
 * paired line it extends with the fewest errors, which forgives a line the
 * detector split in two. The character accuracy is 1 minus the paired edit
 * distances (an unpaired expected line counts in full) over the expected
 * length; read lines that pair with nothing are reported as extras and do
 * not count against it.
 */
function grade(read: readonly string[], lines: readonly string[]): Grade {
  const pool = [...read];
  const missing: string[] = [];
  for (const line of lines) {
    const at = pool.indexOf(line);
    if (at === -1) missing.push(line);
    else pool.splice(at, 1);
  }
  const candidates = missing
    .flatMap((line, i) => pool.map((text, j) => ({ i, j, d: editDistance(text, line) })))
    .sort((a, b) => a.d - b.d);
  const paired = new Map<number, string>();
  const used = new Set<number>();
  for (const { i, j } of candidates) {
    if (paired.has(i) || used.has(j)) continue;
    paired.set(i, pool[j]!);
    used.add(j);
  }
  const extra = pool.filter((_, j) => !used.has(j));
  for (const text of [...extra]) {
    let best: { i: number; joined: string; gain: number } | undefined;
    for (const [i, current] of paired) {
      const want = missing[i]!;
      for (const joined of [`${current} ${text}`, `${text} ${current}`]) {
        const gain = editDistance(current, want) - editDistance(joined, want);
        if (gain > 0 && (!best || gain > best.gain)) best = { i, joined, gain };
      }
    }
    if (best) {
      paired.set(best.i, best.joined);
      extra.splice(extra.indexOf(text), 1);
    }
  }
  let errors = 0;
  missing.forEach((line, i) => {
    const text = paired.get(i);
    errors += text === undefined ? line.length : editDistance(text, line);
  });
  const total = lines.reduce((n, line) => n + line.length, 0);
  const charAccuracy = Math.max(0, 1 - errors / total);
  return { found: lines.length - missing.length, missing, extra, charAccuracy };
}

type Outcome = {
  readonly backend: Backend;
  readonly language: Language;
  readonly sample: SampleKey;
  readonly vertical: boolean;
  readonly detections: OcrDetection[];
  readonly grade: Grade;
  readonly ms: number;
};

function quadPath(quad: Quad, scale: number) {
  const path = Skia.PathBuilder.Make().moveTo(quad[0].x * scale, quad[0].y * scale);
  for (let i = 1; i < 4; i++) path.lineTo(quad[i]!.x * scale, quad[i]!.y * scale);
  return path.close().build();
}

export default function EasyOcrScreen() {
  const pages = {
    sample: useImage(require('../../assets/ocr/sample.png')),
    vertical: useImage(require('../../assets/ocr/vertical.png')),
    cyrillic: useImage(require('../../assets/ocr/cyrillic.png')),
    latin: useImage(require('../../assets/ocr/latin.png')),
    japanese: useImage(require('../../assets/ocr/japanese.png')),
    // eslint-disable-next-line camelcase -- named after `alphabets.zh_sim`
    zh_sim: useImage(require('../../assets/ocr/zh_sim.png')),
    korean: useImage(require('../../assets/ocr/korean.png')),
    telugu: useImage(require('../../assets/ocr/telugu.png')),
    kannada: useImage(require('../../assets/ocr/kannada.png')),
  } satisfies Record<SampleKey, unknown>;
  // `/easyocr?lang=KOREAN&backend=COREML&sample=vertical&save=1` opens the
  // screen in that state (and saves the heatmap after the first run), so a
  // run can be scripted without tapping. `sample=korean` alone picks the
  // Korean model too; `lang` overrides it. `modelUrl` swaps the export for
  // another file (an unpublished candidate served over HTTP), keeping the
  // registry entry's options.
  const params = useLocalSearchParams<{
    lang?: string;
    backend?: string;
    sample?: string;
    save?: string;
    modelUrl?: string;
    runs?: string;
  }>();
  // `runs=N` runs the page N times back to back once the model is ready, so
  // a timing can be taken without tapping; `last-run.json` lists every time.
  const runs = Math.max(1, Number(params.runs) || 1);
  const msAll = useRef<number[]>([]);
  const initialSample = SAMPLES.find((s) => s.key === params.sample) ?? SAMPLES[0]!;
  const [language, setLanguage] = useState<Language>(
    LANGUAGES.find((l) => l === params.lang) ?? initialSample.language ?? 'ENGLISH'
  );
  const [backend, setBackend] = useState<Backend>(
    BACKENDS.find((b) => b === params.backend) ?? 'XNNPACK'
  );
  const [sample, setSample] = useState<SampleKey>(initialSample.key);
  const [vertical, setVertical] = useState(initialSample.vertical);
  const model = useMemo(() => {
    const entry = models.ocr.EASYOCR[language][backend];
    const override = resolveModelOverride(params.modelUrl);
    return override ? { ...entry, modelPath: override } : entry;
  }, [language, backend, params.modelUrl]);
  const { isReady, error, downloadProgress, resource, runOcr } = useOcr(model);
  const image = pages[sample];

  useEffect(() => {
    console.log(`[rnocr-easyocr] ${Platform.OS} ${backend} ${language} loading`);
  }, [backend, language]);
  useEffect(() => {
    if (isReady) console.log(`[rnocr-easyocr] ${Platform.OS} ${backend} ${language} loaded`);
    if (error) {
      console.log(`[rnocr-easyocr] ${Platform.OS} ${backend} ${language} failed: ${error.message}`);
    }
  }, [isReady, error, backend, language]);

  const [outcome, setOutcome] = useState<Outcome>();
  const [failure, setFailure] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [busy, setBusy] = useState(false);
  const { width: windowWidth } = useWindowDimensions();

  const ready = !!image && !!runOcr;

  const run = async () => {
    if (!image || !runOcr) return;
    setBusy(true);
    setFailure(undefined);
    try {
      const started = Date.now();
      const result = await runOcr(skImageToBuffer(image), { vertical });
      const ms = Date.now() - started;
      const texts = result.detections.map((d) => d.text);
      const next: Outcome = {
        backend,
        language,
        sample,
        vertical,
        detections: result.detections,
        grade: grade(texts, expected[sample]),
        ms,
      };
      console.log(
        `[rnocr-easyocr] ${Platform.OS} ${backend} ${language} ${sample} vertical=${vertical} ` +
          `${next.grade.found}/${expected[sample].length} expected lines read, ` +
          `${(next.grade.charAccuracy * 100).toFixed(1)}% chars, ` +
          `${texts.length} detections, ${ms} ms: ${JSON.stringify(texts)}`
      );
      msAll.current.push(ms);
      setOutcome(next);
      // Best effort: the run as a file, for inspection off the device.
      saveLastRun({
        screen: 'easyocr',
        ...next,
        msAll: [...msAll.current],
        models: { ocr: model.modelPath },
      }).catch(() => {});
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      setFailure(`${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const saveFixture = async () => {
    if (!image || !resource) return;
    setBusy(true);
    setFailure(undefined);
    try {
      const info = await saveCraftFixture(resource.modelPath, skImageToBuffer(image), sample, {
        image: `example/assets/ocr/${sample}.png`,
        model: model.modelPath,
        backend,
      });
      const line =
        `saved ${info.path} (detector ${info.detectorSize.width}x${info.detectorSize.height}, ` +
        `heatmap ${info.heatmapSize.width}x${info.heatmapSize.height})`;
      console.log(`[rnocr-easyocr] ${line}`);
      setNotice(line);
    } catch (e) {
      const err = e as { name?: string; code?: string; message?: string };
      setFailure(`${err.name ?? 'Error'} ${err.code ?? ''} ${err.message ?? String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Run as soon as the model and the page are ready, again when the page,
  // the pass or the backend changes.
  useEffect(() => {
    setOutcome(undefined);
    if (ready) {
      msAll.current = [];
      (async () => {
        for (let i = 0; i < runs; i++) await run();
        if (params.save === '1') await saveFixture();
      })();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, sample, vertical]);

  const canvasWidth = windowWidth - 2 * theme.spacing.large;
  const scale = image ? canvasWidth / image.width() : 1;
  const paths = useMemo(
    () => outcome?.detections.map((line) => quadPath(line.quad, scale)) ?? [],
    [outcome, scale]
  );

  const total = outcome ? expected[outcome.sample].length : 0;
  const perfect =
    outcome !== undefined && outcome.grade.found === total && outcome.grade.extra.length === 0;

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.languages}>
        {LANGUAGES.map((key) => (
          <Button
            key={key}
            title={key.toLowerCase()}
            variant={key === language ? 'primary' : 'secondary'}
            onPress={() => setLanguage(key)}
            disabled={busy}
            style={styles.language}
          />
        ))}
      </ScrollView>
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
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.languages}>
        {SAMPLES.map((entry) => (
          <Button
            key={entry.key}
            title={entry.title}
            variant={entry.key === sample ? 'primary' : 'secondary'}
            onPress={() => {
              setSample(entry.key);
              setVertical(entry.vertical);
              if (entry.language) setLanguage(entry.language);
            }}
            disabled={busy}
            style={styles.language}
          />
        ))}
      </ScrollView>
      <View style={commonStyles.buttonRow}>
        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>vertical pass</Text>
          <Switch value={vertical} onValueChange={setVertical} disabled={busy} />
        </View>
      </View>
      <Text style={styles.hint}>
        The page row picks the test image; “Vertical page” is the one with upright glyph columns.
        The switch is the per-run option vertical: true, which also reads text stacked top to
        bottom. Picking a page resets the switch to that page’s default, so you can run any page
        with the pass on or off and compare.
      </Text>
      <ModelStatus
        isReady={isReady}
        downloadProgress={downloadProgress}
        error={error?.message}
        modelTypeLabel={`EasyOCR ${language.toLowerCase()}`}
      />
      {failure && <Text style={styles.failure}>{failure}</Text>}
      {outcome && (
        <View style={styles.verdict}>
          <Text style={[styles.verdictTitle, perfect ? styles.pass : styles.fail]}>
            read: {outcome.grade.found}/{total} expected lines ·{' '}
            {(outcome.grade.charAccuracy * 100).toFixed(1)}% chars
          </Text>
          <Text style={styles.verdictLine}>
            {outcome.detections.length} detections · {outcome.ms} ms · {Platform.OS} ·{' '}
            {outcome.backend} · {outcome.language.toLowerCase()} · {outcome.sample} · vertical{' '}
            {outcome.vertical ? 'on' : 'off'}
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
          {paths.map((path, i) => (
            <Path key={i} path={path} style="stroke" strokeWidth={2} color="#39FF14" />
          ))}
        </Canvas>
      )}

      <View style={commonStyles.buttonRow}>
        <Button title="Run again" onPress={run} disabled={!ready || busy} loading={busy} />
        <Button
          title="Save detector heatmap"
          variant="secondary"
          onPress={saveFixture}
          disabled={!ready || busy}
        />
      </View>
      {notice && <Text style={styles.notice}>{notice}</Text>}

      {outcome?.detections.map((line, i) => (
        <Text key={i} style={styles.line}>
          {i + 1}. {line.text} ({line.confidence.toFixed(3)})
        </Text>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  languages: { alignSelf: 'stretch', flexGrow: 0, marginBottom: theme.spacing.medium },
  language: { marginRight: theme.spacing.small },
  switchRow: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing.small },
  switchLabel: { fontSize: 13, color: theme.colors.textSecondary },
  hint: {
    alignSelf: 'stretch',
    fontSize: 12,
    lineHeight: 17,
    color: theme.colors.textMuted,
    marginBottom: theme.spacing.large,
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
  notice: { fontSize: 12, color: theme.colors.textMuted, marginBottom: theme.spacing.small },
  line: { alignSelf: 'stretch', fontSize: 12, color: theme.colors.textSecondary },
});
