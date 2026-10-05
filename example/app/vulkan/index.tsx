/**
 * Vulkan diagnostics (Android): loads Vulkan exports one at a time, so that a
 * crash inside the GPU driver can be pinned to a model, and for PP-OCRv6 to
 * one of its two methods. Every step logs a `[rnocr-vulkan]` line before it
 * starts: after a crash, the last line in the device log names the culprit.
 */
import { useRef, useState } from 'react';
import { ScrollView, StyleSheet, Text } from 'react-native';
import { download, loadModel, tensor, wrapAsync, type Model } from 'react-native-executorch';
import { models } from '@swmansion/react-native-ocr';
import { Button } from '../../components/Button';
import { commonStyles, theme } from '../../theme';

const HF = 'https://huggingface.co/software-mansion/react-native-executorch';
const TAG = 'resolve/v0.10.0';

const OTHER_MODELS = [
  {
    name: 'MiniLM fp16 (transformer)',
    url: `${HF}-all-MiniLM-L6-v2/${TAG}/vulkan/all_minilm_l6_v2_vulkan_fp16.pte`,
  },
  {
    name: 'CLIP image fp16 (ViT)',
    url: `${HF}-clip-vit-base-patch32/${TAG}/vulkan/clip_vit_base_patch32_image_vulkan_fp16.pte`,
  },
  {
    name: 'Whisper tiny fp16',
    url: `${HF}-whisper-tiny/${TAG}/vulkan/whisper_tiny_vulkan_fp16.pte`,
  },
];

const OCR_URL = models.ocr.PADDLE.PPOCRV6_SMALL.VULKAN.modelPath;

const loadEager = wrapAsync((path: string) => {
  'worklet';
  const model = loadModel(path);
  const methods = Object.keys(model.schema);
  const backends = JSON.stringify(model.backends);
  model.dispose();
  return `${methods.join(', ')} ${backends}`;
});

const loadLazy = wrapAsync((path: string) => {
  'worklet';
  return loadModel(path, { eagerLoadMethods: false });
});

/** Runs one method on zeros at the smallest input size the model declares. */
const runMethod = wrapAsync((model: Model, method: string) => {
  'worklet';
  const smallest = (dim: Model['schema'][string]['inputs'][number] & { kind: 'Tensor' }) =>
    dim.shape.map((d) =>
      d.kind === 'constant' ? d.value : d.kind === 'range' ? d.range.min : Math.min(...d.choices)
    );
  const spec = model.schema[method]!;
  const inputSpec = spec.inputs[0]!;
  const outputSpec = spec.outputs[0]!;
  if (inputSpec.kind !== 'Tensor' || outputSpec.kind !== 'Tensor') {
    throw new Error(`${method}: expected tensor parameters`);
  }
  const inputShape = smallest(inputSpec);
  const outputShape = smallest(outputSpec);
  const input = tensor('float32', inputShape);
  const output = tensor('float32', outputShape);
  try {
    const started = Date.now();
    model.execute(method, [input], [output]);
    return `in [${inputShape}] out [${outputShape}] in ${Date.now() - started} ms`;
  } finally {
    input.dispose();
    output.dispose();
  }
});

export default function VulkanCheckScreen() {
  const [lines, setLines] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const ocrModel = useRef<Model | null>(null);

  const step = async (name: string, body: () => Promise<string>) => {
    setBusy(true);
    console.log(`[rnocr-vulkan] START ${name}`);
    let line: string;
    try {
      line = `OK ${name}: ${await body()}`;
    } catch (e) {
      const err = e as { code?: string; message?: string };
      line = `FAILED ${name}: ${err.code ?? ''} ${err.message ?? String(e)}`;
    }
    console.log(`[rnocr-vulkan] ${line}`);
    setLines((previous) => [...previous, line]);
    setBusy(false);
  };

  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <Text style={commonStyles.description}>
        Tap top to bottom. A crash closes the app; the device log then names the step.
      </Text>
      {OTHER_MODELS.map(({ name, url }) => (
        <Button
          key={name}
          title={`Load ${name}`}
          variant="secondary"
          disabled={busy}
          onPress={() => step(`load ${name}`, async () => loadEager(await download(url)))}
        />
      ))}
      <Button
        title="PP-OCRv6: open without loading methods"
        disabled={busy}
        onPress={() =>
          step('PP-OCRv6 lazy open', async () => {
            ocrModel.current?.dispose();
            ocrModel.current = await loadLazy(await download(OCR_URL));
            return Object.keys(ocrModel.current.schema).join(', ');
          })
        }
      />
      {['recognize', 'detect'].map((method) => (
        <Button
          key={method}
          title={`PP-OCRv6: run ${method}`}
          disabled={busy}
          onPress={() =>
            step(`PP-OCRv6 ${method}`, async () => {
              if (!ocrModel.current) throw new Error('open the model first');
              return runMethod(ocrModel.current, method);
            })
          }
        />
      ))}
      {lines.map((line, i) => (
        <Text key={i} style={styles.line}>
          {line}
        </Text>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  line: {
    alignSelf: 'stretch',
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.small,
  },
});
