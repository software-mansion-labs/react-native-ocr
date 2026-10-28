import {
  alphabets,
  craftExtractBoxes,
  dbnetExtractBoxes,
  DOC_LAYOUT_LABELS,
  models,
  PPOCR_SYMBOLS,
  SLANET_STRUCTURE_VOCAB,
} from '../src';

type Leaf = { modelPath: string };

/** Every `{ modelPath }` object reachable from `root`, with its key path. */
function leaves(root: unknown, path: string[] = []): [string, Leaf][] {
  if (root === null || typeof root !== 'object') return [];
  if ('modelPath' in root) return [[path.join('.'), root as Leaf]];
  return Object.entries(root).flatMap(([key, value]) => leaves(value, [...path, key]));
}

// Every repository is pinned to its v0.10.0 cut, except PP-DocLayoutV3, whose
// v0.11.0 cut carries the Vulkan export that runs (the v0.10.0 one fails at
// execute).
const URL_SHAPE =
  /^https:\/\/huggingface\.co\/software-mansion\/react-native-executorch-(?:(easy-ocr|pp-ocrv6|paddle-helpers)\/resolve\/v0\.10\.0|(pp-doclayout-v3)\/resolve\/v0\.11\.0)\/[a-z0-9_/]+\.pte$/;

describe('models registry', () => {
  it('points every entry at a .pte under its Hugging Face tag', () => {
    const all = leaves(models);
    // 8 EasyOCR alphabets + PP-OCRv6 + PP-DocLayout + PP-Helpers, 3 backends
    // plus DEFAULT each.
    expect(all).toHaveLength(11 * 4);
    const offenders = all
      .filter(([, leaf]) => !URL_SHAPE.test(leaf.modelPath))
      .map(([path, leaf]) => `${path}: ${leaf.modelPath}`);
    expect(offenders).toEqual([]);
  });

  it('names the exports by backend and precision', () => {
    const tail = (leaf: Leaf) => leaf.modelPath.split('/').slice(-3).join('/');
    expect(tail(models.ocr.EASYOCR.ENGLISH.XNNPACK)).toBe(
      'english/xnnpack/easy_ocr_english_xnnpack_int8.pte'
    );
    expect(tail(models.ocr.EASYOCR.ENGLISH.COREML)).toBe(
      'english/coreml/easy_ocr_english_coreml_int8.pte'
    );
    expect(tail(models.ocr.EASYOCR.ENGLISH.VULKAN)).toBe(
      'english/vulkan/easy_ocr_english_vulkan_fp16.pte'
    );
    expect(tail(models.ocr.EASYOCR.ZH_SIM.XNNPACK)).toBe(
      'zh_sim/xnnpack/easy_ocr_zh_sim_xnnpack_int8.pte'
    );
    expect(tail(models.ocr.PADDLE.PPOCRV6_SMALL.XNNPACK)).toBe(
      'v0.10.0/xnnpack/pp_ocrv6_xnnpack_fp32.pte'
    );
    expect(tail(models.ocr.PADDLE.PPOCRV6_SMALL.COREML)).toBe(
      'v0.10.0/coreml/pp_ocrv6_coreml_int8.pte'
    );
    expect(tail(models.ocr.PADDLE.PPOCRV6_SMALL.VULKAN)).toBe(
      'v0.10.0/vulkan/pp_ocrv6_vulkan_fp16.pte'
    );
    expect(tail(models.layoutDetection.PP_DOCLAYOUT.XNNPACK)).toBe(
      'v0.11.0/xnnpack/pp_doclayout_v3_xnnpack_fp32.pte'
    );
    expect(tail(models.layoutDetection.PP_DOCLAYOUT.COREML)).toBe(
      'v0.11.0/coreml/pp_doclayout_v3_coreml_fp16.pte'
    );
    expect(tail(models.documentModels.PP_HELPERS.XNNPACK)).toBe(
      'v0.10.0/xnnpack/paddle_helpers_xnnpack_int8.pte'
    );
    expect(tail(models.documentModels.PP_HELPERS.VULKAN)).toBe(
      'v0.10.0/vulkan/paddle_helpers_vulkan_fp16.pte'
    );
  });

  it('resolves DEFAULT to a member of each group (Core ML on an iOS device under Jest)', () => {
    for (const language of Object.values(models.ocr.EASYOCR)) {
      expect(language.DEFAULT).toBe(language.COREML);
    }
    expect(models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT).toBe(models.ocr.PADDLE.PPOCRV6_SMALL.COREML);
    expect(models.layoutDetection.PP_DOCLAYOUT.DEFAULT).toBe(
      models.layoutDetection.PP_DOCLAYOUT.COREML
    );
    expect(models.documentModels.PP_HELPERS.DEFAULT).toBe(models.documentModels.PP_HELPERS.COREML);
  });

  it('wires the EasyOCR options with the matching alphabet', () => {
    const english = models.ocr.EASYOCR.ENGLISH.XNNPACK.modelOpts;
    expect(english.extractBoxes).toBe(craftExtractBoxes);
    expect(english.recognizerPadMode).toBe('cornerMean');
    expect(english.charset).toBe(alphabets.english);
    expect(english.detectorNorm).toBeDefined();
    expect(models.ocr.EASYOCR.KOREAN.VULKAN.modelOpts.charset).toBe(alphabets.korean);
  });

  it('wires the PP-OCRv6 options', () => {
    const paddle = models.ocr.PADDLE.PPOCRV6_SMALL.XNNPACK.modelOpts;
    expect(paddle.extractBoxes).toBe(dbnetExtractBoxes);
    expect(paddle.minConfidence).toBe(0.5);
    expect(paddle.charset).toBe(PPOCR_SYMBOLS);
  });

  it('wires the layout detector and the document helpers', () => {
    const layout = models.layoutDetection.PP_DOCLAYOUT.XNNPACK.modelOpts;
    expect(layout.labels).toBe(DOC_LAYOUT_LABELS);
    expect(layout.boxFormat).toBe('xyxy');
    expect(layout.resizeMode).toBe('stretch');
    expect(layout.defaultConfidenceThreshold).toBe(0.3);
    expect(layout.defaultIouThreshold).toBe(1.0);

    const helpers = models.documentModels.PP_HELPERS.XNNPACK;
    expect(helpers.table).toEqual({
      structureVocab: SLANET_STRUCTURE_VOCAB,
      eosTokenId: 49,
      maxSteps: 501,
    });
  });
});

describe('charsets and label maps', () => {
  it('ship the expected class counts', () => {
    const count = (s: string) => Array.from(s).length;
    expect(count(alphabets.english)).toBe(96);
    expect(count(alphabets.cyrillic)).toBe(207);
    expect(count(alphabets.latin)).toBe(351);
    expect(count(alphabets.japanese)).toBe(2214);
    expect(count(alphabets.zh_sim)).toBe(6718);
    expect(count(alphabets.korean)).toBe(1008);
    expect(count(alphabets.telugu)).toBe(165);
    expect(count(alphabets.kannada)).toBe(167);
    expect(count(PPOCR_SYMBOLS)).toBe(18709);
    expect(DOC_LAYOUT_LABELS).toHaveLength(25);
    expect(SLANET_STRUCTURE_VOCAB).toHaveLength(50);
    expect(SLANET_STRUCTURE_VOCAB[0]).toBe('sos');
    expect(SLANET_STRUCTURE_VOCAB[49]).toBe('eos');
  });
});
