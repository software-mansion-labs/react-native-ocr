/**
 * Registry of the models the pipeline is pre-configured for: the Hugging Face
 * URL of every export plus its baseline options, so an app can load a model
 * without manual setup.
 *
 * A model that ships several exports lists them under backend-tagged keys and
 * wraps the group in `variants`, which adds the `DEFAULT` alias resolving to
 * the fastest export the current platform can run.
 */
import {
  IMAGENET_NORM,
  models as rneModels,
  type ObjectDetectorModel,
} from 'react-native-executorch';
import { alphabets, PPOCR_SYMBOLS } from './charsets';
import { DOC_LAYOUT_LABELS, SLANET_STRUCTURE_VOCAB, type DocLayoutLabel } from './constants';
import type { DocumentModelsConfig } from './document/types';
import { craftExtractBoxes, dbnetExtractBoxes } from './ocr/detectors';
import type { OcrModel, OcrModelOptions } from './ocr/types';
import { variants } from './registry/variants';

const BASE_URL = 'https://huggingface.co/software-mansion/react-native-executorch';
const VERSION_TAG = 'resolve/v0.10.0';
// PP-DocLayoutV3 alone has a v0.11.0 cut: its Vulkan export re-serialized with
// one graph output per returned value, which the v0.10.0 one lacks (it failed
// at execute). The other repositories stop at v0.10.0.
const DOCLAYOUT_VERSION_TAG = 'resolve/v0.11.0';

// =============================================================================
// OCR
// =============================================================================
// Both OCR families are exported WITHOUT baked input normalization and expect
// the standard ImageNet RGB norm applied client-side (see IMAGENET_NORM). The
// per-language charset is added by each EasyOCR entry below.
const EASYOCR_OPTS: Omit<OcrModelOptions, 'charset'> = {
  extractBoxes: craftExtractBoxes,
  detectorNorm: IMAGENET_NORM,
  recognizerPadMode: 'cornerMean',
};

const PADDLE_PPOCRV6_OPTS: OcrModelOptions = {
  extractBoxes: dbnetExtractBoxes,
  minConfidence: 0.5,
  charset: PPOCR_SYMBOLS,
  detectorNorm: IMAGENET_NORM,
};

// EasyOCR ships int8 on both CPU and ANE; its Vulkan build is fp16-dominated
// (fp16 CRAFT on the GPU, int8 CRNN on XNNPACK).
const EASYOCR_PRECISION = { xnnpack: 'int8', coreml: 'int8', vulkan: 'fp16' } as const;
// int8 loses whole boxes on PP-DocLayoutV3's RT-DETR head, so it stays unquantized.
const DOCLAYOUT_PRECISION = { xnnpack: 'fp32', coreml: 'fp16', vulkan: 'fp16' } as const;
const HELPERS_PRECISION = { xnnpack: 'int8', coreml: 'int8', vulkan: 'fp16' } as const;

type OcrBackend = keyof typeof EASYOCR_PRECISION;

const makeEasyOcr = (
  lang: string,
  backend: OcrBackend,
  charset: string | readonly string[]
): OcrModel => ({
  modelPath:
    `${BASE_URL}-easy-ocr/${VERSION_TAG}/${lang}/${backend}/` +
    `easy_ocr_${lang}_${backend}_${EASYOCR_PRECISION[backend]}.pte`,
  modelOpts: { ...EASYOCR_OPTS, charset },
});
const easyOcr = (lang: string, charset: string | readonly string[]) =>
  variants({
    XNNPACK: makeEasyOcr(lang, 'xnnpack', charset),
    COREML: makeEasyOcr(lang, 'coreml', charset),
    VULKAN: makeEasyOcr(lang, 'vulkan', charset),
  });

// PP-OCRv6 exports are react-native-executorch's own, so their URLs come from
// its registry. XNNPACK takes the fp32 export: a static-activation int8 detect
// computes garbage above ~960 px at any calibration.
const RNE_PPOCRV6 = rneModels.ocr.PADDLE.PPOCRV6_SMALL;
const ppOcrV6 = ({ modelPath }: { modelPath: string }): OcrModel => ({
  modelPath,
  modelOpts: PADDLE_PPOCRV6_OPTS,
});

// =============================================================================
// Document layout: PP-DocLayoutV3
// =============================================================================
const PP_DOCLAYOUT_OPTS = {
  labels: DOC_LAYOUT_LABELS,
  boxFormat: 'xyxy' as const,
  resizeMode: 'stretch' as const,
  interpolation: 'linear' as const,
  normalizeOpts: { alpha: 1 / 255.0, beta: 0.0 },
  defaultConfidenceThreshold: 0.3,
  defaultIouThreshold: 1.0,
};
const ppDocLayout = (backend: OcrBackend): ObjectDetectorModel<'xyxy', DocLayoutLabel> => ({
  modelPath:
    `${BASE_URL}-pp-doclayout-v3/${DOCLAYOUT_VERSION_TAG}/${backend}/` +
    `pp_doclayout_v3_${backend}_${DOCLAYOUT_PRECISION[backend]}.pte`,
  modelOpts: PP_DOCLAYOUT_OPTS,
});

// =============================================================================
// Document helpers: PaddleHelpers (orientation, dewarp, table structure)
// =============================================================================
const SLANET_TABLE = {
  structureVocab: SLANET_STRUCTURE_VOCAB,
  eosTokenId: SLANET_STRUCTURE_VOCAB.indexOf('eos'),
  maxSteps: 501, // decoder step cap: SLANet's longest table structure
};
const ppHelpers = (backend: OcrBackend): DocumentModelsConfig => ({
  modelPath:
    `${BASE_URL}-paddle-helpers/${VERSION_TAG}/${backend}/` +
    `paddle_helpers_${backend}_${HELPERS_PRECISION[backend]}.pte`,
  table: SLANET_TABLE,
});

/**
 * Registry of pre-configured models.
 *
 * Every group carries `XNNPACK`, `COREML` and `VULKAN` exports plus a `DEFAULT`
 * that resolves to the fastest one the current platform can run: Core ML on an
 * iOS device, XNNPACK on the simulator, Vulkan on Android when the app links
 * that backend, else XNNPACK.
 * @category Models
 */
export const models = {
  /**
   * Optical character recognition: a text detector paired with a text
   * recognizer, exported as one fused program.
   */
  ocr: {
    /**
     * EasyOCR: CRAFT detector plus a CRNN recognizer, one model per alphabet.
     * The charset of each entry is the matching member of `alphabets`.
     */
    EASYOCR: {
      ENGLISH: easyOcr('english', alphabets.english),
      CYRILLIC: easyOcr('cyrillic', alphabets.cyrillic),
      LATIN: easyOcr('latin', alphabets.latin),
      JAPANESE: easyOcr('japanese', alphabets.japanese),
      ZH_SIM: easyOcr('zh_sim', alphabets.zh_sim),
      KOREAN: easyOcr('korean', alphabets.korean),
      TELUGU: easyOcr('telugu', alphabets.telugu),
      KANNADA: easyOcr('kannada', alphabets.kannada),
    },
    /**
     * PaddleOCR PP-OCRv6 small: DBNet detector plus an SVTR recognizer, one
     * model for every language, with `PPOCR_SYMBOLS` as its charset.
     */
    PADDLE: {
      PPOCRV6_SMALL: variants({
        XNNPACK: ppOcrV6(RNE_PPOCRV6.XNNPACK_FP32),
        COREML: ppOcrV6(RNE_PPOCRV6.COREML),
        VULKAN: ppOcrV6(RNE_PPOCRV6.VULKAN),
      }),
    },
  },
  /**
   * Document layout detection, for the `layout` slot of an OCR model.
   */
  layoutDetection: {
    /** PP-DocLayoutV3: 25 region classes, see `DOC_LAYOUT_LABELS`. */
    PP_DOCLAYOUT: variants({
      XNNPACK: ppDocLayout('xnnpack'),
      COREML: ppDocLayout('coreml'),
      VULKAN: ppDocLayout('vulkan'),
    }),
  },
  /**
   * Document helpers, for the `documentModels` slot of an OCR model.
   */
  documentModels: {
    /**
     * PaddleHelpers: page orientation, UVDoc dewarping and SLANet table
     * structure in one fused program.
     */
    PP_HELPERS: variants({
      XNNPACK: ppHelpers('xnnpack'),
      COREML: ppHelpers('coreml'),
      VULKAN: ppHelpers('vulkan'),
    }),
  },
};
