/**
 * Label maps and pipeline constants that a model's metadata cannot supply.
 */
import type { NormalizeOptions } from 'react-native-executorch/cv';

/**
 * PP-DocLayoutV3 region classes, in model output order (index = class id).
 * @category Constants
 */
export const DOC_LAYOUT_LABELS = [
  'abstract',
  'algorithm',
  'aside_text',
  'chart',
  'content',
  'display_formula',
  'doc_title',
  'figure_title',
  'footer',
  'footer_image',
  'footnote',
  'formula_number',
  'header',
  'header_image',
  'image',
  'inline_formula',
  'number',
  'paragraph_title',
  'reference',
  'reference_content',
  'seal',
  'table',
  'text',
  'vertical_text',
  'vision_footnote',
] as const;

/**
 * A PP-DocLayoutV3 region class.
 * @category Types
 */
export type DocLayoutLabel = (typeof DOC_LAYOUT_LABELS)[number];

/**
 * SLANet_plus table-structure token vocabulary (50 tokens; index = token id).
 * @category Constants
 */
export const SLANET_STRUCTURE_VOCAB = [
  'sos',
  '<thead>',
  '</thead>',
  '<tbody>',
  '</tbody>',
  '<tr>',
  '</tr>',
  '<td',
  '>',
  '</td>',
  ' colspan="2"',
  ' colspan="3"',
  ' colspan="4"',
  ' colspan="5"',
  ' colspan="6"',
  ' colspan="7"',
  ' colspan="8"',
  ' colspan="9"',
  ' colspan="10"',
  ' colspan="11"',
  ' colspan="12"',
  ' colspan="13"',
  ' colspan="14"',
  ' colspan="15"',
  ' colspan="16"',
  ' colspan="17"',
  ' colspan="18"',
  ' colspan="19"',
  ' colspan="20"',
  ' rowspan="2"',
  ' rowspan="3"',
  ' rowspan="4"',
  ' rowspan="5"',
  ' rowspan="6"',
  ' rowspan="7"',
  ' rowspan="8"',
  ' rowspan="9"',
  ' rowspan="10"',
  ' rowspan="11"',
  ' rowspan="12"',
  ' rowspan="13"',
  ' rowspan="14"',
  ' rowspan="15"',
  ' rowspan="16"',
  ' rowspan="17"',
  ' rowspan="18"',
  ' rowspan="19"',
  ' rowspan="20"',
  '<td></td>',
  'eos',
] as const;

/**
 * Default recognizer normalization, `(x / 255 - 0.5) / 0.5`: maps uint8 pixels
 * to `[-1, 1]`, which is what both the CRNN and SVTR recognizers were trained
 * on.
 */
export const RECOGNIZER_NORM: NormalizeOptions = { alpha: 1 / 127.5, beta: -1 };

/**
 * Default fill for the unused width of a recognizer strip: neutral gray.
 */
export const RECOGNIZER_PAD_VALUE = 128;
