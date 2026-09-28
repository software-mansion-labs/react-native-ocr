/**
 * Configuration types for the optional document-helper model (page
 * orientation, dewarping and table structure). The implementation follows in
 * `documentModels.ts`; the registry only needs the shapes.
 */

/**
 * SLANet table-structure decoding parameters.
 * @category Types
 */
export type TableConfig = {
  /** The structure token vocabulary, index = token id. */
  readonly structureVocab: readonly string[];
  /** The id of the end-of-sequence token in `structureVocab`. */
  readonly eosTokenId: number;
  /** Cap on autoregressive decoder steps per table. */
  readonly maxSteps: number;
};

/**
 * The document-helper model: one fused PTE that carries page-orientation
 * classification, the UVDoc dewarping grid and, optionally, SLANet table
 * structure. The flags pick which stages run by default; a run can override
 * them.
 * @category Types
 */
export type DocumentModelsConfig = {
  /** Local path or remote URL of the fused helper `.pte`. */
  readonly modelPath: string;
  /** Rotate the page upright before OCR. Default `false`. */
  readonly orientation?: boolean;
  /** Flatten a physically warped page before OCR. Default `false`. */
  readonly dewarp?: boolean;
  /** Apply the rotation only when the classifier is at least this confident. Default 0.85. */
  readonly orientationMinConfidence?: number;
  /**
   * Decline the dewarp when it keeps less than this fraction of the source
   * image variance: a bad grid can collapse a page without clear borders to
   * near-blank. Default 0.5; lower is more permissive.
   */
  readonly dewarpMinVarianceRatio?: number;
  /** Table structure recognition; omit to disable it. */
  readonly table?: TableConfig;
};
