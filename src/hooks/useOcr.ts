import { useModel, useResourceDownload, type ResourceOptions } from 'react-native-executorch';
import { createOcr } from '../ocr/createOcr';
import type { OcrModel } from '../ocr/types';

/**
 * React hook for the OCR pipeline (EasyOCR or PaddleOCR). It downloads and
 * loads the model, tracks progress and errors, instantiates the runner, and
 * releases native memory on unmount or config change. Heavy work runs on a
 * worklet thread; `runOcr` resolves with the recognized regions.
 *
 * For imperative usage, see {@link createOcr}.
 * @category Hooks
 * @typeParam L The layout region class-label type (from the `layout` model).
 * @param config The OCR model. Use a preset from `models.ocr.*`.
 * @param options Load and caching options.
 * @returns Readiness flags, download progress, and the `runOcr` and
 * `runOcrWorklet` runners (undefined until the model is ready).
 */
export function useOcr<L = never>(config: OcrModel<L>, options?: ResourceOptions) {
  // Resolves every remote URL in the config in one pass, with progress
  // weighted across all of them.
  const { resource, downloadProgress, downloadError } = useResourceDownload(config, options);
  const { model, error } = useModel(createOcr<L>, resource);

  return {
    isReady: !!model,
    error: downloadError || error,
    downloadProgress,
    resource,
    runOcr: model?.runOcr,
    runOcrWorklet: model?.runOcrWorklet,
  };
}
