/**
 * Per-format facts about an image's pixels. react-native-executorch keeps its
 * own tables of these internal, so the pipeline carries the two it needs.
 */
import type { ColorConversionCode, ImageFormat } from 'react-native-executorch/cv';

/**
 * The number of channels of each pixel format.
 */
export const FORMAT_CHANNELS: Record<ImageFormat, number> = {
  rgb: 3,
  bgr: 3,
  rgba: 4,
  bgra: 4,
  gray: 1,
};

/**
 * The `cvtColor` code that turns each pixel format into RGB, which is what
 * both the detector and the recognizer take; `null` when it already is RGB.
 */
export const TO_RGB: Record<ImageFormat, ColorConversionCode | null> = {
  rgb: null,
  bgr: 'BGR2RGB',
  rgba: 'RGBA2RGB',
  bgra: 'BGRA2RGB',
  gray: 'GRAY2RGB',
};
