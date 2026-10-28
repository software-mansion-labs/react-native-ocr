import { Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import type { SkImage } from '@shopify/react-native-skia';
import ReactNativeBlobUtil from 'react-native-blob-util';

/**
 * Resolves a `...Url` screen parameter that swaps a registry export for another
 * file: a URL is used as is, a bare file name points into the app's documents
 * directory (a file pushed there with `adb push` or `devicectl device copy to`).
 * @param value - The parameter value, if any.
 * @returns The model path, or `undefined` without a parameter.
 */
export const resolveModelOverride = (value: string | undefined): string | undefined => {
  if (!value) return undefined;
  return /^[a-z][a-z0-9+.-]*:/i.test(value)
    ? value
    : `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/${value}`;
};

/**
 * Converts a Skia image into the raw RGBA/HWC image buffer that
 * react-native-executorch vision tasks accept. Throws if the pixel data cannot
 * be read.
 * @param image - The Skia image to read pixels from.
 * @returns The RGBA/HWC image buffer with its `data`, `width`, `height`,
 * `format`, and `layout`.
 */
export const skImageToBuffer = (image: SkImage) => {
  const pixels = image.readPixels();
  if (!pixels) {
    throw new Error('Failed to read pixels from image');
  }
  if (!(pixels instanceof Uint8Array)) {
    throw new Error('Expected Uint8Array from readPixels');
  }
  return {
    data: pixels,
    width: image.width(),
    height: image.height(),
    format: 'rgba' as const,
    layout: 'hwc' as const,
  };
};

export const getImage = async (
  useCamera: boolean,
  targetWidth = 800
): Promise<string | undefined> => {
  const permissionResult = useCamera
    ? await ImagePicker.requestCameraPermissionsAsync()
    : await ImagePicker.requestMediaLibraryPermissionsAsync();

  if (!permissionResult.granted) {
    Alert.alert(
      'Permission Required',
      useCamera
        ? 'Permission to access camera is required!'
        : 'Permission to access camera roll is required!'
    );
    return;
  }

  const options: ImagePicker.ImagePickerOptions = {
    mediaTypes: ['images'],
    allowsEditing: false,
    quality: 1,
  };

  const pickerResult = useCamera
    ? await ImagePicker.launchCameraAsync(options)
    : await ImagePicker.launchImageLibraryAsync(options);

  if (pickerResult.canceled || !pickerResult.assets[0]) {
    return;
  }

  const asset = pickerResult.assets[0];
  let imageRef = await ImageManipulator.manipulate(asset.uri).renderAsync();

  if (imageRef.width > targetWidth) {
    imageRef = await ImageManipulator.manipulate(asset.uri)
      .resize({ width: targetWidth })
      .renderAsync();
  }

  const result = await imageRef.saveAsync({ format: SaveFormat.PNG });
  return result.uri;
};
