import { Stack } from 'expo-router';
import { ColorPalette, theme } from '../theme';

export default function Layout() {
  return (
    <Stack
      screenOptions={{
        headerTintColor: ColorPalette.primary,
        // Same face as the page titles used to have.
        headerTitleStyle: {
          color: ColorPalette.strongPrimary,
          fontSize: theme.typography.title.fontSize,
          fontWeight: theme.typography.title.fontWeight,
        },
        headerTitleAlign: 'center',
        // iOS puts the previous screen's title next to the back arrow; show the arrow alone.
        headerBackButtonDisplayMode: 'minimal',
      }}
    >
      <Stack.Screen name="index" options={{ title: 'react-native-ocr' }} />
      <Stack.Screen name="ocr/index" options={{ title: 'PaddleOCR parity' }} />
      <Stack.Screen name="easyocr/index" options={{ title: 'EasyOCR' }} />
      <Stack.Screen name="layout/index" options={{ title: 'Layout (PP-DocLayout)' }} />
      <Stack.Screen name="document/index" options={{ title: 'Document helpers' }} />
      <Stack.Screen name="vulkan/index" options={{ title: 'Vulkan diagnostics' }} />
      <Stack.Screen name="spike/index" options={{ title: 'Native spike' }} />
    </Stack>
  );
}
