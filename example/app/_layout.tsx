import { Stack } from 'expo-router';
import { ColorPalette } from '../theme';

export default function Layout() {
  return (
    <Stack
      screenOptions={{
        headerTintColor: ColorPalette.primary,
        headerTitleStyle: { color: ColorPalette.primary },
      }}
    >
      <Stack.Screen name="index" options={{ title: 'react-native-ocr' }} />
      <Stack.Screen name="spike/index" options={{ title: 'Native spike' }} />
    </Stack>
  );
}
