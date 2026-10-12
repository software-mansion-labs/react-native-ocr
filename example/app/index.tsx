import { useRouter } from 'expo-router';
import { Platform, ScrollView, StyleSheet, View } from 'react-native';
import { Button } from '../components/Button';
import { commonStyles, theme } from '../theme';

export default function Home() {
  const router = useRouter();
  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <View style={styles.menu}>
        <Button title="PaddleOCR parity" onPress={() => router.push('/ocr')} />
        <Button title="EasyOCR + vertical text" onPress={() => router.push('/easyocr')} />
        <Button title="Layout (PP-DocLayout)" onPress={() => router.push('/layout')} />
        <Button title="Document helpers" onPress={() => router.push('/document')} />
        {Platform.OS === 'android' && (
          <Button
            title="Vulkan diagnostics"
            variant="secondary"
            onPress={() => router.push('/vulkan')}
          />
        )}
        <Button title="Native spike" variant="secondary" onPress={() => router.push('/spike')} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  // Stretches every button to the same full width and spaces them evenly.
  menu: {
    width: '100%',
    gap: theme.spacing.medium,
  },
});
