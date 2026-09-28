import { useRouter } from 'expo-router';
import { ScrollView, Text } from 'react-native';
import { Button } from '../components/Button';
import { commonStyles } from '../theme';

export default function Home() {
  const router = useRouter();
  return (
    <ScrollView
      style={commonStyles.container}
      contentContainerStyle={commonStyles.contentContainer}
    >
      <Text style={commonStyles.title}>react-native-ocr</Text>
      <Button title="PaddleOCR parity" onPress={() => router.push('/ocr')} />
      <Button title="Native spike" variant="secondary" onPress={() => router.push('/spike')} />
    </ScrollView>
  );
}
