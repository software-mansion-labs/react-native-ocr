import { Redirect } from 'expo-router';

// Phase 1: the native spike is the only screen, so open it directly. The
// demo menu returns once the OCR screens land.
export default function Home() {
  return <Redirect href="/spike" />;
}
