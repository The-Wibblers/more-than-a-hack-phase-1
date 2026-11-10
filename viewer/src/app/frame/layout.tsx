import { ViewerShell } from "@/components/viewer-shell";
import { ViewerProvider } from "@/components/viewer-context";

// The layout persists across /frame/[n] navigations, so the <video> element (and its loaded
// index of a 5 GB file) survives every step; each page only swaps the frame's data.
export default function FrameLayout({ children }: { children: React.ReactNode }) {
  return (
    <ViewerProvider>
      <ViewerShell />
      {children}
    </ViewerProvider>
  );
}
