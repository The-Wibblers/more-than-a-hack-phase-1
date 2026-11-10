"use client";

import { useRouter } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState, useTransition } from "react";

import { clampFrame } from "@/lib/match";
import type { FrameData } from "@/lib/types";

export interface OverlayOptions {
  players: boolean;
  labels: boolean;
  ball: boolean;
  pitch: boolean;
}

interface ViewerState {
  data: FrameData | null;
  setData: (d: FrameData) => void;
  hoverSlot: number | null;
  setHoverSlot: (slot: number | null) => void;
  overlay: OverlayOptions;
  setOverlay: (o: OverlayOptions) => void;
  /** Frame we are navigating to (equals data.frame once it has loaded). */
  target: number | null;
  pending: boolean;
  go: (frame: number) => void;
  step: (delta: number) => void;
}

const Ctx = createContext<ViewerState | null>(null);

export function useViewer(): ViewerState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useViewer outside ViewerProvider");
  return v;
}

export function ViewerProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [data, setDataState] = useState<FrameData | null>(null);
  const [hoverSlot, setHoverSlot] = useState<number | null>(null);
  const [overlay, setOverlay] = useState<OverlayOptions>({ players: true, labels: true, ball: true, pitch: true });
  const [target, setTarget] = useState<number | null>(null);
  const [pending, startTransition] = useTransition();
  // Latest requested frame, so held-down arrow keys keep stepping from where we are heading.
  const targetRef = useRef<number | null>(null);

  const setData = useCallback((d: FrameData) => {
    setDataState(d);
    if (targetRef.current === null || targetRef.current === d.frame) {
      targetRef.current = d.frame;
      setTarget(d.frame);
    }
  }, []);

  const go = useCallback(
    (frame: number) => {
      const f = clampFrame(frame);
      targetRef.current = f;
      setTarget(f);
      startTransition(() => router.push(`/frame/${f}`, { scroll: false }));
    },
    [router],
  );

  const step = useCallback(
    (delta: number) => {
      const from = targetRef.current ?? data?.frame;
      if (from !== undefined && from !== null) go(from + delta);
    },
    [data, go],
  );

  // Warm the neighbouring frames so single steps feel instant.
  useEffect(() => {
    if (!data) return;
    router.prefetch(`/frame/${clampFrame(data.frame + 1)}`);
    router.prefetch(`/frame/${clampFrame(data.frame - 1)}`);
  }, [data, router]);

  return (
    <Ctx.Provider value={{ data, setData, hoverSlot, setHoverSlot, overlay, setOverlay, target, pending, go, step }}>
      {children}
    </Ctx.Provider>
  );
}

/** Rendered by each /frame/[n] page: hands that page's server-loaded data to the persistent shell. */
export function FrameBridge({ data }: { data: FrameData }) {
  const { setData } = useViewer();
  useLayoutEffect(() => setData(data), [data, setData]);
  return null;
}
