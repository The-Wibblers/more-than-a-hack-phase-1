"use client";

import { useEffect, useState } from "react";

import { FPS, LAST_FRAME } from "@/lib/match";

import { useViewer } from "./viewer-context";

const STEPS = [
  { delta: -FPS * 10, label: "−10s" },
  { delta: -FPS, label: "−1s" },
  { delta: -1, label: "◀ Prev" },
  { delta: 1, label: "Next ▶" },
  { delta: FPS, label: "+1s" },
  { delta: FPS * 10, label: "+10s" },
];

export function FrameNav() {
  const { data, target, pending, go, step } = useViewer();
  const current = target ?? data?.frame ?? 0;
  const [input, setInput] = useState(String(current));

  useEffect(() => setInput(String(current)), [current]);

  // ←/→ one frame · Shift 1 s · Alt/Option 10 s
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;
      const size = e.altKey ? FPS * 10 : e.shiftKey ? FPS : 1;
      if (e.key === "ArrowRight" || e.key === ".") {
        e.preventDefault();
        step(size);
      } else if (e.key === "ArrowLeft" || e.key === ",") {
        e.preventDefault();
        step(-size);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  const btn =
    "rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800 active:bg-zinc-700 disabled:opacity-40";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex items-center gap-1">
        {STEPS.map((s) => (
          <button
            key={s.label}
            className={`${btn} ${Math.abs(s.delta) === 1 ? "font-semibold" : ""}`}
            onClick={() => step(s.delta)}
            disabled={(s.delta < 0 && current <= 0) || (s.delta > 0 && current >= LAST_FRAME)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <form
        className="flex items-center gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          const v = Number(input);
          if (Number.isFinite(v)) go(v);
        }}
      >
        <label className="text-xs uppercase tracking-wide text-zinc-500" htmlFor="frame-input">
          Frame
        </label>
        <input
          id="frame-input"
          className="w-24 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100 tabular-nums"
          value={input}
          inputMode="numeric"
          onChange={(e) => setInput(e.target.value.replace(/[^\d]/g, ""))}
        />
        <button className={btn} type="submit">
          Go
        </button>
      </form>
      <span className={`h-2 w-2 rounded-full ${pending ? "animate-pulse bg-amber-400" : "bg-transparent"}`} />
    </div>
  );
}
