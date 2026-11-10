"use client";

import { GAME_TEAM_COLOURS, gameClock, isKeyEvent, periodLabel, type GameEvent, type GameMeta } from "@/lib/games";

const PERIOD_TONES = ["#334155", "#3f3f46"];

/** A stable colour per event type, so the strip and the list agree. */
export function eventColour(type: string): string {
  let h = 0;
  for (let i = 0; i < type.length; i++) h = (h * 31 + type.charCodeAt(i)) % 360;
  return `hsl(${h} 70% 60%)`;
}

/** The whole match: periods, key events (goals, shots, cards), where we are. Click to jump. */
export function GameMatchBar({ meta, events, frame, go }: { meta: GameMeta; events: GameEvent[]; frame: number; go: (f: number) => void }) {
  const n = meta.n_frames;
  const pct = (f: number) => `${(100 * f) / n}%`;
  const key = events.filter(isKeyEvent);
  return (
    <div
      className="relative h-9 cursor-pointer select-none overflow-hidden rounded-md border border-zinc-800 bg-zinc-950"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        go(Math.round(((e.clientX - r.left) / r.width) * (n - 1)));
      }}
    >
      {meta.periods.map((p, i) => (
        <div key={p.id} className="absolute inset-y-0 flex items-end px-1 text-[10px] text-zinc-400" style={{ left: pct(p.start), width: pct(p.end - p.start + 1), background: PERIOD_TONES[i % 2] }}>
          {periodLabel(p.id)}
        </div>
      ))}
      {key.map((e, i) => (
        <div
          key={i}
          title={`${gameClock(meta, e.f)} ${e.type}${e.sub ? ` · ${e.sub}` : ""}${e.player ? ` · ${e.player}` : ""}`}
          className="absolute top-0 h-5 w-[3px] -translate-x-1/2"
          style={{ left: pct(e.f), background: /goal/.test(`${e.type} ${e.sub}`) ? "#fde047" : e.team ? GAME_TEAM_COLOURS[e.team] : "#a1a1aa" }}
          onClick={(ev) => {
            ev.stopPropagation();
            go(e.f);
          }}
        />
      ))}
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white" style={{ left: pct(frame) }} />
    </div>
  );
}

/** ±30 s around the frame: every event as a tick (spans for events with a duration), one row per type. */
export function GameEventStrip({
  meta,
  events,
  frame,
  types,
  go,
}: {
  meta: GameMeta;
  events: GameEvent[];
  frame: number;
  types: string[];
  go: (f: number) => void;
}) {
  const half = 30 * meta.fps;
  const lo = frame - half;
  const hi = frame + half;
  const pct = (f: number) => `${(100 * (Math.min(hi, Math.max(lo, f)) - lo)) / (hi - lo)}%`;
  const near = events.filter((e) => (e.f_end ?? e.f) >= lo && e.f <= hi);
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950 p-2">
      <div className="mb-1 flex justify-between font-mono text-[10px] text-zinc-500">
        <span>−30 s</span>
        <span>{gameClock(meta, frame)}</span>
        <span>+30 s</span>
      </div>
      <div className="relative">
      {types.map((type) => (
        <div key={type} className="relative h-4 border-t border-zinc-900">
          <span className="absolute left-0 top-0 z-10 bg-zinc-950/80 pr-1 text-[10px] text-zinc-500">{type}</span>
          {near
            .filter((e) => e.type === type)
            .map((e, i) => {
              const span = e.f_end && e.f_end > e.f;
              return (
                <button
                  type="button"
                  key={i}
                  title={`${gameClock(meta, e.f)} ${e.type}${e.sub ? ` · ${e.sub}` : ""}${e.player ? ` · ${e.player}` : ""}`}
                  className="absolute top-0.5 h-3 rounded-sm"
                  style={{
                    left: pct(e.f),
                    width: span ? `calc(${pct(e.f_end!)} - ${pct(e.f)})` : "3px",
                    minWidth: 3,
                    background: e.team ? GAME_TEAM_COLOURS[e.team] : eventColour(type),
                    opacity: span ? 0.45 : 1,
                  }}
                  onClick={() => go(e.f)}
                />
              );
            })}
        </div>
      ))}
      <div className="pointer-events-none absolute inset-y-0 left-1/2 w-0.5 bg-white/80" />
      </div>
    </div>
  );
}
