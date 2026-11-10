"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  GAME_TEAM_COLOURS,
  gameClock,
  periodAt,
  periodLabel,
  type GameEvent,
  type GameFrame,
  type GameMeta,
} from "@/lib/games";

import { GamePitch } from "./game-pitch";
import { GameEventStrip, GameMatchBar, eventColour } from "./game-timeline";

const WINDOW_S = 30;

export function GameViewer({ meta, events, initialFrame }: { meta: GameMeta; events: GameEvent[]; initialFrame: number }) {
  const last = meta.n_frames - 1;
  const [frame, setFrame] = useState(initialFrame);
  const [data, setData] = useState<GameFrame | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [selected, setSelected] = useState<GameEvent | null>(null);
  const fps = meta.fps;

  // Phases are spans of play, not events: they get their own stat, not list rows.
  const phases = useMemo(() => events.filter((e) => e.type === "phase"), [events]);
  const pointEvents = useMemo(() => events.filter((e) => e.type !== "phase"), [events]);
  const types = useMemo(() => [...new Set(events.map((e) => e.type))].sort(), [events]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const shown = useMemo(() => pointEvents.filter((e) => !hidden.has(e.type)), [pointEvents, hidden]);

  const go = useCallback((f: number) => setFrame(Math.min(last, Math.max(0, Math.round(f)))), [last]);
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const step = useCallback((d: number) => go(frameRef.current + d), [go]);

  // Latest request wins, so holding an arrow key never shows an older frame after a newer one.
  useEffect(() => {
    const ctl = new AbortController();
    fetch(`/api/games/${meta.id}/frame/${frame}`, { signal: ctl.signal })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setData(d))
      .catch(() => {});
    window.history.replaceState(null, "", `/games/${meta.id}?f=${frame}`);
    return () => ctl.abort();
  }, [meta.id, frame]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA")) return;
      const size = e.altKey ? fps * 10 : e.shiftKey ? fps : 1;
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
  }, [step, fps]);

  const period = periodAt(meta, frame);
  const phase = phases.find((p) => p.f <= frame && frame <= (p.f_end ?? p.f));
  const lastEvent = [...shown].reverse().find((e) => e.f <= frame) ?? null;
  const near = shown.filter((e) => Math.abs(e.f - frame) <= WINDOW_S * fps);
  const teamName = (t: "H" | "A" | null) => (t === "H" ? meta.home : t === "A" ? meta.away : "");
  const highlighted = selected ?? lastEvent;

  return (
    <main className="mx-auto flex max-w-[1800px] flex-col gap-4 px-4 py-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">
            {meta.title}
            {meta.score && <span className="ml-2 font-mono text-zinc-400">{meta.score}</span>}
          </h1>
          <p className="text-xs text-zinc-500">
            {[meta.competition, meta.date].filter(Boolean).join(" · ")} · {meta.dataset_info.name} ({meta.dataset_info.licence}) ·{" "}
            <Link href="/games" className="text-zinc-400 hover:text-zinc-100 hover:underline">
              all games
            </Link>
          </p>
        </div>
      </header>

      <Nav frame={frame} last={last} fps={fps} step={step} go={go} />

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <Stat label="Match clock">
          <span className="font-mono text-2xl font-semibold tabular-nums text-zinc-50">{gameClock(meta, frame)}</span>
          <span className="ml-2 text-xs text-zinc-400">{periodLabel(period.id)}</span>
        </Stat>
        <Stat label="Ball">
          {data?.ball ? (
            <span className="font-mono text-sm text-zinc-100">
              ({data.ball.x.toFixed(1)}, {data.ball.y.toFixed(1)}
              {data.ball.z !== null && meta.has.ball_z ? `, ${data.ball.z.toFixed(1)}` : ""}) m
            </span>
          ) : (
            <span className="text-sm text-zinc-500">{meta.has.ball ? "Not tracked this frame" : "No ball in this dataset"}</span>
          )}
          {data && meta.has.alive && data.alive !== null && (
            <span className={`ml-2 rounded px-1.5 text-xs ${data.alive ? "bg-emerald-700 text-white" : "bg-zinc-700 text-zinc-200"}`}>{data.alive ? "in play" : "dead"}</span>
          )}
        </Stat>
        <Stat label="Possession (provider)">
          {data?.owner ? (
            <span className="rounded-md px-2 py-0.5 text-sm font-semibold text-white" style={{ background: GAME_TEAM_COLOURS[data.owner] }}>
              {teamName(data.owner)}
            </span>
          ) : (
            <span className="text-sm text-zinc-500">{meta.has.possession ? "None" : "Not in this dataset"}</span>
          )}
        </Stat>
        <Stat label={phases.length ? "Phase of play" : "Last event"}>
          {phases.length ? (
            phase ? (
              <span className="text-sm text-zinc-100">
                <span style={{ color: phase.team ? GAME_TEAM_COLOURS[phase.team] : undefined }}>{teamName(phase.team)}</span> · {phase.sub}
              </span>
            ) : (
              <span className="text-sm text-zinc-500">No phase</span>
            )
          ) : lastEvent ? (
            <button type="button" className="text-left text-sm text-zinc-100 hover:underline" onClick={() => go(lastEvent.f)}>
              {lastEvent.type}
              {lastEvent.sub ? ` · ${lastEvent.sub}` : ""} · {((frame - lastEvent.f) / fps).toFixed(1)} s ago
            </button>
          ) : (
            <span className="text-sm text-zinc-500">None yet</span>
          )}
        </Stat>
        <Stat label="In this dataset">
          <div className="flex flex-wrap gap-1">
            {(["tracking", "ball", "events", "possession", "alive", "video"] as const).map((k) => (
              <span key={k} className={`rounded px-1.5 text-[11px] ${meta.has[k] ? "bg-zinc-700 text-zinc-100" : "bg-zinc-900 text-zinc-600 line-through"}`}>
                {k}
              </span>
            ))}
          </div>
        </Stat>
      </div>

      <GameMatchBar meta={meta} events={pointEvents} frame={frame} go={go} />

      <div className={`grid gap-4 ${meta.has.video ? "xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" : "xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]"}`}>
        {meta.has.video ? <GameVideo meta={meta} frame={frame} data={data} hover={hover} /> : null}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Tracking</h2>
            <div className="flex gap-3 text-xs text-zinc-400">
              <Legend colour={GAME_TEAM_COLOURS.H} label={meta.home} />
              <Legend colour={GAME_TEAM_COLOURS.A} label={meta.away} />
              {meta.objects.some((o) => o.team === "R") && <Legend colour={GAME_TEAM_COLOURS.R} label="Officials" />}
            </div>
          </div>
          <GamePitch meta={meta} frame={data} event={highlighted} hover={hover} setHover={setHover} />
          <p className="text-xs text-zinc-500">{meta.notes}</p>
        </div>
        {!meta.has.video && <PlayersTable meta={meta} data={data} hover={hover} setHover={setHover} />}
      </div>

      {types.length > 0 && (
        <section className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <h2 className="mr-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">Events ±{WINDOW_S} s</h2>
            {types
              .filter((t) => t !== "phase")
              .map((t) => (
                <button
                  type="button"
                  key={t}
                  onClick={() => setHidden((h) => (h.has(t) ? new Set([...h].filter((x) => x !== t)) : new Set([...h, t])))}
                  className={`rounded-full border px-2 py-0.5 text-[11px] ${hidden.has(t) ? "border-zinc-800 text-zinc-600" : "border-zinc-600 text-zinc-200"}`}
                >
                  <span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: eventColour(t) }} />
                  {t} <span className="text-zinc-500">{pointEvents.filter((e) => e.type === t).length}</span>
                </button>
              ))}
          </div>
          <GameEventStrip
            meta={meta}
            events={events}
            frame={frame}
            types={types.filter((t) => !hidden.has(t) && events.some((e) => e.type === t && (e.f_end ?? e.f) >= frame - WINDOW_S * fps && e.f <= frame + WINDOW_S * fps))}
            go={go}
          />
          <div className={`grid gap-4 ${meta.has.video ? "lg:grid-cols-2" : ""}`}>
            <EventList meta={meta} events={near} frame={frame} go={go} selected={selected} setSelected={setSelected} />
            {meta.has.video && <PlayersTable meta={meta} data={data} hover={hover} setHover={setHover} />}
          </div>
        </section>
      )}

      <footer className="text-xs text-zinc-500">
        {meta.dataset_info.provider} ·{" "}
        <a className="hover:text-zinc-200 hover:underline" href={meta.dataset_info.url} target="_blank" rel="noreferrer">
          {meta.dataset_info.host}
        </a>{" "}
        · {meta.dataset_info.licence} · source match id {meta.source_id}
      </footer>
    </main>
  );
}

function Nav({ frame, last, fps, step, go }: { frame: number; last: number; fps: number; step: (d: number) => void; go: (f: number) => void }) {
  const [input, setInput] = useState(String(frame));
  useEffect(() => setInput(String(frame)), [frame]);
  const btn = "rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800 disabled:opacity-40";
  const steps = [
    { d: -fps * 10, l: "−10s" },
    { d: -fps, l: "−1s" },
    { d: -1, l: "◀ Prev" },
    { d: 1, l: "Next ▶" },
    { d: fps, l: "+1s" },
    { d: fps * 10, l: "+10s" },
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((s) => (
        <button key={s.l} type="button" className={btn} onClick={() => step(s.d)} disabled={(s.d < 0 && frame <= 0) || (s.d > 0 && frame >= last)}>
          {s.l}
        </button>
      ))}
      <form
        className="flex items-center gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          if (input) go(Number(input));
        }}
      >
        <input
          aria-label="Frame"
          className="w-24 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100 tabular-nums"
          value={input}
          inputMode="numeric"
          onChange={(e) => setInput(e.target.value.replace(/[^\d]/g, ""))}
        />
        <button className={btn} type="submit">
          Go
        </button>
      </form>
      <span className="text-xs text-zinc-500">
        frame {frame} / {last} · {fps} fps · ← → frame · Shift 1 s · Option/Alt 10 s
      </span>
    </div>
  );
}

function GameVideo({ meta, frame, data, hover }: { meta: GameMeta; frame: number; data: GameFrame | null; hover: number | null }) {
  // Off by default: they are the annotated pitch positions projected back into the image, not
  // detections, and land up to a player's width (sometimes ~1 s of movement) away from them.
  const [boxes, setBoxes] = useState(false);
  const [failed, setFailed] = useState<number | null>(null);
  const v = meta.video!;
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Video</h2>
        {meta.has.boxes && (
          <label className="flex items-center gap-1 text-xs text-zinc-400">
            <input type="checkbox" checked={boxes} onChange={(e) => setBoxes(e.target.checked)} /> projected positions (approximate)
          </label>
        )}
      </div>
      <div className="relative overflow-hidden rounded-lg border border-zinc-800 bg-black" style={{ aspectRatio: `${v.width} / ${v.height}` }}>
        {failed === frame ? (
          <div className="grid h-full place-items-center text-sm text-zinc-500">No video at this frame (is ffmpeg installed and the video downloaded?)</div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element -- a fresh JPEG per frame from our own API
          <img src={`/api/games/${meta.id}/video/${frame}`} alt={`Frame ${frame}`} className="absolute inset-0 h-full w-full" onError={() => setFailed(frame)} />
        )}
        {boxes && data && data.frame === frame && (
          <svg viewBox={`0 0 ${v.width} ${v.height}`} className="pointer-events-none absolute inset-0 h-full w-full">
            {data.boxes.map((b) => {
              const team = meta.objects[b.obj].team;
              return (
                <rect key={b.obj} x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={GAME_TEAM_COLOURS[team]} strokeWidth={hover === b.obj ? 6 : 3} />
              );
            })}
          </svg>
        )}
      </div>
    </section>
  );
}

function PlayersTable({ meta, data, hover, setHover }: { meta: GameMeta; data: GameFrame | null; hover: number | null; setHover: (o: number | null) => void }) {
  const rows = useMemo(() => {
    if (!data) return [];
    const b = data.ball;
    return data.players
      .map((p) => ({ ...p, dist: b ? Math.hypot(p.x - b.x, p.y - b.y) : null }))
      .sort((a, z) => (a.dist ?? 1e9) - (z.dist ?? 1e9) || a.team.localeCompare(z.team));
  }, [data]);
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Players by distance to the ball ({rows.length})</h2>
      <div className="max-h-[32rem] overflow-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-zinc-900 text-[11px] uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-2 py-1 text-left">Player</th>
              <th className="px-2 py-1 text-right">m/s</th>
              <th className="px-2 py-1 text-right">to ball</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.obj} className={hover === p.obj ? "bg-zinc-800" : ""} onMouseEnter={() => setHover(p.obj)} onMouseLeave={() => setHover(null)}>
                <td className="px-2 py-0.5">
                  <span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: GAME_TEAM_COLOURS[p.team] }} />
                  {p.jersey !== null ? `#${p.jersey} ` : ""}
                  <span className="text-zinc-300">{p.name ?? (p.team === "R" ? "official" : meta.objects[p.obj].key)}</span>
                  {p.position && <span className="ml-1 text-xs text-zinc-500">{p.position}</span>}
                </td>
                <td className="px-2 py-0.5 text-right font-mono tabular-nums text-zinc-400">{p.speed?.toFixed(1) ?? "—"}</td>
                <td className="px-2 py-0.5 text-right font-mono tabular-nums text-zinc-400">{p.dist?.toFixed(1) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EventList({
  meta,
  events,
  frame,
  go,
  selected,
  setSelected,
}: {
  meta: GameMeta;
  events: GameEvent[];
  frame: number;
  go: (f: number) => void;
  selected: GameEvent | null;
  setSelected: (e: GameEvent | null) => void;
}) {
  return (
    <div className="max-h-[32rem] overflow-auto rounded-lg border border-zinc-800">
      {events.length === 0 && <div className="p-3 text-sm text-zinc-500">No events within {WINDOW_S} s.</div>}
      {events.map((e, i) => {
        const past = e.f <= frame;
        return (
          <button
            type="button"
            key={i}
            onClick={() => {
              setSelected(e);
              go(e.f);
            }}
            className={`flex w-full items-center gap-2 border-b border-zinc-900 px-2 py-1 text-left text-sm hover:bg-zinc-800 ${selected === e ? "bg-zinc-800" : ""} ${past ? "text-zinc-200" : "text-zinc-500"}`}
          >
            <span className="w-12 font-mono text-xs tabular-nums text-zinc-500">{gameClock(meta, e.f)}</span>
            <span className="w-14 text-right font-mono text-xs tabular-nums text-zinc-500">{((e.f - frame) / meta.fps).toFixed(1)}s</span>
            <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: e.team ? GAME_TEAM_COLOURS[e.team] : "#71717a" }} />
            <span className="font-medium" style={{ color: eventColour(e.type) }}>
              {e.type}
            </span>
            {e.sub && <span className="text-zinc-400">{e.sub}</span>}
            {e.player && <span className="ml-auto truncate text-zinc-400">{e.player}</span>}
          </button>
        );
      })}
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2">
      <div className="mb-1 text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="flex flex-wrap items-center">{children}</div>
    </div>
  );
}

function Legend({ colour, label }: { colour: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: colour }} />
      {label}
    </span>
  );
}
