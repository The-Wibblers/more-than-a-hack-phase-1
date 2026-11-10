"use client";

import Link from "next/link";
import { FPS, LABEL_COLOURS, PUBLIC_GT, matchClock, videoClock, type Label } from "@/lib/match";
import { touchSlot, type FrameData, type Whistle } from "@/lib/types";

import { FrameNav } from "./frame-nav";
import { BroadcastPanel, HomographyPanel, PlayersPanel, TouchesPanel } from "./panels";
import { TacticalPitch } from "./tactical-pitch";
import { MatchBar, Timeline, WHISTLE_ROLE_TEXT, WhistleIcon, periodTone, whistleTitle } from "./timeline";
import { VideoPanel } from "./video-panel";
import { useViewer } from "./viewer-context";

export function ViewerShell() {
  const { data } = useViewer();

  return (
    <main className="mx-auto flex max-w-[1800px] flex-col gap-4 px-4 py-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">
            Barcelona v Athletic Club <span className="font-normal text-zinc-500">· tactical camera</span>
          </h1>
          <p className="text-xs text-zinc-500">
            ← → frame · Shift 1 s · Option/Alt 10 s ·{" "}
            <Link href="/games" className="text-zinc-400 hover:text-zinc-100 hover:underline">
              all games
            </Link>
          </p>
        </div>
        <FrameNav />
      </header>

      {data ? <StatusStrip data={data} /> : <div className="h-[74px] animate-pulse rounded-lg bg-zinc-900" />}
      <MatchBar />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <VideoPanel />
        <TacticalPitch />
      </div>

      <Timeline />

      <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-4">
        <TouchesPanel />
        <PlayersPanel />
        <HomographyPanel />
        <BroadcastPanel />
      </div>
    </main>
  );
}

const GT_TEXT: Record<Label, string> = {
  H: "Home in possession",
  A: "Away in possession",
  D: "Dead ball",
  X: "No label",
};

function StatusStrip({ data }: { data: FrameData }) {
  const clock = matchClock(data.frame);
  const gtSide = data.gt === "H" ? data.teams.H : data.gt === "A" ? data.teams.A : null;
  const inPublic = data.frame >= PUBLIC_GT.start && data.frame <= PUBLIC_GT.end;

  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
      <Stat label="Match clock">
        {clock ? (
          <span className="font-mono text-2xl font-semibold tabular-nums text-zinc-50">{clock}</span>
        ) : (
          <span className="text-base font-semibold text-zinc-400">Not in play</span>
        )}
        <span className="ml-2 font-mono text-xs text-zinc-500">video {videoClock(data.frame)}</span>
      </Stat>

      <Stat label="Period">
        <span className="rounded-md px-2 py-0.5 text-base font-semibold text-white" style={{ background: periodTone(data.frame) }}>
          {data.period.label}
        </span>
        {data.period.graded ? (
          <span className="ml-2 text-xs text-emerald-400">in play · graded</span>
        ) : (
          <span className="ml-2 text-xs text-zinc-400">not part of the game · ungraded</span>
        )}
      </Stat>

      <Stat label="Ground-truth possession">
        <span
          className="rounded-md px-2 py-0.5 text-base font-semibold"
          style={{ background: data.gt === "X" ? "transparent" : LABEL_COLOURS[data.gt], color: data.gt === "D" ? "#09090b" : "#fff" }}
        >
          {data.gt === "X" ? "—" : data.gt}
        </span>
        <span className="ml-2 text-sm text-zinc-300">
          {data.gt === "X" ? (inPublic ? "No label" : "No public label for this frame") : gtSide ?? GT_TEXT[data.gt]}
        </span>
        {data.gt === "X" && (
          <div className="mt-0.5 text-xs text-zinc-500">
            Public ground truth covers frames {PUBLIC_GT.start}–{PUBLIC_GT.end}
          </div>
        )}
      </Stat>

      <PredictionStat data={data} />

      <WhistleStat data={data} />

      <BallPlayedStat data={data} />
    </div>
  );
}

const WHISTLE_NOW_FRAMES = FPS; // a whistle counts as "now" until 1 s after it ends

/** The whistle sounding now, or the last and next one, from the broadcast audio. */
function WhistleStat({ data }: { data: FrameData }) {
  const { go } = useViewer();
  if (!data.whistles) {
    return (
      <Stat label="Referee whistle">
        <span className="text-sm text-zinc-500">Run alignment/detect_whistles.py</span>
      </Stat>
    );
  }
  const f = data.frame;
  const now = data.whistles.find((w) => w.start <= f && f <= w.end + WHISTLE_NOW_FRAMES);
  const last = [...data.whistles].reverse().find((w) => w.start <= f);
  const next = data.whistles.find((w) => w.start > f);
  const seconds = (w: Whistle) => Math.abs(w.start - f) / FPS;
  const ago = (s: number) => (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

  const jump = (w: Whistle, text: string) => (
    <button type="button" className="text-left text-xs text-zinc-400 hover:text-zinc-100 hover:underline" title={whistleTitle(w)} onClick={() => go(w.start)}>
      {text}
    </button>
  );

  return (
    <Stat label="Referee whistle">
      {now ? (
        <button
          type="button"
          className="flex items-center gap-1.5 rounded-md bg-zinc-100 px-2 py-0.5 text-base font-semibold text-zinc-900"
          title={whistleTitle(now)}
          onClick={() => go(now.start)}
        >
          <WhistleIcon className="h-4 w-4" />
          {WHISTLE_ROLE_TEXT[now.role]}
          <span className="text-xs font-normal text-zinc-600">{f <= now.end ? "now" : `${((f - now.end) / FPS).toFixed(1)} s ago`}</span>
        </button>
      ) : (
        <div className="flex flex-col">
          {last ? jump(last, `Last: ${WHISTLE_ROLE_TEXT[last.role].toLowerCase()} ${matchClock(last.start) ?? ""} · ${ago(seconds(last))} ago`) : <span className="text-xs text-zinc-500">No whistle yet</span>}
          {next && jump(next, `Next: ${WHISTLE_ROLE_TEXT[next.role].toLowerCase()} ${matchClock(next.start) ?? ""} · in ${ago(seconds(next))}`)}
        </div>
      )}
    </Stat>
  );
}

function PredictionStat({ data }: { data: FrameData }) {
  const p = data.pred;
  if (!p) {
    return (
      <Stat label="Our prediction">
        <span className="text-sm text-zinc-500">No prediction file</span>
        <div className="mt-0.5 text-xs text-zinc-500">Run python possession/predict.py</div>
      </Stat>
    );
  }
  const side = p === "H" ? data.teams.H : p === "A" ? data.teams.A : null;
  const graded = p !== "X" && data.gt !== "X";
  return (
    <Stat label="Our prediction">
      <span
        className="rounded-md px-2 py-0.5 text-base font-semibold"
        style={{ background: p === "X" ? "transparent" : LABEL_COLOURS[p], color: p === "D" ? "#09090b" : "#fff" }}
      >
        {p === "X" ? "—" : p}
      </span>
      <span className="ml-2 text-sm text-zinc-300">{p === "X" ? "Not graded (outside the halves)" : side ?? GT_TEXT[p]}</span>
      {graded && (
        <div className={`mt-0.5 text-xs font-medium ${p === data.gt ? "text-emerald-400" : "text-rose-400"}`}>
          {p === data.gt ? "✓ matches the ground truth" : `✗ ground truth says ${data.gt}`}
        </div>
      )}
    </Stat>
  );
}

/** Our own ball tracking: the last touch, and the ball being played now. */
function BallPlayedStat({ data }: { data: FrameData }) {
  const { go } = useViewer();
  const t = data.touches;
  if (!t || data.frame < t.covered.start || data.frame > t.covered.end) {
    return (
      <Stat label="Ball played">
        <span className="text-sm text-zinc-500">{t ? `Touches cover frames ${t.covered.start}–${t.covered.end}` : "No touch data"}</span>
      </Stat>
    );
  }
  const last = t.lastTouch;
  const play = t.play;
  return (
    <Stat label="Ball played">
      {last ? (
        <button className="text-left text-sm text-zinc-200 hover:underline" onClick={() => go(last.f)}>
          Last touch{" "}
          <span className="font-semibold" style={{ color: LABEL_COLOURS[last.team as "H" | "A"] }}>
            {touchSlot(last)} {last.team}
          </span>{" "}
          · {last.body} · {((data.frame - last.f) / FPS).toFixed(1)}s ago
        </button>
      ) : (
        <span className="text-sm text-zinc-500">No touch yet</span>
      )}
      {play && (
        <div className="mt-0.5 text-xs text-zinc-400">
          {play.kind}
          {play.outcome !== "kept" && (
            <>
              {" "}→ #{play.to_slot}{" "}
              <span className={play.outcome === "lost" ? "text-rose-400" : "text-emerald-400"}>{play.outcome}</span>
            </>
          )}
          {play.length_m !== null && ` · ${play.length_m} m`}
        </div>
      )}
    </Stat>
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
