"use client";

import { useEffect, useMemo, useState } from "react";

import { PITCH_LINES, PITCH_SPOTS, project, type Pt } from "@/lib/geometry";
import { BROADCAST_VIDEO_FRAMES, FPS, LAST_FRAME, TEAM_COLOURS, VIDEO_PX } from "@/lib/match";
import type { BallSource, HomographyData, OwnTrack, OwnTrackKey } from "@/lib/types";
import type { Side } from "@/lib/match";

import { useViewer, type OverlayOptions } from "./viewer-context";

type Source = "tactical" | "broadcast";
const LAST: Record<Source, number> = { tactical: LAST_FRAME, broadcast: BROADCAST_VIDEO_FRAMES - 1 };
export const frameImageUrl = (n: number, source: Source = "tactical") =>
  source === "tactical" ? `/api/frame-image/${n}` : `/api/frame-image/${n}?source=${source}`;

// Ball markers: Metrica's ball in the tactical ATD and in the broadcast ATD
export type MarkerSource = BallSource;
export const TACTICAL_BALL_COLOUR = "#22d3ee";
export const BROADCAST_BALL_COLOUR = "#f472b6";
export const BALL_COLOURS: Record<MarkerSource, string> = { tactical: TACTICAL_BALL_COLOUR, broadcast: BROADCAST_BALL_COLOUR };
export const BALL_LABELS: Record<MarkerSource, string> = { tactical: "Tactical ball", broadcast: "Broadcast ball" };
// Our own ball track: a ring round the ball
export const OWN_TRACK_COLOURS: Record<OwnTrackKey, string> = { ball: "#fb923c" };
const OWN_TRACK_RING: Record<OwnTrackKey, number> = { ball: 12 };

export function VideoPanel() {
  const { data, overlay, setOverlay, hoverSlot, setHoverSlot } = useViewer();
  const frame = data?.frame;

  return (
    <section className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Frame + homography overlay</h2>
          {frame !== undefined && (
            <a className="text-[11px] text-zinc-500 hover:text-zinc-200 hover:underline" href={frameImageUrl(frame)} download>
              save jpg
            </a>
          )}
        </div>
        <OverlayToggles overlay={overlay} setOverlay={setOverlay} />
      </div>
      <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-zinc-800 bg-black">
        <FrameImage frame={frame} />
        {data && (
          <Overlay
            homography={data.homography}
            options={overlay}
            players={data.players.map((p) => ({ slot: p.slot, team: p.team, x: p.x, y: p.y, stale: p.speed === null }))}
            balls={data.ownTracks}
            hoverSlot={hoverSlot}
            onHover={setHoverSlot}
          />
        )}
        {data && !data.homography.valid && (
          <span className="pointer-events-none absolute left-2 top-2 rounded bg-red-600/80 px-1.5 py-0.5 font-mono text-[11px] text-white">
            no homography
          </span>
        )}
      </div>
    </section>
  );
}

/** Server-extracted JPEG. Keeps the previous frame on screen until the next one has decoded. */
export function FrameImage({ frame, source = "tactical", className = "" }: { frame: number | undefined | null; source?: Source; className?: string }) {
  const [shown, setShown] = useState<number | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (frame === undefined || frame === null) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      setError(false);
      setShown(frame);
      // Warm the neighbours so single steps are instant.
      for (const n of [frame + 1, frame - 1]) if (n >= 0 && n <= LAST[source]) new Image().src = frameImageUrl(n, source);
    };
    img.onerror = () => {
      if (!cancelled) setError(true);
    };
    img.src = frameImageUrl(frame, source);
    return () => {
      cancelled = true;
    };
  }, [frame, source]);

  return (
    <>
      {shown !== null && (
        // eslint-disable-next-line @next/next/no-img-element -- a raw frame, not an optimisable asset
        <img
          src={frameImageUrl(shown, source)}
          alt={`${source} video frame ${shown}`}
          className={`absolute inset-0 h-full w-full object-contain ${className}`}
        />
      )}
      {error && (
        <div className="absolute inset-0 grid place-items-center bg-black/80 p-6 text-center text-sm text-zinc-300">
          Could not extract this frame. Check that ffmpeg is installed and the {source} mp4 is in data/{source}/.
        </div>
      )}
    </>
  );
}

function OverlayToggles({ overlay, setOverlay }: { overlay: OverlayOptions; setOverlay: (o: OverlayOptions) => void }) {
  const items: { key: keyof OverlayOptions; label: string }[] = [
    { key: "players", label: "Players" },
    { key: "labels", label: "Slot #" },
    { key: "ball", label: "Ball" },
    { key: "pitch", label: "Pitch lines" },
  ];
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((it) => (
        <button
          key={it.key}
          onClick={() => setOverlay({ ...overlay, [it.key]: !overlay[it.key] })}
          className={`rounded-full border px-2 py-0.5 text-xs ${
            overlay[it.key] ? "border-zinc-500 bg-zinc-700 text-zinc-100" : "border-zinc-800 text-zinc-500"
          }`}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

/** Ball marker: a ring in its track's colour (dashed for "last seen"). */
export function BallMarker({ x, y, size, source, dashed }: { x: number; y: number; size: number; source: MarkerSource; dashed?: boolean }) {
  const colour = BALL_COLOURS[source];
  return (
    <circle cx={x} cy={y} r={size} fill="none" stroke={colour} strokeWidth={size / 4} strokeDasharray={dashed ? `${size / 2.5} ${size / 2.5}` : undefined} />
  );
}

interface OverlayPlayer {
  slot: number;
  team: Side;
  x: number;
  y: number;
  stale?: boolean;
}

export function Overlay({
  homography,
  options,
  players,
  balls,
  hoverSlot = null,
  onHover,
}: {
  homography: HomographyData;
  options: OverlayOptions;
  players: OverlayPlayer[];
  balls: OwnTrack[]; // our ball tracks, on the grass
  hoverSlot?: number | null;
  onHover?: (slot: number | null) => void;
}) {
  const m = homography.m;
  const ok = homography.valid;

  const lines = useMemo(() => {
    if (!ok) return [];
    return PITCH_LINES.map((line) =>
      line
        .map(([x, y]) => project(m, x, y))
        .filter((p): p is Pt => p !== null)
        .map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`)
        .join(" "),
    );
  }, [m, ok]);

  if (!ok) return null;


  return (
    <svg viewBox={`0 0 ${VIDEO_PX.width} ${VIDEO_PX.height}`} preserveAspectRatio="xMidYMid meet" className="absolute inset-0 h-full w-full">
      {options.pitch && (
        <g stroke="#ffffff" strokeOpacity={0.45} strokeWidth={2} fill="none">
          {lines.map((pts, i) => (
            <polyline key={i} points={pts} />
          ))}
          {PITCH_SPOTS.map(([x, y], i) => {
            const p = project(m, x, y);
            return p && <circle key={i} cx={p[0]} cy={p[1]} r={3} fill="#ffffff" fillOpacity={0.5} stroke="none" />;
          })}
        </g>
      )}
      {options.players &&
        players.map((pl) => {
          const p = project(m, pl.x, pl.y);
          if (!p) return null;
          const colour = TEAM_COLOURS[pl.team];
          const hot = hoverSlot === pl.slot;
          return (
            <g key={pl.slot} onMouseEnter={() => onHover?.(pl.slot)} onMouseLeave={() => onHover?.(null)}>
              <ellipse
                cx={p[0]}
                cy={p[1]}
                rx={hot ? 22 : 16}
                ry={hot ? 8 : 6}
                fill={colour}
                fillOpacity={pl.stale ? 0.1 : 0.35}
                stroke={colour}
                strokeWidth={hot ? 4 : 2.5}
                strokeDasharray={pl.stale ? "5 4" : undefined}
              />
              {options.labels && (
                <text
                  x={p[0]}
                  y={p[1] - 12}
                  textAnchor="middle"
                  fontSize={hot ? 22 : 16}
                  fontWeight={700}
                  fill="#ffffff"
                  stroke="#000000"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {pl.slot}
                </text>
              )}
            </g>
          );
        })}
      {options.ball &&
        balls.map((t) => {
          const colour = OWN_TRACK_COLOURS[t.key];
          const now = t.ball ? project(m, t.ball.x, t.ball.y) : null;
          return (
            <g key={t.key}>
              {t.trail.map((b) => {
                const p = b.ago > 0 ? project(m, b.x, b.y) : null;
                return p && <circle key={b.ago} cx={p[0]} cy={p[1]} r={3} fill={colour} fillOpacity={0.7 - (0.5 * b.ago) / (2 * FPS)} />;
              })}
              {now && <circle cx={now[0]} cy={now[1]} r={OWN_TRACK_RING[t.key]} fill="none" stroke={colour} strokeWidth={3} />}
            </g>
          );
        })}
    </svg>
  );
}
