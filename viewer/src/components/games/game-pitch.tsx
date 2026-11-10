"use client";

import { useMemo } from "react";

import { GAME_TEAM_COLOURS, type GameEvent, type GameFrame, type GameMeta } from "@/lib/games";
import { pitchMarkings } from "@/lib/geometry";

const PAD = 5;
const BALL_COLOUR = "#a3e635";

/** Top-down pitch in metres: players with 1 s heading arrows, the ball with a 2 s trail, the selected event. */
export function GamePitch({
  meta,
  frame,
  event,
  hover,
  setHover,
}: {
  meta: GameMeta;
  frame: GameFrame | null;
  event: GameEvent | null;
  hover: number | null;
  setHover: (obj: number | null) => void;
}) {
  const [L, W] = meta.pitch;
  const { lines, spots } = useMemo(() => pitchMarkings(L, W), [L, W]);

  return (
    <div className="rounded-lg border border-zinc-800 bg-[#14532d]/40">
      <svg viewBox={`${-PAD} ${-PAD} ${L + 2 * PAD} ${W + 2 * PAD}`} className="block w-full">
        <rect x={0} y={0} width={L} height={W} fill="#166534" fillOpacity={0.35} />
        <g stroke="#ffffff" strokeOpacity={0.5} strokeWidth={0.25} fill="none">
          {lines.map((line, i) => (
            // Rounded to the millimetre so the server and browser agree (hydration).
            <polyline key={i} points={line.map(([x, y]) => `${(x * L).toFixed(3)},${(y * W).toFixed(3)}`).join(" ")} />
          ))}
          <rect x={-2} y={W / 2 - 3.66} width={2} height={7.32} />
          <rect x={L} y={W / 2 - 3.66} width={2} height={7.32} />
        </g>
        {spots.map(([x, y], i) => (
          <circle key={i} cx={x * L} cy={y * W} r={0.3} fill="#ffffff" fillOpacity={0.6} />
        ))}

        {event && event.x !== null && event.y !== null && (
          <g>
            {event.ex !== null && event.ey !== null && (event.ex !== event.x || event.ey !== event.y) && (
              <line x1={event.x} y1={event.y} x2={event.ex} y2={event.ey} stroke="#fde047" strokeWidth={0.35} strokeDasharray="1 0.6" />
            )}
            <circle cx={event.x} cy={event.y} r={2.6} fill="none" stroke="#fde047" strokeWidth={0.35} />
          </g>
        )}

        {frame && (
          <>
            {frame.ballTrail.map((b) => (
              <circle key={b.ago} cx={b.x} cy={b.y} r={0.25} fill={BALL_COLOUR} fillOpacity={0.75 - (0.55 * b.ago) / (2 * meta.fps)} />
            ))}
            {frame.players.map((p) => {
              const colour = GAME_TEAM_COLOURS[p.team];
              const hot = hover === p.obj;
              const moving = p.vx !== null && p.vy !== null && Math.hypot(p.vx, p.vy) > 0.4;
              return (
                <g key={p.obj} onMouseEnter={() => setHover(p.obj)} onMouseLeave={() => setHover(null)}>
                  <title>{`${p.jersey === null ? "" : `#${p.jersey} `}${p.name ?? ""}${p.position ? ` · ${p.position}` : ""}\n${p.speed === null ? "" : `${p.speed.toFixed(1)} m/s`}`}</title>
                  {moving && <line x1={p.x} y1={p.y} x2={p.x + p.vx!} y2={p.y + p.vy!} stroke={colour} strokeWidth={0.35} strokeOpacity={0.8} />}
                  <circle cx={p.x} cy={p.y} r={hot ? 1.8 : p.team === "R" ? 1 : 1.4} fill={colour} stroke={hot ? "#ffffff" : "#000000"} strokeWidth={0.2} />
                  {p.jersey !== null && p.team !== "R" && (
                    <text x={p.x} y={p.y + 0.55} textAnchor="middle" fontSize={1.5} fontWeight={700} fill="#ffffff" style={{ pointerEvents: "none" }}>
                      {p.jersey}
                    </text>
                  )}
                </g>
              );
            })}
            {frame.ball && (
              <circle
                cx={frame.ball.x}
                cy={frame.ball.y}
                r={0.8 + Math.min(1.2, (frame.ball.z ?? 0) / 4)}
                fill={BALL_COLOUR}
                stroke="#000"
                strokeWidth={0.2}
                fillOpacity={frame.alive === false ? 0.45 : 1}
              />
            )}
          </>
        )}
      </svg>
    </div>
  );
}
