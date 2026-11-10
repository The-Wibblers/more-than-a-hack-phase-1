"use client";

import { PITCH_LINES, PITCH_SPOTS } from "@/lib/geometry";
import { FPS, PITCH_M, TEAM_COLOURS } from "@/lib/match";

import { OWN_TRACK_COLOURS } from "./video-panel";
import { useViewer } from "./viewer-context";

const L = PITCH_M.length;
const W = PITCH_M.width;
const PAD = 5;
const ASSISTANT_COLOUR = "#bef264"; // the officials' fluorescent lime

/** Top-down pitch in metres, oriented like the tactical camera (y = 1 touchline at the bottom). */
export function TacticalPitch() {
  const { data, hoverSlot, setHoverSlot } = useViewer();

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Tactical view (ATD)</h2>
        {data && (
          <div className="flex gap-3 text-xs text-zinc-400">
            <Legend colour={TEAM_COLOURS.H} label={`${data.teams.H} (H)`} />
            <Legend colour={TEAM_COLOURS.A} label={`${data.teams.A} (A)`} />
            {data.ownTracks.map((t) => (
              <Legend key={t.key} colour={OWN_TRACK_COLOURS[t.key]} label={t.label} round />
            ))}
            {data.assistants && <Legend colour={ASSISTANT_COLOUR} label="Assistant" />}
          </div>
        )}
      </div>
      <div className="rounded-lg border border-zinc-800 bg-[#14532d]/40">
        <svg viewBox={`${-PAD} ${-PAD} ${L + 2 * PAD} ${W + 2 * PAD}`} className="block w-full">
          <rect x={0} y={0} width={L} height={W} fill="#166534" fillOpacity={0.35} />
          <g stroke="#ffffff" strokeOpacity={0.5} strokeWidth={0.25} fill="none">
            {PITCH_LINES.map((line, i) => (
              // Rounded to the millimetre: the arcs come from Math.sin/cos, whose last digits differ
              // between the server and the browser, which would fail hydration.
              <polyline key={i} points={line.map(([x, y]) => `${(x * L).toFixed(3)},${(y * W).toFixed(3)}`).join(" ")} />

            ))}
            <rect x={-2} y={W / 2 - 3.66} width={2} height={7.32} />
            <rect x={L} y={W / 2 - 3.66} width={2} height={7.32} />
          </g>
          {PITCH_SPOTS.map(([x, y], i) => (
            <circle key={i} cx={x * L} cy={y * W} r={0.3} fill="#ffffff" fillOpacity={0.6} />
          ))}

          {data && (
            <>
              {/* ball trails: our two ball tracks over the last two seconds */}
              {data.ownTracks.flatMap((t) =>
                t.trail.map(
                  (b) =>
                    b.ago > 0 && (
                      <circle
                        key={`${t.key}${b.ago}`}
                        cx={b.x * L}
                        cy={b.y * W}
                        r={0.25}
                        fill={OWN_TRACK_COLOURS[t.key]}
                        fillOpacity={0.75 - (0.55 * b.ago) / (2 * FPS)}
                      />
                    ),
                ),
              )}

              {data.players.map((p) => {
                const cx = p.x * L;
                const cy = p.y * W;
                const colour = TEAM_COLOURS[p.team];
                const stale = p.speed === null;
                const hot = hoverSlot === p.slot;
                // Heading: where the player would be in 1 s at the current velocity.
                const hx = p.vx !== null && p.vy !== null ? cx + p.vx * L : null;
                const hy = p.vx !== null && p.vy !== null ? cy + (p.vy as number) * W : null;
                const moving = hx !== null && hy !== null && Math.hypot(hx - cx, hy - cy) > 0.4;
                return (
                  <g
                    key={p.slot}
                    onMouseEnter={() => setHoverSlot(p.slot)}
                    onMouseLeave={() => setHoverSlot(null)}
                  >
                    <title>
                      {`Slot ${p.slot} · ${data.teams[p.team]}\n` +
                        `speed ${p.speed === null ? "n/a" : `${p.speed.toFixed(1)} m/s`}\n` +
                        `${p.distToBallM === null ? "" : `${p.distToBallM.toFixed(1)} m from ball`}`}
                    </title>
                    {moving && (
                      <line x1={cx} y1={cy} x2={hx!} y2={hy!} stroke={colour} strokeWidth={0.35} strokeOpacity={0.8} />
                    )}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={hot ? 1.8 : 1.4}
                      fill={stale ? "transparent" : colour}
                      stroke={stale ? colour : hot ? "#ffffff" : "#000000"}
                      strokeWidth={stale ? 0.3 : 0.2}
                      strokeDasharray={stale ? "0.5 0.4" : undefined}
                    />
                    <text
                      x={cx}
                      y={cy + 0.55}
                      textAnchor="middle"
                      fontSize={1.5}
                      fontWeight={700}
                      fill={stale ? colour : "#ffffff"}
                      style={{ pointerEvents: "none" }}
                    >
                      {p.slot}
                    </text>
                  </g>
                );
              })}

              {/* the assistant referees on their touchlines, where the model finds them; a flag drawn when it is up */}
              {data.assistants?.map((a) => {
                const ax = a.x;
                const ay = a.line === "far" ? -1.2 : W + 1.2;
                const up = a.line === "far" ? -1 : 1; // the flag on the outside of the line
                return (
                  <g key={a.line}>
                    <title>{`${a.line} assistant · ${a.x.toFixed(1)} m along the line · score ${a.score.toFixed(2)} · flag ${a.flagUp === null ? "not seen" : a.flagUp ? "up" : "down"}`}</title>
                    <rect x={ax - 0.9} y={ay - 0.9} width={1.8} height={1.8} fill={ASSISTANT_COLOUR} stroke="#000" strokeWidth={0.2} />
                    {a.flagUp && (
                      <>
                        <line x1={ax} y1={ay} x2={ax} y2={ay + up * 2.2} stroke="#fafafa" strokeWidth={0.25} />
                        <rect x={ax} y={ay + up * 2.2 - (up > 0 ? 0 : 1.2)} width={1.6} height={1.2} fill="#facc15" stroke="#dc2626" strokeWidth={0.2} />
                      </>
                    )}
                  </g>
                );
              })}

              {/* our ball now */}
              {data.ownTracks.map(
                (t) =>
                  t.ball && (
                    <circle
                      key={t.key}
                      cx={t.ball.x * L}
                      cy={t.ball.y * W}
                      r={0.9}
                      fill="none"
                      stroke={OWN_TRACK_COLOURS[t.key]}
                      strokeWidth={0.3}
                    />
                  ),
              )}
            </>
          )}
        </svg>
      </div>
      {data && <BallStatus />}
    </section>
  );
}

function Legend({ colour, label, round }: { colour: string; label: string; round?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className={`inline-block h-2.5 w-2.5 ${round ? "rounded-full" : "rounded-sm"}`}
        style={{ background: colour }}
      />
      {label}
    </span>
  );
}

function BallStatus() {
  const { data } = useViewer();
  if (!data) return null;
  const { ownTracks, nearest, teams } = data;
  return (
    <div className="grid grid-cols-2 gap-2 text-sm">
      {ownTracks.map((t) => (
        <div key={t.key} className="rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide" style={{ color: OWN_TRACK_COLOURS[t.key] }}>
            {t.label}
          </div>
          {t.ball ? (
            <div className="font-mono text-zinc-100">
              ({(t.ball.x * L).toFixed(1)}, {(t.ball.y * W).toFixed(1)}) m
            </div>
          ) : (
            <div className="text-amber-300">Not tracked</div>
          )}
        </div>
      ))}
      <div className="col-span-2 rounded-md border border-zinc-800 bg-zinc-900/60 px-3 py-2">
        <div className="text-[11px] uppercase tracking-wide text-zinc-500">Nearest player to Metrica&apos;s ball</div>
        {nearest ? (
          <div className="text-zinc-100">
            <span className="font-semibold" style={{ color: TEAM_COLOURS[nearest.team] }}>
              Slot {nearest.slot}
            </span>{" "}
            · {teams[nearest.team]} · <span className="font-mono">{nearest.distM.toFixed(1)} m</span>
          </div>
        ) : (
          <div className="text-zinc-500">—</div>
        )}
      </div>
    </div>
  );
}
