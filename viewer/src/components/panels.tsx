"use client";

import { useState } from "react";

import { project } from "@/lib/geometry";
import { BROADCAST_STATUS, FPS, PITCH_M, TEAM_COLOURS } from "@/lib/match";
import { touchSlot } from "@/lib/types";

import { FrameImage, Overlay } from "./video-panel";
import { useViewer } from "./viewer-context";

function Card({ title, children, extra }: { title: string; children: React.ReactNode; extra?: React.ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">{title}</h2>
        {extra}
      </div>
      {children}
    </section>
  );
}

const BODY_TEXT = { foot: "foot", body: "body/chest", head: "head" } as const;
const CONF_TEXT = { break: "", appears: "ball appears at the player", disappears: "ball disappears at the player" } as const;

const NEXT_EVENTS = 4; // upcoming events listed after the most recent one

/** Our own ball tracking: the most recent event (touch, bounce or deflection) and the next few, touches with the ball played after them. */
export function TouchesPanel() {
  const { data, go } = useViewer();
  if (!data) return null;
  const t = data.touches;
  if (!t) {
    return (
      <Card title="Touches">
        <p className="text-sm text-zinc-500">No touch data. Run the own-ball pipeline (ownball.py events).</p>
      </Card>
    );
  }
  const inRange = data.frame >= t.covered.start && data.frame <= t.covered.end;
  const playFrom = new Map(t.plays.map((p) => [p.f0, p]));
  // the most recent event at or before this frame goes at the top, then the next few
  const first = Math.max(0, t.events.findLastIndex((e) => e.f <= data.frame));
  const events = t.events.slice(first, first + 1 + NEXT_EVENTS);
  return (
    <Card title={`Touches · latest + next ${NEXT_EVENTS}`}>
      {!inRange && (
        <p className="text-sm text-zinc-500">
          Touches computed for frames {t.covered.start}–{t.covered.end} only.
        </p>
      )}
      <ul className="flex max-h-[420px] flex-col gap-1 overflow-y-auto pr-1">
        {events.map((e) => {
          const dt = (e.f - data.frame) / FPS;
          const now = Math.abs(dt) < 0.05; // reads as 0.0 s: happening on this frame
          const play = playFrom.get(e.f);
          if (e.kind !== "touch") {
            return (
              <li key={`b${e.f}`}>
                <button
                  className={`flex w-full items-baseline gap-2 rounded px-2 py-0.5 text-left text-xs hover:bg-zinc-800 ${
                    now ? "bg-amber-400/15 text-amber-200 ring-1 ring-amber-400" : "text-zinc-500"
                  }`}
                  onClick={() => go(e.f)}
                >
                  <span className="w-12 shrink-0 font-mono">{dt >= 0 ? "+" : ""}{dt.toFixed(1)}s</span>
                  {e.kind === "bounce" ? "bounce on the ground" : "deflection, nobody in reach"}
                  {e.nearest ? ` (nearest #${e.nearest} ${e.dist_m} m)` : ""}
                </button>
              </li>
            );
          }
          const colour = TEAM_COLOURS[e.team as "H" | "A"];
          return (
            <li
              key={`t${e.f}`}
              className={`rounded-md border-l-4 ${now ? "bg-amber-400/15 ring-1 ring-amber-400" : "bg-zinc-900"}`}
              style={{ borderLeftColor: colour }}
            >
              <button className="flex w-full flex-col px-2 py-1 text-left hover:bg-zinc-800" onClick={() => go(e.f)}>
                <span className="flex items-baseline gap-2 text-sm">
                  <span className={`w-12 shrink-0 font-mono text-xs ${now ? "font-bold text-amber-300" : "text-zinc-400"}`}>{dt >= 0 ? "+" : ""}{dt.toFixed(1)}s</span>
                  <span className="font-semibold" style={{ color: colour }}>
                    {touchSlot(e)} {data.teams[e.team as "H" | "A"]}
                  </span>
                  <span className="text-zinc-300">{BODY_TEXT[e.body ?? "foot"]}</span>
                  {e.conf && e.conf !== "break" && <span className="text-[11px] text-amber-400/80">{CONF_TEXT[e.conf]}</span>}
                  {e.role && e.role !== "control" && (
                    <span
                      className={`rounded px-1.5 text-[11px] font-medium ${
                        e.role === "interception" || e.role === "keeper claim" ? "bg-emerald-900/60 text-emerald-300" : "bg-zinc-800 text-zinc-300"
                      }`}
                      title={
                        e.role === "keeper claim"
                          ? `the ball vanished: ${e.how ?? "with the keeper"}`
                          : e.role === "interception"
                          ? "possession changes here: this team's next touch is theirs again"
                          : "contact only (the other team's next touch is theirs): possession does not change"
                      }
                    >
                      {e.role}
                    </span>
                  )}
                </span>
                {play && (
                  <span className="ml-14 text-xs text-zinc-400">
                    → {play.kind}
                    {play.outcome !== "kept" && (
                      <>
                        {" "}to <span style={{ color: TEAM_COLOURS[play.to_team] }}>#{play.to_slot}</span>{" "}
                        <span className={play.outcome === "lost" ? "text-rose-400" : "text-emerald-400"}>({play.outcome})</span>
                      </>
                    )}
                    {play.length_m !== null && ` · ${play.length_m} m`}
                    {play.bounces ? ` · ${play.bounces} bounce${play.bounces > 1 ? "s" : ""}` : ""}
                  </span>
                )}
              </button>
            </li>
          );
        })}
        {inRange && events.length === 0 && <li className="text-sm text-zinc-500">No touches nearby.</li>}
      </ul>
    </Card>
  );
}

export function PlayersPanel() {
  const { data, hoverSlot, setHoverSlot } = useViewer();
  if (!data) return null;
  const players = [...data.players].sort((a, b) =>
    a.distToBallM !== null && b.distToBallM !== null ? a.distToBallM - b.distToBallM : a.team.localeCompare(b.team) || a.slot - b.slot,
  );
  const count = (side: "H" | "A") => data.players.filter((p) => p.team === side).length;
  return (
    <Card
      title={`Tracked players · ${data.ballInUse ? "by distance to ball" : "by team"}`}
      extra={
        <span className="text-xs text-zinc-500">
          <span style={{ color: TEAM_COLOURS.H }}>{count("H")} H</span> · <span style={{ color: TEAM_COLOURS.A }}>{count("A")} A</span>
        </span>
      }
    >
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-left text-xs tabular-nums">
          <thead className="sticky top-0 bg-zinc-900 text-zinc-500">
            <tr>
              <th className="py-1 pr-2 font-medium">Slot</th>
              <th className="py-1 pr-2 font-medium">Team</th>
              <th className="py-1 pr-2 text-right font-medium">x, y (m)</th>
              <th className="py-1 pr-2 text-right font-medium">Speed</th>
              <th className="py-1 text-right font-medium">To ball</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p) => (
              <tr
                key={p.slot}
                onMouseEnter={() => setHoverSlot(p.slot)}
                onMouseLeave={() => setHoverSlot(null)}
                className={`border-t border-zinc-800/70 ${hoverSlot === p.slot ? "bg-zinc-800" : ""}`}
              >
                <td className="py-1 pr-2 font-mono font-semibold" style={{ color: TEAM_COLOURS[p.team] }}>
                  {p.slot}
                  {data.nearest?.slot === p.slot && <span className="ml-1 text-amber-300">●</span>}
                </td>
                <td className="py-1 pr-2 text-zinc-300">{p.team}</td>
                <td className="py-1 pr-2 text-right font-mono text-zinc-400">
                  {(p.x * PITCH_M.length).toFixed(1)}, {(p.y * PITCH_M.width).toFixed(1)}
                </td>
                <td className="py-1 pr-2 text-right font-mono text-zinc-300">
                  {p.speed === null ? <span className="text-zinc-600">stale</span> : `${p.speed.toFixed(1)} m/s`}
                </td>
                <td className="py-1 text-right font-mono text-zinc-300">{p.distToBallM === null ? "—" : `${p.distToBallM.toFixed(1)} m`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-zinc-500">
        Slots are tracker IDs, not shirt numbers; they drop out and swap identity. Dashed/hollow = no speed reported
        (usually a frozen placeholder). ● = nearest to the ball.
        {data.teamSource === "inferred" && " Slot→team split inferred from the data (ATD metadata XML not found)."}
      </p>
    </Card>
  );
}

export function HomographyPanel() {
  const { data } = useViewer();
  if (!data) return null;
  const { m, reprojectionError, valid } = data.homography;
  const corners = valid
    ? ([
        ["(0,0)", 0, 0],
        ["(1,0)", 1, 0],
        ["(1,1)", 1, 1],
        ["(0,1)", 0, 1],
      ] as const).map(([name, x, y]) => [name, project(m, x, y)] as const)
    : [];
  return (
    <Card
      title="Homography"
      extra={
        <span className={`rounded px-1.5 py-0.5 text-[11px] ${valid ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"}`}>
          {valid ? "valid" : "invalid"}
        </span>
      }
    >
      <div className="text-sm text-zinc-300">
        Reprojection error{" "}
        <span className={`font-mono ${reprojectionError > 1 ? "text-amber-300" : "text-zinc-100"}`}>
          {Number.isFinite(reprojectionError) ? reprojectionError.toFixed(3) : "—"}
        </span>{" "}
        <span className="text-xs text-zinc-500">(lower is better; ~0.24 median in play)</span>
      </div>
      {valid && (
        <>
          <div className="grid grid-cols-3 gap-1 font-mono text-[11px] text-zinc-400">
            {m.map((v, i) => (
              <span key={i} className="rounded bg-zinc-900 px-1.5 py-0.5 text-right">
                {Math.abs(v) < 0.1 && v !== 0 ? v.toExponential(2) : v.toFixed(2)}
              </span>
            ))}
          </div>
          <div className="text-[11px] text-zinc-500">
            Pitch corners → px:{" "}
            {corners.map(([name, p]) => (
              <span key={name} className="mr-2 font-mono">
                {name} {p ? `${p[0].toFixed(0)},${p[1].toFixed(0)}` : "behind camera"}
              </span>
            ))}
          </div>
        </>
      )}
      <p className="text-[11px] text-zinc-500">Maps normalised pitch (0–1) to 1920×1080 pixels; used to draw the overlay on the video.</p>
    </Card>
  );
}

/** Secondary: the broadcast camera at the same instant, for checking what the tactical view shows. */
export function BroadcastPanel() {
  const { data, overlay } = useViewer();
  const [open, setOpen] = useState(true); // open: the zoomed-in view is where the two ball tracks are checked
  if (!data) return null;
  const bc = data.broadcast;
  const status = bc ? BROADCAST_STATUS[bc.status] : null;
  const disregarded = bc?.status === "replay" || bc?.status === "wipe";

  return (
    <Card
      title="Broadcast check"
      extra={
        <button className="flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-100" onClick={() => setOpen(!open)}>
          {status && (
            <span className="rounded px-1.5 py-0.5 text-[11px] font-medium text-white" style={{ background: status.colour }}>
              {status.label}
            </span>
          )}
          {open ? "hide" : "show"}
        </button>
      }
    >
      {!bc ? (
        <p className="text-sm text-zinc-500">
          Broadcast not loaded. Unpack it into data/broadcast/ and run alignment/align_broadcast.py.
        </p>
      ) : (
        <>
          <p className="text-[11px] text-zinc-500">
            {bc.frame === null
              ? "No broadcast footage at this instant (before kickoff, half-time or after the recording)."
              : `Broadcast frame ${bc.frame}, same instant as this tactical frame.`}{" "}
            {disregarded && "A replay shows an earlier moment, so its tracking is ignored."}
          </p>
          {open && bc.frame !== null && (
            <div className="relative aspect-video w-full overflow-hidden rounded-md border border-zinc-800 bg-black">
              <FrameImage frame={bc.frame} source="broadcast" className={disregarded ? "opacity-40 grayscale" : ""} />
              {bc.status === "live" && bc.homography && (
                <Overlay
                  homography={bc.homography}
                  options={overlay}
                  players={bc.players}
                  balls={data.ownTracks.map((t) => ({ ...t, trail: [] }))}
                />
              )}
              {disregarded && (
                <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 bg-black/70 py-2 text-center text-sm font-semibold text-amber-300">
                  {bc.status === "replay" ? "REPLAY · disregarded" : "Replay transition · disregarded"}
                </div>
              )}
              {bc.status === "live" && !bc.homography?.valid && (
                <span className="absolute left-2 top-2 rounded bg-zinc-800/80 px-1.5 py-0.5 text-[11px] text-zinc-300">
                  close-up · no pitch view
                </span>
              )}
            </div>
          )}
        </>
      )}
    </Card>
  );
}
