"use client";

import { BROADCAST_STATUS, FPS, LABEL_COLOURS, N_FRAMES, PERIODS, PUBLIC_GT, TEAM_COLOURS, matchClock, periodOf } from "@/lib/match";
import { touchSlot, type Whistle } from "@/lib/types";

import { useViewer } from "./viewer-context";

const PERIOD_FILL: Record<string, string> = {
  pre: "#27272a",
  h1: "#065f46",
  ht: "#3f3f46",
  h2: "#1e40af",
  post: "#27272a",
};

export const WHISTLE_ROLE_TEXT: Record<Whistle["role"], string> = { stoppage: "Stoppage", restart: "Restart", unclear: "Whistle" };

export function whistleTitle(w: Whistle): string {
  const where = w.kind === "live" ? "" : ` · heard during ${w.kind}, may not be live`;
  return `${WHISTLE_ROLE_TEXT[w.role]} whistle ${matchClock(w.start) ?? `frame ${w.start}`} · ${((w.end - w.start + 1) / FPS).toFixed(2)} s · score ${w.score} dB · ${w.hz} Hz${where}`;
}

/** A referee's whistle, drawn small enough to sit inside a line of text. */
export function WhistleIcon({ className = "h-3.5 w-3.5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M2 9h11l2-3h5" />
      <circle cx="9" cy="14" r="5" />
      <circle cx="9" cy="14" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Whole-match scrubber: periods, the public ground-truth window, and where we are. */
export function MatchBar() {
  const { data, target, go } = useViewer();
  const current = target ?? data?.frame ?? 0;
  const pct = (f: number) => `${(f / N_FRAMES) * 100}%`;

  return (
    <div className="flex flex-col gap-1">
      {data?.whistles && (
        <div className="relative h-2" aria-label="Whistles">
          {data.whistles.map((w) => (
            <button
              key={w.start}
              type="button"
              className="absolute top-0 h-0 w-0 -translate-x-1/2 cursor-pointer border-x-[4px] border-t-[7px] border-x-transparent border-t-zinc-200 hover:border-t-white"
              style={{ left: pct(w.start), opacity: w.kind === "live" ? 1 : 0.35 }}
              title={whistleTitle(w)}
              onClick={() => go(w.start)}
            />
          ))}
        </div>
      )}
      <div
        className="relative h-7 cursor-pointer overflow-hidden rounded-md border border-zinc-800"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          go(((e.clientX - r.left) / r.width) * N_FRAMES);
        }}
        title="Click to jump"
      >
        {PERIODS.map((p) => (
          <div
            key={p.key}
            className="absolute inset-y-0 flex items-center justify-center overflow-hidden text-[10px] font-medium text-white/80"
            style={{ left: pct(p.start), width: pct(p.end - p.start + 1), background: PERIOD_FILL[p.key] }}
          >
            {p.end - p.start > 15000 ? p.short : ""}
          </div>
        ))}
        <div
          className="absolute bottom-0 h-1.5 bg-amber-300"
          style={{ left: pct(PUBLIC_GT.start), width: pct(PUBLIC_GT.end - PUBLIC_GT.start + 1) }}
          title="Public ground truth window"
        />
        <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_4px_white]" style={{ left: pct(current) }} />
      </div>
      <div className="flex justify-between text-[10px] text-zinc-500">
        <span>frame 0</span>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block h-1.5 w-3 bg-amber-300" /> public ground truth
          </span>
          {data?.whistles && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-0 w-0 border-x-[4px] border-t-[7px] border-x-transparent border-t-zinc-200" /> referee
              whistle ({data.whistles.length}, from the broadcast audio)
            </span>
          )}
        </span>
        <span>frame {N_FRAMES - 1}</span>
      </div>
    </div>
  );
}

// What put a stoppage's start where it is.
const TIMED_BY_TEXT = {
  whistle: "whistle",
  "assistant flag": "flag",
  "ball exit": "ball out",
  "speed drop": "players ease off",
  "ball in the goal": "ball in the goal",
  "before the kick-off": "half start",
  "broadcast break": "broadcast break",
  within: "the stoppage before",
  "referee pip": "referee's pip",
  "players stopped": "players stop running",
} as const;
const CAUSE_TEXT: Partial<Record<string, string>> = {
  "set-up": "set-up: ", flag: "flag: ", goal: "goal → ", exit: "untagged ", replay: "replays: ", "restart whistle": "restart whistle: ",
};
const FLAG_COLOUR = "#facc15";

// Our ball's path between breaks: flight (falls at ~gravity in the image), rolling, or neither clearly.
const STRETCH_COLOURS = { flight: "#f59e0b", roll: "#22c55e", unclear: "#52525b" } as const;

/** ±30 s around the current frame: period, ground truth, prediction, ball visibility, touches and balls played. */
export function Timeline() {
  const { data, go } = useViewer();
  if (!data) return null;
  const { start, end } = data.window;
  const span = end - start + 1;
  const pos = (f: number) => ((Math.max(start, Math.min(end + 1, f)) - start) / span) * 100;
  const whistles = (data.whistles ?? []).filter((w) => w.start >= start && w.start <= end);
  const guides = whistles.map((w) => pos(w.start));
  const setPieces = data.tagsWindow.filter((t) => t.code === "SET PIECES" || t.code === "GOALS");

  const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    go(start + ((e.clientX - r.left) / r.width) * span);
  };

  const ticks: number[] = [];
  for (let s = -30; s <= 30; s += 5) ticks.push(data.frame + s * FPS);

  const periodRuns: { start: number; end: number; key: string; label: string }[] = [];
  for (const p of PERIODS) {
    if (p.end >= start && p.start <= end) periodRuns.push({ start: Math.max(p.start, start), end: Math.min(p.end, end), key: p.key, label: p.short });
  }

  return (
    <section className="flex flex-col gap-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Timeline · ±30 s</h2>
      <div className="grid grid-cols-[110px_1fr] gap-x-2 gap-y-1 text-[11px]">
        <div />
        <div className="relative h-4 text-zinc-500">
          {ticks.map((f) => (
            <span key={f} className="absolute -translate-x-1/2 font-mono" style={{ left: `${pos(f)}%` }}>
              {f === data.frame ? "now" : `${f > data.frame ? "+" : ""}${Math.round((f - data.frame) / FPS)}s`}
            </span>
          ))}
        </div>

        <Row label="Period" onClick={onClick} marker={pos(data.frame)} guides={guides}>
          {periodRuns.map((r) => (
            <Bar key={r.key} left={pos(r.start)} width={pos(r.end + 1) - pos(r.start)} colour={PERIOD_FILL[r.key]} text={r.label} />
          ))}
        </Row>
        <Row label="Ground truth" onClick={onClick} marker={pos(data.frame)} guides={guides}>
          {data.gtRuns.map((r) => (
            <Bar
              key={r.start}
              left={pos(r.start)}
              width={pos(r.end + 1) - pos(r.start)}
              colour={LABEL_COLOURS[r.value]}
              text={r.value}
              title={`${r.value} · frames ${r.start}–${r.end}`}
            />
          ))}
        </Row>
        {data.predRuns && (
          <Row label="Our prediction" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.predRuns.map((r) => (
              <Bar
                key={r.start}
                left={pos(r.start)}
                width={pos(r.end + 1) - pos(r.start)}
                colour={LABEL_COLOURS[r.value]}
                text={r.value}
                title={`predicted ${r.value} · frames ${r.start}–${r.end}`}
              />
            ))}
          </Row>
        )}
        {data.oldRuleRuns && (
          <Row label="Old rule" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.oldRuleRuns.map((r) => (
              <Bar
                key={r.start}
                left={pos(r.start)}
                width={pos(r.end + 1) - pos(r.start)}
                colour={LABEL_COLOURS[r.value]}
                text={r.value}
                title={`old rule ${r.value} · frames ${r.start}–${r.end}`}
              />
            ))}
          </Row>
        )}
        {data.stoppages && (
          <Row label="Stoppages" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.stoppages.length === 0 && <span className="absolute inset-0 flex items-center pl-2 text-zinc-600">no stoppage in this window</span>}
            {data.stoppages.map((s) => (
              <Bar
                key={s.start}
                left={pos(s.start)}
                width={pos(s.end + 1) - pos(s.start)}
                colour={LABEL_COLOURS.D}
                text={`${CAUSE_TEXT[s.cause] ?? ""}${s.waits_for}${s.restart_team ? ` (${s.restart_team})` : ""} · from ${TIMED_BY_TEXT[s.timed_by] ?? s.timed_by}`}
                title={`${
                  s.cause === "tag"
                    ? `tagged ${s.waits_for} (tag at frame ${s.tag})`
                    : s.cause === "set-up"
                      ? `untagged ${s.waits_for}: a player ${s.waits_for === "corner" ? "at a corner" : "off the pitch over a touchline"} at frame ${s.tag}`
                      : s.cause === "flag"
                        ? `assistant's flag up (${s.waits_for === "restart" ? "its restart not seen" : `restarted by a ${s.waits_for}`})`
                        : s.cause === "goal"
                          ? "a goal (tagged): dead until the kick-off"
                          : s.cause === "half start"
                            ? "the half's tracking starts before its kick-off"
                            : s.cause === "half-time"
                              ? "half-time: the broadcast's break, to the end of the half"
                              : s.cause === "exit"
                                ? "the ball out over a touchline, nobody tagged it, and the players eased off: a throw-in"
                                : `whistle → ${s.waits_for}`
                }${
                  s.restart_team ? `, taken by ${s.restart_team}` : ""
                }; dead from frame ${s.start}, timed by ${s.timed_by}; ${
                  s.restart ? `restarts at frame ${s.restart}` : s.ended_by === "half-time" ? "the half ends" : "restart not seen: ends when the players resume"
                } (frame ${s.end + 1})`}
                dim={s.ended_by !== "restart"}
              />
            ))}
            {data.stoppages
              .filter((s) => s.restart !== null && s.restart >= start && s.restart <= end)
              .map((s) => (
                <button
                  key={`r${s.start}`}
                  type="button"
                  className="absolute inset-y-0 -translate-x-1/2 rounded-sm bg-emerald-500 px-1 text-[10px] font-semibold text-zinc-950 hover:brightness-110"
                  style={{ left: `${pos(s.restart as number)}%` }}
                  title={`${s.waits_for} taken at frame ${s.restart}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    go(s.restart as number);
                  }}
                >
                  restart
                </button>
              ))}
          </Row>
        )}
        <Row label="Set-piece & goal tags" onClick={onClick} marker={pos(data.frame)} guides={guides}>
          {setPieces.length === 0 && <span className="absolute inset-0 flex items-center pl-2 text-zinc-600">no set piece or goal tagged in this window</span>}
          {setPieces.map((t) => {
            const left = pos(t.start);
            return (
              <button
                key={t.id}
                type="button"
                className={`absolute inset-y-0 whitespace-nowrap rounded-sm px-1 text-[10px] font-semibold text-white hover:brightness-125 ${left > 80 ? "-translate-x-full" : ""}`}
                style={{ left: `${left}%`, background: t.team ? TEAM_COLOURS[t.team] : "#52525b" }}
                title={
                  t.code === "GOALS"
                    ? `tagged goal${t.teamName ? ` · ${t.teamName}` : ""} · frames ${t.start}–${t.end} (from the move's start to the kick-off after it)`
                    : `tagged ${t.labels.Type ?? t.code}${t.teamName ? ` · ${t.teamName}` : ""} · frames ${t.start}–${t.end} (the tag marks the set piece's clip, not the moment play stopped)`
                }
                onClick={(e) => {
                  e.stopPropagation();
                  go(t.start);
                }}
              >
                {t.code === "GOALS" ? "Goal" : (t.labels.Type ?? t.code).replace(" Kick", "").replace(" (After Goal)", "").replace(" (Start)", "")}
              </button>
            );
          })}
        </Row>
        {data.assistantFlags && (
          <Row label="Assistant flag" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.assistantFlags.length === 0 && <span className="absolute inset-0 flex items-center pl-2 text-zinc-600">no flag up in this window</span>}
            {data.assistantFlags.map((f) => (
              <Bar
                key={`${f.line}${f.start}`}
                left={pos(f.start)}
                width={pos(f.end + 1) - pos(f.start)}
                colour={FLAG_COLOUR}
                text={`${f.line} flag`}
                title={`${f.line} assistant's flag up at ${f.x.toFixed(0)} m along the line · frames ${f.start}–${f.end}`}
                dark
              />
            ))}
          </Row>
        )}
        {data.broadcast && (
          <Row label="Broadcast" onClick={onClick} marker={pos(data.frame)} guides={guides} thin>
            {data.broadcast.runs.map((r) => (
              <Bar
                key={r.start}
                left={pos(r.start)}
                width={pos(r.end + 1) - pos(r.start)}
                colour={BROADCAST_STATUS[r.value].colour}
                title={`broadcast ${BROADCAST_STATUS[r.value].label.toLowerCase()} · frames ${r.start}–${r.end}`}
              />
            ))}
          </Row>
        )}
        {data.whistles && (
          <Row label="Referee whistle" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {whistles.length === 0 && <span className="absolute inset-0 flex items-center pl-2 text-zinc-600">no whistle in this window</span>}
            {whistles.map((w) => {
              const left = pos(w.start);
              const flip = left > 80; // keep the label inside the row near the right edge
              return (
                <button
                  key={w.start}
                  type="button"
                  className={`absolute inset-y-0 flex items-center gap-1 whitespace-nowrap rounded-sm border border-zinc-500 bg-zinc-800 px-1.5 text-[10px] font-medium text-zinc-100 hover:border-zinc-300 hover:bg-zinc-700 ${flip ? "-translate-x-full" : ""}`}
                  style={{ left: `${left}%`, opacity: w.kind === "live" ? 1 : 0.5 }}
                  title={whistleTitle(w)}
                  onClick={(e) => {
                    e.stopPropagation();
                    go(w.start);
                  }}
                >
                  <WhistleIcon className="h-3 w-3" />
                  {WHISTLE_ROLE_TEXT[w.role]}
                </button>
              );
            })}
          </Row>
        )}
        {data.touches && (
          <Row label="Touches" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.touches.events.map((e) =>
              e.kind === "touch" ? (
                <button
                  key={e.f}
                  type="button"
                  className="absolute inset-y-0 -translate-x-1/2 rounded-sm px-0.5 text-[10px] font-semibold text-white hover:brightness-125"
                  style={{ left: `${pos(e.f)}%`, background: TEAM_COLOURS[e.team as "H" | "A"], opacity: e.conf === "break" ? 1 : 0.7 }}
                  title={`${touchSlot(e)} ${e.team} · ${e.body} · frame ${e.f}${e.conf !== "break" ? ` · ${e.conf}` : ""}`}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    go(e.f);
                  }}
                >
                  {e.slot ?? "?"}
                </button>
              ) : (
                <span
                  key={e.f}
                  className="absolute top-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-500"
                  style={{ left: `${pos(e.f)}%` }}
                  title={`${e.kind === "bounce" ? "bounce on the ground" : "deflection, nobody in reach"} · frame ${e.f}`}
                />
              ),
            )}
          </Row>
        )}
        {data.touches && (
          <Row label="On the ball" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.touches.spells.map((s) => (
              <Bar
                key={`${s.f0}-${s.slot ?? "?"}`}
                left={pos(s.f0)}
                width={pos(s.f1 + 1) - pos(s.f0)}
                colour={TEAM_COLOURS[s.team]}
                text={s.slot != null ? `#${s.slot}` : "?"}
                title={`${s.slot != null ? `#${s.slot}` : "untracked"} ${s.team} on the ball · frames ${s.f0}–${s.f1}`}
              />
            ))}
          </Row>
        )}
        {data.touches && (
          <Row label="Ball motion" onClick={onClick} marker={pos(data.frame)} guides={guides} thin>
            {data.touches.stretches.map((g) => (
              <Bar
                key={g.f0}
                left={pos(g.f0)}
                width={pos(g.f1 + 1) - pos(g.f0)}
                colour={STRETCH_COLOURS[g.kind]}
                title={`${g.kind} · ${g.g_frac}× gravity downward · frames ${g.f0}–${g.f1}`}
              />
            ))}
          </Row>
        )}
        {data.touches && (
          <Row label="Ball played" onClick={onClick} marker={pos(data.frame)} guides={guides}>
            {data.touches.plays.map((p) => (
              <Bar
                key={`${p.f0}-${p.f1}`}
                left={pos(p.f0)}
                width={pos(p.f1) - pos(p.f0)}
                colour={TEAM_COLOURS[p.from_team]}
                text={p.kind.replace("ground pass", "pass").replace("lofted ball", "lofted")}
                title={`${p.kind} #${p.from_slot} → #${p.to_slot} (${p.outcome})${p.length_m !== null ? ` · ${p.length_m} m` : ""} · frames ${p.f0}–${p.f1}`}
              />
            ))}
          </Row>
        )}
      </div>
    </section>
  );
}

function Row({
  label,
  children,
  onClick,
  marker,
  guides = [],
  thin,
}: {
  label: string;
  children: React.ReactNode;
  onClick: (e: React.MouseEvent<HTMLDivElement>) => void;
  marker: number;
  guides?: number[]; // whistle onsets, as a dashed line through every row
  thin?: boolean;
}) {
  return (
    <>
      <div className={`truncate text-right text-zinc-400 ${thin ? "text-[10px] leading-3 text-zinc-500" : "py-0.5"}`}>{label}</div>
      <div className={`relative cursor-pointer overflow-hidden rounded-sm bg-zinc-900 ${thin ? "h-2.5 self-center" : "h-5"}`} onClick={onClick}>
        {children}
        {guides.map((g) => (
          <div key={g} className="pointer-events-none absolute inset-y-0 border-l border-dashed border-zinc-300/60" style={{ left: `${g}%` }} />
        ))}
        <div className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${marker}%` }} />
      </div>
    </>
  );
}

function Bar({
  left,
  width,
  colour,
  text,
  title,
  dim,
  dark,
}: {
  left: number;
  width: number;
  colour: string;
  text?: string;
  title?: string;
  dim?: boolean;
  dark?: boolean; // dark text, on a light colour
}) {
  return (
    <div
      className={`absolute inset-y-0 overflow-hidden whitespace-nowrap px-1 text-[10px] leading-5 ${dark ? "text-zinc-950" : "text-white"}`}
      style={{ left: `${left}%`, width: `${Math.max(width, 0.15)}%`, background: colour, opacity: dim ? 0.45 : 1 }}
      title={title}
    >
      {text}
    </div>
  );
}

export function periodTone(frame: number): string {
  return PERIOD_FILL[periodOf(frame).key];
}
