// Match constants and helpers shared by server and client. Numbers come from DATA.md / games.py.

export const FPS = 25;
export const N_FRAMES = 199_319;
export const LAST_FRAME = N_FRAMES - 1;
export const PITCH_M = { length: 105.3, width: 68.0 };
export const VIDEO_PX = { width: 1920, height: 1080 };
export const BROADCAST_VIDEO_FRAMES = 164_487;

export const FIRST_HALF_KICKOFF = 11_185;
export const SECOND_HALF_KICKOFF = 113_742;

// Public ground truth window (evaluation/gt/BAR-ATH_possession_gt_public.csv).
export const PUBLIC_GT = { start: 11_185, end: 18_684 };

export type PeriodKey = "pre" | "h1" | "ht" | "h2" | "post";

export interface Period {
  key: PeriodKey;
  label: string;
  short: string;
  start: number;
  end: number; // inclusive
  graded: boolean;
}

export const PERIODS: Period[] = [
  { key: "pre", label: "Before the match", short: "Pre-match", start: 0, end: 11_184, graded: false },
  { key: "h1", label: "1st half", short: "1st half", start: 11_185, end: 90_651, graded: true },
  { key: "ht", label: "Half-time", short: "Half-time", start: 90_652, end: 110_807, graded: false },
  { key: "h2", label: "2nd half", short: "2nd half", start: 110_808, end: 190_197, graded: true },
  { key: "post", label: "After the final whistle", short: "Post-match", start: 190_198, end: LAST_FRAME, graded: false },
];

export function periodOf(frame: number): Period {
  return PERIODS.find((p) => frame >= p.start && frame <= p.end) ?? PERIODS[PERIODS.length - 1];
}

export function clampFrame(frame: number): number {
  return Math.min(LAST_FRAME, Math.max(0, Math.round(frame)));
}

function mmss(totalSeconds: number, withFrames?: number): string {
  const sign = totalSeconds < 0 ? "-" : "";
  const s = Math.abs(totalSeconds);
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const base = `${sign}${m}:${String(sec).padStart(2, "0")}`;
  return withFrames === undefined ? base : `${base}.${String(withFrames).padStart(2, "0")}`;
}

/** Position in the video file, hh:mm:ss.ff. */
export function videoClock(frame: number): string {
  const seconds = Math.floor(frame / FPS);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(frame % FPS).padStart(2, "0")}`;
}

/** Approximate match clock, counted from each half's kickoff. Null outside the halves. */
export function matchClock(frame: number): string | null {
  const period = periodOf(frame);
  if (period.key === "h1") return mmss((frame - FIRST_HALF_KICKOFF) / FPS);
  if (period.key === "h2") {
    const since = (frame - SECOND_HALF_KICKOFF) / FPS;
    return since < 0 ? `${mmss(since)} to kickoff` : mmss(45 * 60 + since);
  }
  return null;
}

export type Side = "H" | "A";
export type Label = "H" | "A" | "D" | "X";

export const TEAM_COLOURS: Record<Side, string> = { H: "#e5484d", A: "#3b82f6" };
export const LABEL_COLOURS: Record<Label, string> = { H: "#e5484d", A: "#3b82f6", D: "#a1a1aa", X: "#3f3f46" };

export type BroadcastStatus = "live" | "replay" | "wipe" | "none";
export const BROADCAST_STATUS: Record<BroadcastStatus, { label: string; colour: string }> = {
  live: { label: "Live", colour: "#059669" },
  replay: { label: "Replay · disregarded", colour: "#d97706" },
  wipe: { label: "Replay transition · disregarded", colour: "#e11d48" },
  none: { label: "No broadcast", colour: "#3f3f46" },
};
