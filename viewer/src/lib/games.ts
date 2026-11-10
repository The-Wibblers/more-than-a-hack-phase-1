// The open datasets beyond the challenge match (external/ in the repo root writes them).
// Every game is the same bundle whatever its source, so these types cover all of them.
// Coordinates are metres from the pitch's top-left corner, y down, like the tactical view.

export type GameTeam = "H" | "A" | "R" | "ball";

export interface GameObject {
  key: string;
  team: GameTeam;
  jersey: number | null;
  name: string | null;
  position: string | null;
}

export interface GamePeriod {
  id: number;
  start: number;
  end: number; // inclusive
  clock_start_s: number;
}

export interface DatasetInfo {
  name: string;
  provider: string;
  licence: string;
  host: string;
  url: string;
  paper?: string;
  games: number;
  contents: string;
  video: boolean;
  size: string;
}

export interface GameHas {
  tracking: boolean;
  ball: boolean;
  ball_z: boolean;
  alive: boolean;
  possession: boolean;
  events: boolean;
  video: boolean;
  boxes?: boolean;
}

export interface GameSummary {
  id: string;
  dataset: string;
  title: string;
  competition: string | null;
  date: string | null;
  score: string | null;
  fps: number;
  n_frames: number;
  has: GameHas;
}

export interface GameIndex {
  datasets: Record<string, DatasetInfo>;
  games: GameSummary[];
}

export interface GameMeta extends GameSummary {
  source_id: string;
  home: string;
  away: string;
  pitch: [number, number];
  periods: GamePeriod[];
  objects: GameObject[];
  notes: string;
  /** seconds a speed is measured over, where positions are too coarse for the default 0.2 s */
  speed_window_s?: number;
  dataset_info: DatasetInfo;
  video: {
    width: number;
    height: number;
    /**
     * seek_offset_s: where the video stream starts relative to the file, added to every seek.
     * frames_dir: path is a folder of JPEGs, frame k = {k+1:06d}.jpg (SoccerNet clips).
     */
    files: { period: number; start: number; frames: number; path: string; seek_offset_s?: number; frames_dir?: boolean }[];
  } | null;
}

export interface GameEvent {
  f: number;
  f_end?: number | null;
  period: number;
  type: string;
  sub: string | null;
  team: "H" | "A" | null;
  obj: number | null;
  player: string | null;
  x: number | null;
  y: number | null;
  ex: number | null;
  ey: number | null;
}

export interface GamePlayer {
  obj: number;
  team: GameTeam;
  jersey: number | null;
  name: string | null;
  position: string | null;
  x: number;
  y: number;
  /** metres per second, over the last 0.2 s (or the game's speed_window_s); null at the start of a track */
  vx: number | null;
  vy: number | null;
  speed: number | null;
}

export interface GameFrame {
  frame: number;
  period: number;
  ball: { x: number; y: number; z: number | null } | null;
  ballTrail: { x: number; y: number; ago: number }[];
  players: GamePlayer[];
  alive: boolean | null;
  owner: "H" | "A" | null;
  /** player boxes in video pixels, where the dataset has them */
  boxes: { obj: number; x: number; y: number; w: number; h: number }[];
}

/** Games that are not bundles: the challenge match has its own, richer viewer. */
export const CHALLENGE_GAME = {
  id: "BAR-ATH",
  title: "Barcelona v Athletic Club",
  competition: "LaLiga 2024/25 · the challenge match",
  href: "/frame/11185",
};

const PERIOD_LABELS: Record<number, string> = { 1: "1st half", 2: "2nd half", 3: "Extra time 1", 4: "Extra time 2", 5: "Penalties" };
export const periodLabel = (id: number) => PERIOD_LABELS[id] ?? `Period ${id}`;

export function periodAt(meta: GameMeta, frame: number): GamePeriod {
  return meta.periods.find((p) => frame >= p.start && frame <= p.end) ?? meta.periods[meta.periods.length - 1];
}

/** Match clock, mm:ss, counted from the period's nominal start (45:00 for the 2nd half). */
export function gameClock(meta: GameMeta, frame: number): string {
  const p = periodAt(meta, frame);
  const s = Math.max(0, Math.floor((frame - p.start) / meta.fps + p.clock_start_s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export const GAME_TEAM_COLOURS: Record<GameTeam, string> = { H: "#e5484d", A: "#3b82f6", R: "#facc15", ball: "#fafafa" };

/** Events worth a tick on the whole-match bar. */
export function isKeyEvent(e: GameEvent): boolean {
  return /goal|shot|card/.test(e.type) || /shot|goal/.test(e.sub ?? "");
}
