import "server-only";

import fs from "node:fs";
import path from "node:path";

import {
  BROADCAST_VIDEO_FRAMES,
  FIRST_HALF_KICKOFF,
  FPS,
  N_FRAMES,
  PERIODS,
  PITCH_M,
  PUBLIC_GT,
  SECOND_HALF_KICKOFF,
  periodOf,
  type BroadcastStatus,
  type Label,
  type PeriodKey,
  type Side,
} from "../match";
import { unproject } from "../geometry";
import type { Assistant, AssistantFlag, Ball, BallPlay, BallPoint, BallStretch, BroadcastData, FrameData, HomographyData, OwnTrack, OwnTrackKey, Player, Run, Stoppage, Tag, TouchData, TouchEvent, Whistle } from "../types";

// ---------------------------------------------------------------------------------------------
// Paths. The viewer lives in <repo>/viewer, so the repo root is one level up. MTAH_DATA_DIR moves
// the data bundle, the same variable games.py honours.
// ---------------------------------------------------------------------------------------------

const REPO_ROOT = path.resolve(process.cwd(), "..");
const DATA_DIR = process.env.MTAH_DATA_DIR ?? path.join(REPO_ROOT, "data");
export type Source = "tactical" | "broadcast";
const SOURCE_DIR: Record<Source, string> = {
  tactical: path.join(DATA_DIR, "tactical"),
  broadcast: path.join(DATA_DIR, "broadcast"),
};
const STEM: Record<Source, string> = { tactical: "BAR-ATH_tactical", broadcast: "BAR-ATH_broadcast" };
const asset = (source: Source, suffix: string) => path.join(SOURCE_DIR[source], `${STEM[source]}${suffix}`);

export const PATHS = {
  atdRaw: asset("tactical", "_atd.txt"),
  atdMeta: asset("tactical", "_atd.xml"),
  homography: asset("tactical", "_homography.csv"),
  smartTagging: asset("tactical", "_smart_tagging.xml"),
  gtPublic: path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"),
  broadcastAtdRaw: asset("broadcast", "_atd.txt"),
  broadcastAtdMeta: asset("broadcast", "_atd.xml"),
  broadcastHomography: asset("broadcast", "_homography.csv"),
  // written by alignment/align_broadcast.py
  broadcastAlignment: asset("broadcast", "_to_tactical.csv"),
  // written by alignment/detect_whistles.py
  whistles: asset("broadcast", "_whistles.csv"),
  // written by possession/predict.py: our possession label for every frame (touch-based first, then nearest player)
  prediction: [
    path.join(REPO_ROOT, "evaluation", "results", "possession_touches.csv"),
  ],
  // written by our own ball tracker (ownball.py events): pixel ball track, touches, balls played
  touches: path.join(DATA_DIR, "ballcv", "BAR-ATH_touches.json"),
  // written by ballcv/stoppages.py: each stoppage, the restart it waits for, and when that came
  stoppages: path.join(DATA_DIR, "ballcv", "BAR-ATH_stoppages.json"),
  // written by possession/touch_possession.py public: the touch model's team in live play, predict.py's dead ball
  touchModel: path.join(REPO_ROOT, "evaluation", "results", "possession_touch_model.csv"),
  // written by ballcv/viewer_export.py: the assistant model's referees at each flag sample
  assistants: path.join(DATA_DIR, "ballcv", "BAR-ATH_assistants.json"),
  // written by ballcv/viewer_export.py: where each assistant's flag is, and the flag model's raises
  flagSignals: path.join(DATA_DIR, "ballcv", "BAR-ATH_flags.json"),
  // written by ballcv/viewer_export.py: our player tracks (player_track.py), float32 [frame, slot, x | y] 0-1
  playerTracks: path.join(DATA_DIR, "ballcv", "BAR-ATH_players_tracked.f32"),
  playerTracksMeta: path.join(DATA_DIR, "ballcv", "BAR-ATH_players_tracked.json"),
};

const TEAM_NAMES: Record<Side, string> = { H: "Barcelona", A: "Athletic Club" };
// Smart Tagging spells Athletic "Atletic Club"; match on a prefix so either spelling works.
const TAG_TEAM_PREFIX: Record<Side, string> = { H: "barc", A: "at" };

const WINDOW_FRAMES = 30 * FPS; // timeline context either side of the current frame
const TRAIL_FRAMES = 2 * FPS; // ball trail length
const LAST_SEEN_LOOKBACK = 10 * FPS;
const HEADING_LOOKBACK = 5; // frames used to estimate each player's direction of travel

/** A source's video. games.py expects BAR-ATH_<source>.mp4; downloads arrive as -001/-002.mp4. */
export function videoPath(source: Source = "tactical"): string | null {
  const dir = SOURCE_DIR[source];
  const exact = asset(source, ".mp4");
  if (fs.existsSync(exact)) return exact;
  if (!fs.existsSync(dir)) return null;
  const match = fs
    .readdirSync(dir)
    .filter((name) => name.startsWith(STEM[source]) && name.toLowerCase().endsWith(".mp4"))
    .sort()[0];
  return match ? path.join(dir, match) : null;
}

// ---------------------------------------------------------------------------------------------
// ATD: "<frame_id>:<x,y,speed>;...x46:<ball x,y,v>". Kept in memory with a line-offset index so
// any frame (and its neighbours, for trails) is a slice away.
// ---------------------------------------------------------------------------------------------

interface ParsedFrame {
  xs: Float64Array;
  ys: Float64Array;
  speeds: Float64Array;
  ball: { x: number; y: number; v: number } | null;
}

class Atd {
  private buf: Buffer;
  private offsets: Uint32Array; // offsets[i] = byte offset of video frame i; offsets[n] = EOF
  readonly nSlots: number;
  readonly nFrames: number;

  constructor(file: string, nFrames: number) {
    this.buf = fs.readFileSync(file);
    this.nFrames = nFrames;
    const offsets = new Uint32Array(nFrames + 1);
    let line = 0;
    offsets[0] = 0;
    for (let i = 0; i < this.buf.length && line < nFrames; i++) {
      if (this.buf[i] === 10) offsets[++line] = i + 1;
    }
    if (line < nFrames) offsets[nFrames] = this.buf.length;
    this.offsets = offsets;
    this.nSlots = this.raw(0).split(":")[1].split(";").length;
  }

  private raw(frame: number): string {
    return this.buf.toString("latin1", this.offsets[frame], this.offsets[frame + 1]).trim();
  }

  frame(frame: number): ParsedFrame {
    const [, playersPart = "", ballPart = ""] = this.raw(frame).split(":");
    const groups = playersPart.split(";");
    const xs = new Float64Array(this.nSlots).fill(NaN);
    const ys = new Float64Array(this.nSlots).fill(NaN);
    const speeds = new Float64Array(this.nSlots).fill(NaN);
    for (let s = 0; s < groups.length && s < this.nSlots; s++) {
      const [x, y, v] = groups[s].split(",");
      xs[s] = Number(x);
      ys[s] = Number(y);
      speeds[s] = Number(v);
    }
    const [bx, by, bv] = ballPart.split(",").map(Number);
    const ball = Number.isFinite(bx) && Number.isFinite(by) ? { x: bx, y: by, v: bv } : null;
    return { xs, ys, speeds, ball };
  }

  /** Whether the ball has a position. Some records carry a ball value with no position ("NaN,NaN,1.07"). */
  hasBall(frame: number): boolean {
    const s = this.raw(frame);
    const [x, y] = s.slice(s.lastIndexOf(":") + 1).split(",", 2);
    return x !== "NaN" && y !== "NaN";
  }
}

// ---------------------------------------------------------------------------------------------
// Homography: one 3x3 per frame, normalised pitch -> pixels.
// ---------------------------------------------------------------------------------------------

class Homography {
  readonly m: Float64Array;
  readonly err: Float64Array;
  readonly valid: Uint8Array;

  constructor(file: string, nFrames: number) {
    this.m = new Float64Array(nFrames * 9).fill(NaN);
    this.err = new Float64Array(nFrames).fill(NaN);
    this.valid = new Uint8Array(nFrames);
    const lines = fs.readFileSync(file, "utf8").split("\n");
    const header = lines[0].trim().split(",");
    const col = (name: string) => header.indexOf(name);
    const matCols = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => col(`h${r}${c}`)));
    const fCol = col("frame");
    const eCol = col("reprojection_error");
    const vCol = col("valid");
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      if (!line) continue;
      const parts = line.split(",");
      const f = Number(parts[fCol]);
      if (!(f >= 0 && f < nFrames)) continue;
      for (let k = 0; k < 9; k++) this.m[f * 9 + k] = Number(parts[matCols[k]]);
      this.err[f] = Number(parts[eCol]);
      this.valid[f] = Number(parts[vCol]) === 1 ? 1 : 0;
    }
  }

  at(frame: number): HomographyData {
    const m = Array.from(this.m.subarray(frame * 9, frame * 9 + 9));
    return {
      m: m.map((v) => (Number.isFinite(v) ? v : 0)),
      reprojectionError: this.err[frame],
      valid: this.valid[frame] === 1 && m.every(Number.isFinite),
    };
  }
}

// ---------------------------------------------------------------------------------------------
// Smart Tagging: SportsCode XML. Small and regular, so a regex parse is enough.
// ---------------------------------------------------------------------------------------------

function sideOfTeamName(name: string | undefined): Side | null {
  const n = (name ?? "").trim().toLowerCase();
  if (n.startsWith(TAG_TEAM_PREFIX.H)) return "H";
  if (n.startsWith(TAG_TEAM_PREFIX.A)) return "A";
  return null;
}

function loadSmartTagging(file: string): Tag[] {
  if (!fs.existsSync(file)) return [];
  const xml = fs.readFileSync(file, "utf8");
  const text = (block: string, tag: string) => block.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`))?.[1]?.trim();
  const tags: Tag[] = [];
  for (const [, block] of xml.matchAll(/<instance>([\s\S]*?)<\/instance>/g)) {
    const labels: Record<string, string> = {};
    for (const [, label] of block.matchAll(/<label>([\s\S]*?)<\/label>/g)) {
      const group = text(label, "group");
      const value = text(label, "text");
      if (group && value !== undefined) labels[group] = value;
    }
    const teamName = labels.Team && labels.Team !== "N/A" ? labels.Team : null;
    tags.push({
      id: Number(text(block, "ID")),
      code: text(block, "code") ?? "",
      start: Math.round(Number(text(block, "start")) * FPS),
      end: Math.round(Number(text(block, "end")) * FPS),
      team: sideOfTeamName(teamName ?? undefined),
      teamName,
      labels,
    });
  }
  return tags.sort((a, b) => a.start - b.start || a.end - b.end);
}

// ---------------------------------------------------------------------------------------------
// Ground truth (public 5 minutes).
// ---------------------------------------------------------------------------------------------

const LABEL_CODES: Label[] = ["X", "H", "A", "D"];

function loadGroundTruth(file: string): Uint8Array {
  const out = new Uint8Array(N_FRAMES); // 0 = X
  if (!fs.existsSync(file)) return out;
  const lines = fs.readFileSync(file, "utf8").split("\n");
  for (let i = 1; i < lines.length; i++) {
    const [f, label] = lines[i].trim().split(",");
    const code = LABEL_CODES.indexOf(label as Label);
    const frame = Number(f);
    if (code > 0 && frame >= 0 && frame < N_FRAMES) out[frame] = code;
  }
  return out;
}

// Our predictions: same frame,label format. Re-read whenever a file changes, so rerunning
// possession/predict.py shows up without restarting the viewer.
const predictionCache = new Map<string, { mtimeMs: number; codes: Uint8Array }>();

function loadPrediction(files: string[]): Uint8Array | null {
  const file = files.find((f) => fs.existsSync(f));
  if (!file) return null;
  const { mtimeMs } = fs.statSync(file);
  const hit = predictionCache.get(file);
  if (hit?.mtimeMs === mtimeMs) return hit.codes;
  const codes = loadGroundTruth(file);
  predictionCache.set(file, { mtimeMs, codes });
  return codes;
}

// Our player tracks (ballcv/player_track.py, exported by ballcv/viewer_export.py): where they cover a frame they
// replace Metrica's players. Read a frame at a time from the file.
let tracks: { mtimeMs: number; fd: number; frames: [number, number]; slots: number } | null = null;

function trackedPlayers(frame: number): Float32Array | null {
  if (!fs.existsSync(PATHS.playerTracks) || !fs.existsSync(PATHS.playerTracksMeta)) return null;
  const { mtimeMs } = fs.statSync(PATHS.playerTracks);
  if (tracks?.mtimeMs !== mtimeMs) {
    if (tracks) fs.closeSync(tracks.fd);
    const meta = JSON.parse(fs.readFileSync(PATHS.playerTracksMeta, "utf8")) as { frames: [number, number]; slots: number };
    tracks = { mtimeMs, fd: fs.openSync(PATHS.playerTracks, "r"), ...meta };
  }
  const [a, b] = tracks.frames;
  if (frame < a || frame > b) return null;
  const n = tracks.slots * 2;
  const buf = Buffer.alloc(n * 4);
  fs.readSync(tracks.fd, buf, 0, n * 4, (frame - a) * n * 4);
  return new Float32Array(buf.buffer, buf.byteOffset, n);
}

/** Metrica's frame, with our tracked players in place of its own where our tracks cover it (speeds from the
 * displacement since lookback frames before, m/s). */
function playersFrame(atd: Atd, frame: number, lookback: number): ParsedFrame {
  const fr = atd.frame(frame);
  const now = trackedPlayers(frame);
  if (!now) return fr;
  const before = frame >= lookback ? trackedPlayers(frame - lookback) : null;
  for (let s = 0; s < atd.nSlots; s++) {
    fr.xs[s] = now[2 * s];
    fr.ys[s] = now[2 * s + 1];
    const d = before ? metres({ x: now[2 * s], y: now[2 * s + 1] }, { x: before[2 * s], y: before[2 * s + 1] }) : NaN;
    fr.speeds[s] = Number.isFinite(d) ? (d * FPS) / lookback : NaN;
  }
  return fr;
}

// Our own ball track and touches. Re-read whenever the file changes.
interface TouchFile {
  frames: [number, number];
  track: Record<string, [number, number]>;
  events: TouchEvent[];
  plays: BallPlay[];
  segments?: BallStretch[];
}
// Stoppages from ballcv/stoppages.py. Re-read whenever the file changes.
const stoppageCache: { mtimeMs: number; data: Stoppage[] | null } = { mtimeMs: -1, data: null };

function loadStoppages(file: string): Stoppage[] | null {
  if (!fs.existsSync(file)) return null;
  const { mtimeMs } = fs.statSync(file);
  if (mtimeMs !== stoppageCache.mtimeMs) {
    stoppageCache.data = (JSON.parse(fs.readFileSync(file, "utf8")) as { stoppages: Stoppage[] }).stoppages;
    stoppageCache.mtimeMs = mtimeMs;
  }
  return stoppageCache.data;
}

// The assistant referees and their flags (ballcv/flag_self.py, via viewer_export.py). Re-read whenever a file changes.
interface AssistantFile {
  step: number;
  threshold: number;
  frames: Record<string, Partial<Record<Assistant["line"], [number, number]>>>; // [x m, score]
}
interface FlagFile {
  step: number;
  // [above the shoulders, out from the body (person heights), signalling 0/1 -- level or above, away from the body]
  samples: Record<string, Partial<Record<Assistant["line"], [number, number, number] | null>>>;
  signals: AssistantFlag[];
}
function cachedJson<T>(): (file: string) => T | null {
  const cache: { mtimeMs: number; data: T | null } = { mtimeMs: -1, data: null };
  return (file) => {
    if (!fs.existsSync(file)) return null;
    const { mtimeMs } = fs.statSync(file);
    if (mtimeMs !== cache.mtimeMs) {
      cache.data = JSON.parse(fs.readFileSync(file, "utf8")) as T;
      cache.mtimeMs = mtimeMs;
    }
    return cache.data;
  };
}
const loadAssistants = cachedJson<AssistantFile>();
const loadFlags = cachedJson<FlagFile>();

/** Each assistant the model found at the sample nearest this frame, with where their flag was. */
function assistantsAt(det: AssistantFile, flags: FlagFile | null, frame: number): Assistant[] {
  const f = String(Math.round(frame / det.step) * det.step);
  const out: Assistant[] = [];
  for (const line of ["far", "near"] as const) {
    const d = det.frames[f]?.[line];
    if (!d || d[1] < det.threshold) continue;
    const flag = flags?.samples[f]?.[line];
    out.push({ line, x: d[0], score: d[1], flagUp: flag ? flag[2] === 1 : null });
  }
  return out;
}

const touchCache = new Map<string, { mtimeMs: number; data: TouchFile }>(); // by file: one per ball detector

function loadTouches(file: string): TouchFile | null {
  if (!fs.existsSync(file)) return null;
  const { mtimeMs } = fs.statSync(file);
  const hit = touchCache.get(file);
  if (hit?.mtimeMs === mtimeMs) return hit.data;
  const raw = JSON.parse(fs.readFileSync(file, "utf8")) as TouchFile;
  raw.events.sort((a, b) => a.f - b.f);
  raw.plays.sort((a, b) => a.f0 - b.f0);
  touchCache.set(file, { mtimeMs, data: raw });
  return raw;
}

// Our ball track: the self-trained ball detector (ballcv/selftrain.py, model 17) linked by ownball.py.
const OWN_TRACKS: { key: OwnTrackKey; label: string; file: string }[] = [{ key: "ball", label: "Our ball", file: PATHS.touches }];

/** A track's ball on frame f, on the grass: its pixel position through frame f's own homography. */
function ownBallAt(tf: TouchFile, homography: Homography, f: number): Ball | null {
  const p = tf.track[String(f)];
  const h = p ? homography.at(f) : null;
  const g = h?.valid ? unproject(h.m, p[0], p[1]) : null;
  return g ? { x: g[0], y: g[1] } : null;
}

function ownTracksAt(homography: Homography, frame: number): OwnTrack[] {
  const out: OwnTrack[] = [];
  for (const { key, label, file } of OWN_TRACKS) {
    const tf = loadTouches(file);
    if (!tf) continue;
    const trail: OwnTrack["trail"] = [];
    for (let ago = 0; ago <= TRAIL_FRAMES && frame - ago >= 0; ago++) {
      const b = ownBallAt(tf, homography, frame - ago);
      if (b) trail.push({ ...b, ago });
    }
    out.push({ key, label, ball: trail[0]?.ago === 0 ? { x: trail[0].x, y: trail[0].y } : null, trail });
  }
  return out;
}

function touchesAt(tf: TouchFile, frame: number, window: { start: number; end: number }): TouchData {
  const touches = tf.events.filter((e) => e.kind === "touch");
  let lastTouch: TouchEvent | null = null;
  let nextTouch: TouchEvent | null = null;
  for (const e of touches) {
    if (e.f <= frame) lastTouch = e;
    else {
      nextTouch = e;
      break;
    }
  }
  return {
    covered: { start: tf.frames[0], end: tf.frames[1] },
    ownBall: tf.track[String(frame)] ?? null,
    lastTouch,
    nextTouch,
    play: tf.plays.find((p) => p.f0 <= frame && frame < p.f1) ?? null,
    events: tf.events.filter((e) => e.f >= window.start && e.f <= window.end),
    plays: tf.plays.filter((p) => p.f1 >= window.start && p.f0 <= window.end),
    stretches: (tf.segments ?? []).filter((g) => g.f1 >= window.start && g.f0 <= window.end),
  };
}

// ---------------------------------------------------------------------------------------------
// Which track slot belongs to which team.
//
// The ATD metadata XML assigns every slot to a team; the first team listed is home. When that file
// is not present we infer the split from the data: the two teams defend opposite ends and swap at
// half-time, so each slot's mean x in the 1st half minus its mean x in the 2nd half separates them.
// The public ground truth then tells us which group is home -- the team whose player is nearest the
// ball on H frames.
// ---------------------------------------------------------------------------------------------

function teamsFromXml(file: string, nSlots: number): Side[] | null {
  if (!fs.existsSync(file)) return null;
  const xml = fs.readFileSync(file, "utf8");
  const teamIds = [...xml.matchAll(/<Team\b[^>]*\bid="([^"]+)"/g)].map((m) => m[1]);
  const slots = [...xml.matchAll(/<Player\b[^>]*\bteamId="([^"]+)"/g)].map((m) => m[1]);
  if (teamIds.length !== 2 || slots.length !== nSlots) return null;
  return slots.map((id) => (id === teamIds[0] ? "H" : "A"));
}

/** Slot indices (0-based) the metadata marks as goalkeepers. */
function keeperSlotsFromXml(file: string): number[] {
  if (!fs.existsSync(file)) return [];
  const xml = fs.readFileSync(file, "utf8");
  return [...xml.matchAll(/<Player\b[^>]*\bid="(\d+)"[^>]*>([\s\S]*?)<\/Player>/g)]
    .filter((m) => /<Name>position_type<\/Name>\s*<Value>Goalkeeper<\/Value>/.test(m[2]))
    .map((m) => Number(m[1]) - 1);
}

/** For each half, whether the home team defends the x = 0 end: its players' median x is the smaller. */
function homeDefendsLeft(atd: Atd, slotTeams: Side[]): Partial<Record<PeriodKey, boolean>> {
  const out: Partial<Record<PeriodKey, boolean>> = {};
  for (const p of PERIODS.filter((q) => q.graded)) {
    const xs: Record<Side, number[]> = { H: [], A: [] };
    for (let f = p.start; f <= p.end; f += 250) {
      const fr = atd.frame(f);
      for (let s = 0; s < atd.nSlots; s++) if (Number.isFinite(fr.xs[s])) xs[slotTeams[s]].push(fr.xs[s]);
    }
    const median = (v: number[]) => v.sort((a, b) => a - b)[v.length >> 1] ?? 0.5;
    out[p.key] = median(xs.H) < median(xs.A);
  }
  return out;
}

// The EPTS export lists slots team by team, so the two teams are two contiguous blocks of slot
// indices; only the boundary (and which block is home) has to be found. Individual slots are too
// noisy to classify one by one -- the tracker swaps identities -- but a single boundary is robust.
const KICKOFF_SIDE_MARGIN = 0.03; // ~3 m: ignore players standing on the halfway line

function inferTeams(atd: Atd, gt: Uint8Array, tags: Tag[]): Side[] {
  const n = atd.nSlots;
  // score[s] < 0: slot looks like it defends the x=0 end in the 1st half ("left group").
  const score = new Float64Array(n);
  const inSecondHalf = (f: number) => f >= PERIODS[2].start;

  // 1. Kickoffs: everyone stands in their own half. Sample just after each tagged kickoff starts.
  const kickoffs = tags.filter((t) => (t.labels.Type ?? "").toLowerCase().startsWith("kick off"));
  for (const t of kickoffs) {
    const f = Math.min(t.start + FPS, N_FRAMES - 1);
    const { xs, speeds } = atd.frame(f);
    const flip = inSecondHalf(f) ? -1 : 1;
    for (let s = 0; s < n; s++) {
      if (!Number.isFinite(xs[s]) || !Number.isFinite(speeds[s])) continue;
      const d = xs[s] - 0.5;
      if (Math.abs(d) > KICKOFF_SIDE_MARGIN) score[s] += 3 * flip * Math.sign(d);
    }
  }

  // 2. Ends swap at half-time: mean x in the 1st half minus mean x in the 2nd, sampled once a second.
  const sums = [new Float64Array(n), new Float64Array(n)];
  const counts = [new Float64Array(n), new Float64Array(n)];
  [
    [FIRST_HALF_KICKOFF, PERIODS[1].end],
    [SECOND_HALF_KICKOFF, PERIODS[3].end],
  ].forEach(([start, end], h) => {
    for (let f = start; f <= end; f += FPS) {
      const { xs, speeds } = atd.frame(f);
      for (let s = 0; s < n; s++) {
        // Slots with no speed are frozen placeholders; they say nothing about a team's half.
        if (Number.isFinite(xs[s]) && Number.isFinite(speeds[s])) {
          sums[h][s] += xs[s];
          counts[h][s] += 1;
        }
      }
    }
  });
  for (let s = 0; s < n; s++) {
    if (counts[0][s] > 50 && counts[1][s] > 50) {
      score[s] += 20 * (sums[0][s] / counts[0][s] - sums[1][s] / counts[1][s]);
    }
  }

  // 3. Best single boundary k: slots [0, k) in one team, [k, n) in the other.
  let best = { k: Math.floor(n / 2), value: -Infinity, firstIsLeft: true };
  for (let k = 1; k < n; k++) {
    let firstLeft = 0;
    for (let s = 0; s < n; s++) firstLeft += s < k ? -score[s] : score[s];
    for (const [value, firstIsLeft] of [
      [firstLeft, true],
      [-firstLeft, false],
    ] as const) {
      if (value > best.value) best = { k, value, firstIsLeft };
    }
  }
  const leftGroup = Array.from({ length: n }, (_, s) => (s < best.k) === best.firstIsLeft);

  // Orient with the ground truth: on H/A frames, does the nearest player belong to the left group?
  let leftIsHomeVotes = 0;
  for (let f = PUBLIC_GT.start; f <= PUBLIC_GT.end; f++) {
    const label = LABEL_CODES[gt[f]];
    if (label !== "H" && label !== "A") continue;
    const fr = atd.frame(f);
    if (!fr.ball) continue;
    let bestDist = Infinity;
    let bestSlot = -1;
    for (let s = 0; s < n; s++) {
      if (!Number.isFinite(fr.xs[s]) || !Number.isFinite(fr.speeds[s])) continue;
      const d = Math.hypot((fr.xs[s] - fr.ball.x) * PITCH_M.length, (fr.ys[s] - fr.ball.y) * PITCH_M.width);
      if (d < bestDist) [bestDist, bestSlot] = [d, s];
    }
    if (bestSlot < 0) continue;
    const nearestIsLeft = leftGroup[bestSlot];
    leftIsHomeVotes += (label === "H") === nearestIsLeft ? 1 : -1;
  }
  const leftIsHome = leftIsHomeVotes >= 0;
  console.log(
    `[viewer] inferred team blocks: slots 1-${best.k} / ${best.k + 1}-${n}; ground-truth orientation vote ${leftIsHomeVotes}`,
  );
  return leftGroup.map((isLeft) => (isLeft === leftIsHome ? "H" : "A"));
}

// ---------------------------------------------------------------------------------------------
// The broadcast camera, aligned by alignment/align_broadcast.py. Each tactical frame maps to at
// most one broadcast frame (one constant offset per half). Replays show an earlier moment while the
// clock runs on, so only LIVE broadcast frames contribute tracking; replays and their wipes are
// shown for context and otherwise disregarded.
// ---------------------------------------------------------------------------------------------

const BROADCAST_KINDS = ["live", "replay", "wipe", "break"] as const;
const BROADCAST_ATD_FRAMES = BROADCAST_VIDEO_FRAMES + 1; // the ATD has one record more than the video

interface Broadcast {
  atd: Atd;
  homography: Homography;
  slotTeams: Side[];
  toBroadcast: Int32Array; // tactical frame -> broadcast frame at the same instant, -1 if none
  kind: Uint8Array; // broadcast frame -> index into BROADCAST_KINDS
}

function loadBroadcast(): Broadcast | null {
  const needed = [PATHS.broadcastAtdRaw, PATHS.broadcastAtdMeta, PATHS.broadcastHomography, PATHS.broadcastAlignment];
  const missing = needed.filter((p) => !fs.existsSync(p));
  if (missing.length) {
    console.log(`[viewer] broadcast disabled, missing: ${missing.join(", ")}`);
    return null;
  }
  const atd = new Atd(PATHS.broadcastAtdRaw, BROADCAST_ATD_FRAMES);
  const slotTeams = teamsFromXml(PATHS.broadcastAtdMeta, atd.nSlots);
  if (!slotTeams) {
    console.log("[viewer] broadcast disabled: could not read slot teams from the ATD metadata");
    return null;
  }
  const toBroadcast = new Int32Array(N_FRAMES).fill(-1);
  const kind = new Uint8Array(BROADCAST_ATD_FRAMES).fill(BROADCAST_KINDS.indexOf("break"));
  const lines = fs.readFileSync(PATHS.broadcastAlignment, "utf8").split("\n");
  const header = lines[0].trim().split(",");
  const [bCol, tCol, kCol] = ["broadcast_frame", "tactical_frame", "kind"].map((c) => header.indexOf(c));
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const parts = lines[i].split(",");
    const b = Number(parts[bCol]);
    const k = BROADCAST_KINDS.indexOf(parts[kCol] as (typeof BROADCAST_KINDS)[number]);
    if (!(b >= 0 && b < BROADCAST_ATD_FRAMES) || k < 0) continue;
    kind[b] = k;
    const t = parts[tCol] === "" ? NaN : Number(parts[tCol]);
    if (t >= 0 && t < N_FRAMES) toBroadcast[t] = b;
  }
  return { atd, homography: new Homography(PATHS.broadcastHomography, BROADCAST_ATD_FRAMES), slotTeams, toBroadcast, kind };
}

function broadcastStatus(bc: Broadcast, frame: number): BroadcastStatus {
  const b = bc.toBroadcast[frame];
  if (b < 0) return "none";
  const k = BROADCAST_KINDS[bc.kind[b]];
  return k === "break" ? "none" : k;
}

// ---------------------------------------------------------------------------------------------
// Whistles heard in the broadcast audio, already on the tactical timeline. A few dozen rows.
// ---------------------------------------------------------------------------------------------

const WHISTLE_ROLES: Whistle["role"][] = ["stoppage", "restart", "unclear"];

function loadWhistles(file: string): Whistle[] | null {
  if (!fs.existsSync(file)) {
    console.log(`[viewer] whistles disabled, missing: ${file}`);
    return null;
  }
  const lines = fs.readFileSync(file, "utf8").split("\n");
  const header = lines[0].trim().split(",");
  const [sCol, eCol, kCol, scCol, hzCol, rCol] = ["tactical_frame", "tactical_frame_end", "kind", "score", "peak_hz", "role"].map(
    (c) => header.indexOf(c),
  );
  const out: Whistle[] = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const parts = lines[i].split(",");
    const start = Number(parts[sCol]);
    const end = Number(parts[eCol]);
    if (!(start >= 0 && start < N_FRAMES)) continue;
    out.push({
      start,
      end: Math.min(Math.max(end, start), N_FRAMES - 1),
      kind: BROADCAST_KINDS.includes(parts[kCol] as Whistle["kind"]) ? (parts[kCol] as Whistle["kind"]) : "live",
      score: Number(parts[scCol]),
      hz: Number(parts[hzCol]),
      role: WHISTLE_ROLES.includes(parts[rCol] as Whistle["role"]) ? (parts[rCol] as Whistle["role"]) : "unclear",
    });
  }
  return out.sort((a, b) => a.start - b.start);
}

// ---------------------------------------------------------------------------------------------
// Cache everything once per server process (and across dev hot reloads).
// ---------------------------------------------------------------------------------------------

interface Store {
  atd: Atd;
  homography: Homography;
  tags: Tag[];
  gt: Uint8Array;
  slotTeams: Side[];
  keepers: number[]; // goalkeeper slots (0-based)
  homeLeft: Partial<Record<PeriodKey, boolean>>; // per half: the home team defends x = 0
  teamSource: "xml" | "inferred";
  broadcast: Broadcast | null;
  whistles: Whistle[] | null;
}

const globalStore = globalThis as unknown as { __mtahStore?: Store };

export class MissingDataError extends Error {}

function store(): Store {
  if (globalStore.__mtahStore?.keepers) return globalStore.__mtahStore; // (rebuilt when an older copy lacks fields)
  for (const p of [PATHS.atdRaw, PATHS.homography]) {
    if (!fs.existsSync(p)) throw new MissingDataError(`Missing data file: ${p}`);
  }
  const t0 = Date.now();
  const atd = new Atd(PATHS.atdRaw, N_FRAMES);
  const homography = new Homography(PATHS.homography, N_FRAMES);
  const tags = loadSmartTagging(PATHS.smartTagging);
  const gt = loadGroundTruth(PATHS.gtPublic);
  const fromXml = teamsFromXml(PATHS.atdMeta, atd.nSlots);
  const slotTeams = fromXml ?? inferTeams(atd, gt, tags);
  const teamSource = fromXml ? "xml" : "inferred";
  const broadcast = loadBroadcast();
  const whistles = loadWhistles(PATHS.whistles);
  console.log(
    `[viewer] loaded in ${Date.now() - t0} ms; teams ${teamSource}: ` +
      `H = slots ${slotTeams.flatMap((t, s) => (t === "H" ? [s + 1] : [])).join(",")}`,
  );
  const keepers = keeperSlotsFromXml(PATHS.atdMeta);
  const homeLeft = homeDefendsLeft(atd, slotTeams);
  globalStore.__mtahStore = { atd, homography, tags, gt, slotTeams, teamSource, broadcast, whistles, keepers, homeLeft };
  return globalStore.__mtahStore;
}

// ---------------------------------------------------------------------------------------------
// Frame assembly.
// ---------------------------------------------------------------------------------------------

function runs<T extends string>(start: number, end: number, valueAt: (f: number) => T | null): Run<T>[] {
  const out: Run<T>[] = [];
  for (let f = start; f <= end; f++) {
    const v = valueAt(f);
    const last = out[out.length - 1];
    if (v !== null && last && last.value === v && last.end === f - 1) last.end = f;
    else if (v !== null) out.push({ start: f, end: f, value: v });
  }
  return out;
}

const metres = (a: Ball, b: Ball) => Math.hypot((a.x - b.x) * PITCH_M.length, (a.y - b.y) * PITCH_M.width);

/** Metrica's ball in the tactical ATD (normalised 0-1), or null where it has none. */
function tacticalBallAt(atd: Atd, frame: number): (Ball & { source: BallPoint["source"] }) | null {
  if (frame < 0) return null;
  const b = atd.frame(frame).ball;
  return b && Number.isFinite(b.x) && Number.isFinite(b.y) ? { x: b.x, y: b.y, source: "tactical" } : null;
}

/** Metrica's ball in the broadcast ATD at the same instant as tactical frame `frame`: live broadcast frames only (a
 * replay shows an earlier moment). */
function broadcastBallAt(bc: Broadcast | null, frame: number): (Ball & { source: BallPoint["source"] }) | null {
  if (!bc || frame < 0 || broadcastStatus(bc, frame) !== "live") return null;
  const b = bc.atd.frame(bc.toBroadcast[frame]).ball;
  return b && Number.isFinite(b.x) && Number.isFinite(b.y) ? { x: b.x, y: b.y, source: "broadcast" } : null;
}

function broadcastAt(bc: Broadcast, frame: number, window: { start: number; end: number }): BroadcastData {
  const status = broadcastStatus(bc, frame);
  const b = bc.toBroadcast[frame];
  const players: BroadcastData["players"] = [];
  if (status === "live") {
    const fr = bc.atd.frame(b);
    for (let s = 0; s < bc.atd.nSlots; s++) {
      if (Number.isFinite(fr.xs[s]) && Number.isFinite(fr.ys[s])) {
        players.push({ slot: s + 1, team: bc.slotTeams[s], x: fr.xs[s], y: fr.ys[s] });
      }
    }
  }
  return {
    frame: b >= 0 ? b : null,
    status,
    players,
    homography: b >= 0 ? bc.homography.at(b) : null,
    runs: runs(window.start, window.end, (f) => {
      const st = broadcastStatus(bc, f);
      return st === "none" ? null : st;
    }),
  };
}

export function getFrame(frame: number): FrameData {
  const { atd, homography, tags, gt, slotTeams, teamSource, broadcast: bc, whistles, keepers, homeLeft } = store();
  // a goalkeeper is the team defending the half he is in: keepers do not cross halfway, while Metrica's keeper tracks
  // swap (#46 follows the home keeper at times) and a frozen track drifts
  const left = homeLeft[periodOf(frame).key];
  const teamOf = (s: number, x: number): Side =>
    left === undefined || !keepers.includes(s) ? slotTeams[s] : (x < 0.5) === left ? "H" : "A";
  const fr = playersFrame(atd, frame, HEADING_LOOKBACK);
  const prev = frame >= HEADING_LOOKBACK ? playersFrame(atd, frame - HEADING_LOOKBACK, HEADING_LOOKBACK) : null;
  const window = { start: Math.max(0, frame - WINDOW_FRAMES), end: Math.min(N_FRAMES - 1, frame + WINDOW_FRAMES) };
  const broadcast = bc ? broadcastAt(bc, frame, window) : null;
  const pred = loadPrediction(PATHS.prediction);
  const touchModel = loadPrediction([PATHS.touchModel]);
  const touchFile = loadTouches(PATHS.touches);
  const ballInUse = tacticalBallAt(atd, frame);
  const broadcastBall = broadcastBallAt(bc, frame);
  const ballForPlayers = ballInUse ?? broadcastBall;
  const assistantDet = loadAssistants(PATHS.assistants);
  const flagFile = loadFlags(PATHS.flagSignals);

  const players: Player[] = [];
  for (let s = 0; s < atd.nSlots; s++) {
    const x = fr.xs[s];
    const y = fr.ys[s];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const px = prev?.xs[s];
    const py = prev?.ys[s];
    const moving = prev && Number.isFinite(px) && Number.isFinite(py);
    const scale = FPS / HEADING_LOOKBACK;
    players.push({
      slot: s + 1,
      team: teamOf(s, x),
      x,
      y,
      speed: Number.isFinite(fr.speeds[s]) ? fr.speeds[s] : null,
      vx: moving ? (x - (px as number)) * scale : null,
      vy: moving ? (y - (py as number)) * scale : null,
      distToBallM: ballForPlayers ? metres({ x, y }, ballForPlayers) : null,
    });
  }

  const candidates = players.filter((p) => p.distToBallM !== null && p.speed !== null);
  const nearestPlayer = candidates.sort((a, b) => (a.distToBallM as number) - (b.distToBallM as number))[0];

  const ballTrail: BallPoint[] = [];
  const broadcastBallTrail: BallPoint[] = [];
  for (let ago = 0; ago <= TRAIL_FRAMES && frame - ago >= 0; ago++) {
    const b = ago === 0 ? ballInUse : tacticalBallAt(atd, frame - ago);
    if (b) ballTrail.push({ ...b, ago });
    const c = ago === 0 ? broadcastBall : broadcastBallAt(bc, frame - ago);
    if (c) broadcastBallTrail.push({ ...c, ago });
  }

  let ballLastSeen: BallPoint | null = null;
  if (!ballInUse) {
    for (let ago = 1; ago <= LAST_SEEN_LOOKBACK && frame - ago >= 0; ago++) {
      const b = tacticalBallAt(atd, frame - ago);
      if (b) {
        ballLastSeen = { ...b, ago };
        break;
      }
    }
  }

  const period = periodOf(frame);
  return {
    frame,
    period: { key: period.key, label: period.label, graded: period.graded },
    teams: TEAM_NAMES,
    teamSource,
    players,
    ballInUse,
    ballLastSeen,
    ballTrail,
    broadcastBall,
    broadcastBallTrail,
    ballGapM: ballInUse && broadcastBall ? metres(ballInUse, broadcastBall) : null,
    nearest: nearestPlayer
      ? { slot: nearestPlayer.slot, team: nearestPlayer.team, distM: nearestPlayer.distToBallM as number }
      : null,
    homography: homography.at(frame),
    broadcast,
    whistles,
    tagsActive: tags.filter((t) => t.start <= frame && frame < t.end),
    window,
    tagsWindow: tags.filter((t) => t.end > window.start && t.start <= window.end),
    gt: LABEL_CODES[gt[frame]],
    gtRuns: runs(window.start, window.end, (f) => (gt[f] ? LABEL_CODES[gt[f]] : null)),
    touches: touchFile ? touchesAt(touchFile, frame, window) : null,
    ownTracks: ownTracksAt(homography, frame),
    stoppages: loadStoppages(PATHS.stoppages)?.filter((s) => s.end >= window.start && s.start <= window.end) ?? null,
    assistants: assistantDet ? assistantsAt(assistantDet, flagFile, frame) : null,
    assistantFlags: flagFile ? flagFile.signals.filter((s) => s.end >= window.start && s.start <= window.end) : null,
    pred: pred ? LABEL_CODES[pred[frame]] : null,
    predRuns: pred ? runs(window.start, window.end, (f) => (pred[f] ? LABEL_CODES[pred[f]] : null)) : null,
    touchModelRuns: touchModel ? runs(window.start, window.end, (f) => (touchModel[f] ? LABEL_CODES[touchModel[f]] : null)) : null,
  };
}
