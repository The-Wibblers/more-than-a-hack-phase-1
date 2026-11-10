import "server-only";

import fs from "node:fs";
import path from "node:path";

import type { GameEvent, GameFrame, GameIndex, GameMeta, GamePlayer } from "../games";

// Bundles written by `python -m external.build` (see external/__init__.py for the format).
// A game's tracking is read whole into memory on first use (25-75 MB) and kept for the few most
// recently viewed games; after that a frame is a slice.

const REPO_ROOT = path.resolve(process.cwd(), "..");
export const DATA_DIR = process.env.MTAH_DATA_DIR ?? path.join(REPO_ROOT, "data");
const GAMES_DIR = path.join(DATA_DIR, "external", "games");
const KEEP_GAMES = 3;
const TRAIL_S = 2;
const SPEED_S = 0.2;

interface Loaded {
  meta: GameMeta;
  events: GameEvent[];
  xyz: Float32Array; // [n_frames, n_objects, 3]
  state: Uint8Array;
  boxes: Int16Array | null; // [n_frames, n_objects, 4]
}

const loaded = new Map<string, Loaded>();

const SAFE_ID = /^[A-Za-z0-9_-]+$/;

export function gameIndex(): GameIndex {
  const file = path.join(GAMES_DIR, "index.json");
  if (!fs.existsSync(file)) return { datasets: {}, games: [] };
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function floatView(buf: Buffer): Float32Array {
  // Copy into an aligned buffer: a Buffer from readFileSync may sit at any byte offset.
  const out = new Float32Array(buf.length / 4);
  new Uint8Array(out.buffer).set(buf);
  return out;
}

function load(id: string): Loaded | null {
  if (!SAFE_ID.test(id)) return null;
  const hit = loaded.get(id);
  if (hit) {
    loaded.delete(id);
    loaded.set(id, hit); // refresh LRU position
    return hit;
  }
  const dir = path.join(GAMES_DIR, id);
  if (!fs.existsSync(path.join(dir, "meta.json"))) return null;
  const meta: GameMeta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
  const events: GameEvent[] = JSON.parse(fs.readFileSync(path.join(dir, "events.json"), "utf8"));
  const xyz = floatView(fs.readFileSync(path.join(dir, "tracking.f32")));
  const state = new Uint8Array(fs.readFileSync(path.join(dir, "state.u8")));
  const boxesFile = path.join(dir, "boxes.i16");
  let boxes: Int16Array | null = null;
  if (fs.existsSync(boxesFile)) {
    const buf = fs.readFileSync(boxesFile);
    boxes = new Int16Array(buf.length / 2);
    new Uint8Array(boxes.buffer).set(buf);
  }
  const game = { meta, events, xyz, state, boxes };
  loaded.set(id, game);
  if (loaded.size > KEEP_GAMES) loaded.delete(loaded.keys().next().value as string);
  return game;
}

export function gameMeta(id: string): { meta: GameMeta; events: GameEvent[] } | null {
  const g = load(id);
  return g && { meta: g.meta, events: g.events };
}

export function clampGameFrame(meta: GameMeta, frame: number): number {
  return Math.min(meta.n_frames - 1, Math.max(0, Math.round(frame)));
}

const round2 = (v: number) => Math.round(v * 100) / 100;

export function gameFrame(id: string, n: number): GameFrame | null {
  const g = load(id);
  if (!g) return null;
  const { meta, xyz, state, boxes } = g;
  const k = meta.objects.length;
  const frame = clampGameFrame(meta, n);
  const period = meta.periods.find((p) => frame >= p.start && frame <= p.end) ?? meta.periods[0];
  const at = (f: number, j: number, c: number) => xyz[(f * k + j) * 3 + c];
  const lag = Math.max(1, Math.round(meta.fps * (meta.speed_window_s ?? SPEED_S)));

  const players: GamePlayer[] = [];
  for (let j = 1; j < k; j++) {
    const x = at(frame, j, 0);
    const y = at(frame, j, 1);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const back = frame - lag;
    let vx: number | null = null;
    let vy: number | null = null;
    if (back >= period.start && Number.isFinite(at(back, j, 0))) {
      vx = ((x - at(back, j, 0)) * meta.fps) / lag;
      vy = ((y - at(back, j, 1)) * meta.fps) / lag;
    }
    const o = meta.objects[j];
    players.push({
      obj: j, team: o.team, jersey: o.jersey, name: o.name, position: o.position,
      x: round2(x), y: round2(y),
      vx: vx === null ? null : round2(vx), vy: vy === null ? null : round2(vy),
      speed: vx === null || vy === null ? null : round2(Math.hypot(vx, vy)),
    });
  }

  const bx = at(frame, 0, 0);
  const by = at(frame, 0, 1);
  const bz = at(frame, 0, 2);
  const ball = Number.isFinite(bx) && Number.isFinite(by) ? { x: round2(bx), y: round2(by), z: Number.isFinite(bz) ? round2(bz) : null } : null;
  const ballTrail: GameFrame["ballTrail"] = [];
  const step = Math.max(1, Math.round(meta.fps / 10));
  for (let ago = step; ago <= TRAIL_S * meta.fps; ago += step) {
    const f = frame - ago;
    if (f < period.start) break;
    const x = at(f, 0, 0);
    const y = at(f, 0, 1);
    if (Number.isFinite(x) && Number.isFinite(y)) ballTrail.push({ x: round2(x), y: round2(y), ago });
  }

  const s = state[frame];
  const frameBoxes: GameFrame["boxes"] = [];
  if (boxes) {
    for (let j = 1; j < k; j++) {
      const o = (frame * k + j) * 4;
      if (boxes[o + 2] > 0) frameBoxes.push({ obj: j, x: boxes[o], y: boxes[o + 1], w: boxes[o + 2], h: boxes[o + 3] });
    }
  }

  return {
    frame,
    period: period.id,
    ball,
    ballTrail,
    players,
    alive: s & 2 ? (s & 1) === 1 : null,
    owner: s & 4 ? "H" : s & 8 ? "A" : null,
    boxes: frameBoxes,
  };
}

/**
 * What shows frame n: a video file and the time in it, or (for clips shipped as frames) the JPEG
 * itself. Null when the game has no video there.
 */
export function gameVideoAt(id: string, n: number): { file: string; seconds: number } | { jpeg: string } | null {
  const g = load(id);
  if (!g?.meta.video) return null;
  const v = g.meta.video.files.find((f) => n >= f.start && n < f.start + f.frames);
  if (!v) return null;
  const file = path.join(DATA_DIR, v.path);
  if (v.frames_dir) {
    const jpeg = path.join(file, `${String(n - v.start + 1).padStart(6, "0")}.jpg`);
    return fs.existsSync(jpeg) ? { jpeg } : null;
  }
  if (!fs.existsSync(file)) return null;
  return { file, seconds: (n - v.start) / g.meta.fps + (v.seek_offset_s ?? 0) };
}
