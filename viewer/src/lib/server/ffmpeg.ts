import "server-only";

import { spawn } from "node:child_process";
import fs from "node:fs";

import { FPS } from "../match";
import { videoPath, type Source } from "./data";

// Frame extraction with ffmpeg. Both mp4s are constant 25 fps from t = 0 with frequent keyframes
// (2 s tactical, 1.2 s broadcast), so an input-side seek to n/25 is frame-exact (checked against a
// full decode) and costs ~70-160 ms. n/25 has at most two decimals, so the timestamp is exact.

const CANDIDATES = [process.env.FFMPEG_PATH, "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"];
const MAX_CONCURRENT = 4;
const CACHE_SIZE = 200; // ~300 KB per JPEG

export function ffmpegPath(): string | null {
  return CANDIDATES.find((p): p is string => !!p && fs.existsSync(p)) ?? null;
}

const cache = new Map<string, Buffer>();
let running = 0;
const queue: (() => void)[] = [];

async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (running >= MAX_CONCURRENT) await new Promise<void>((resolve) => queue.push(resolve));
  running++;
  try {
    return await fn();
  } finally {
    running--;
    queue.shift()?.();
  }
}

function run(bin: string, args: string[]): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const proc = spawn(bin, args, { stdio: ["ignore", "pipe", "pipe"] });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    proc.stdout.on("data", (d: Buffer) => out.push(d));
    proc.stderr.on("data", (d: Buffer) => err.push(d));
    proc.on("error", reject);
    proc.on("close", (code) => {
      const buf = Buffer.concat(out);
      if (code === 0 && buf.length > 0) resolve(buf);
      else reject(new Error(`ffmpeg exited ${code}: ${Buffer.concat(err).toString().trim()}`));
    });
  });
}

async function cached(key: string, make: () => Promise<Buffer>): Promise<Buffer> {
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit); // refresh LRU position
    return hit;
  }
  const jpeg = await withSlot(make);
  cache.set(key, jpeg);
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value as string);
  return jpeg;
}

/** JPEG of video frame n of a source, or null when ffmpeg or the video is unavailable. */
export async function extractFrame(n: number, source: Source = "tactical"): Promise<Buffer | null> {
  const bin = ffmpegPath();
  const file = videoPath(source);
  if (!bin || !file) return null;
  const t = (n / FPS).toFixed(2);
  return cached(`${source}:${n}`, () =>
    run(bin, ["-nostdin", "-v", "error", "-ss", t, "-i", file, "-frames:v", "1", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]),
  );
}

/** JPEG of the frame at `seconds` into any video file, scaled to `width` px wide. */
export async function extractFileFrame(file: string, seconds: number, width: number): Promise<Buffer | null> {
  const bin = ffmpegPath();
  if (!bin || !fs.existsSync(file)) return null;
  const t = seconds.toFixed(3);
  return cached(`${file}:${t}:${width}`, () =>
    run(bin, ["-nostdin", "-v", "error", "-ss", t, "-i", file, "-frames:v", "1", "-vf", `scale=${width}:-2`, "-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]),
  );
}

export const VIDEO_SIZE = { width: 1920, height: 1080 };

/** Square crop of side `size` px whose top-left corner is (x0, y0), scaled to `out` px. */
export async function extractCrop(
  n: number,
  x0: number,
  y0: number,
  size: number,
  out: number,
  source: Source = "tactical",
): Promise<Buffer | null> {
  const bin = ffmpegPath();
  const file = videoPath(source);
  if (!bin || !file) return null;
  const t = (n / FPS).toFixed(2);
  const vf = `crop=${size}:${size}:${x0}:${y0},scale=${out}:${out}:flags=lanczos`;
  return cached(`${source}:${n}:${x0}:${y0}:${size}:${out}`, () =>
    run(bin, ["-nostdin", "-v", "error", "-ss", t, "-i", file, "-frames:v", "1", "-vf", vf, "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]),
  );
}
