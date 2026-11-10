import fs from "node:fs";

import { extractFileFrame } from "@/lib/server/ffmpeg";
import { gameVideoAt } from "@/lib/server/games";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const WIDTH = 1920; // the panoramas are 4K; the page never shows them wider than this
const HEADERS = { "Content-Type": "image/jpeg", "Cache-Control": "private, max-age=31536000, immutable" };

// GET /api/games/<id>/video/<n> -> JPEG of the game's video at frame n (videos scaled to 1920 px
// wide; clips that ship as frames are served as they are).
export async function GET(_request: Request, { params }: { params: Promise<{ id: string; n: string }> }) {
  const { id, n } = await params;
  if (!/^\d+$/.test(n)) return new Response("Bad frame", { status: 400 });
  const at = gameVideoAt(id, Number(n));
  if (!at) return new Response("No video for this frame", { status: 404 });
  try {
    const jpeg = "jpeg" in at ? fs.readFileSync(at.jpeg) : await extractFileFrame(at.file, at.seconds, WIDTH);
    if (!jpeg) return new Response("ffmpeg is not available", { status: 404 });
    return new Response(new Uint8Array(jpeg), { headers: { ...HEADERS, "Content-Length": String(jpeg.length) } });
  } catch (error) {
    return new Response(String(error), { status: 500 });
  }
}
