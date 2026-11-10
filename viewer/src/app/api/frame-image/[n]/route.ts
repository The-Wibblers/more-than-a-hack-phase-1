import { BROADCAST_VIDEO_FRAMES, LAST_FRAME } from "@/lib/match";
import type { Source } from "@/lib/server/data";
import { extractFrame } from "@/lib/server/ffmpeg";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const LAST: Record<Source, number> = { tactical: LAST_FRAME, broadcast: BROADCAST_VIDEO_FRAMES - 1 };

// GET /api/frame-image/<n>[?source=broadcast] -> JPEG of video frame n, extracted with ffmpeg.
export async function GET(request: Request, { params }: { params: Promise<{ n: string }> }) {
  const { n } = await params;
  const source = (new URL(request.url).searchParams.get("source") ?? "tactical") as Source;
  if (!(source in LAST)) return new Response("Unknown source", { status: 400 });
  const frame = Number(n);
  if (!/^\d+$/.test(n) || frame > LAST[source]) return new Response("Bad frame", { status: 400 });

  try {
    const jpeg = await extractFrame(frame, source);
    if (!jpeg) return new Response(`ffmpeg or the ${source} video is not available`, { status: 404 });
    return new Response(new Uint8Array(jpeg), {
      headers: {
        "Content-Type": "image/jpeg",
        "Content-Length": String(jpeg.length),
        "Cache-Control": "private, max-age=31536000, immutable",
        "Content-Disposition": `inline; filename="BAR-ATH_${source}_${frame}.jpg"`,
      },
    });
  } catch (error) {
    return new Response(String(error), { status: 500 });
  }
}
