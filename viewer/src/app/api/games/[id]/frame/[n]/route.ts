import { gameFrame } from "@/lib/server/games";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// GET /api/games/<id>/frame/<n> -> players, ball and its trail, state and boxes at frame n.
export async function GET(_request: Request, { params }: { params: Promise<{ id: string; n: string }> }) {
  const { id, n } = await params;
  if (!/^\d+$/.test(n)) return new Response("Bad frame", { status: 400 });
  const frame = gameFrame(id, Number(n));
  if (!frame) return new Response("Unknown game", { status: 404 });
  return Response.json(frame, { headers: { "Cache-Control": "private, max-age=3600" } });
}
