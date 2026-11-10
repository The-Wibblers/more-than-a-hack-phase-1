import type { Metadata } from "next";
import Link from "next/link";

import { GameViewer } from "@/components/games/game-viewer";
import { clampGameFrame, gameMeta } from "@/lib/server/games";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ f?: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  return { title: gameMeta(id)?.meta.title ?? id };
}

export default async function GamePage({ params, searchParams }: Props) {
  const { id } = await params;
  const { f } = await searchParams;
  const game = gameMeta(id);
  if (!game) {
    return (
      <main className="grid min-h-screen place-items-center p-8 text-center text-zinc-300">
        <div>
          <p className="text-lg font-semibold text-zinc-100">No game “{id}”</p>
          <p className="mt-2 text-sm">
            Build it with <code className="font-mono">python -m external.build</code>, or pick one from{" "}
            <Link href="/games" className="underline">
              the list
            </Link>
            .
          </p>
        </div>
      </main>
    );
  }
  const { meta, events } = game;
  const start = f !== undefined && /^\d+$/.test(f) ? clampGameFrame(meta, Number(f)) : meta.periods[0].start;
  return <GameViewer key={meta.id} meta={meta} events={events} initialFrame={start} />;
}
