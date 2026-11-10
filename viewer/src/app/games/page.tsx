import type { Metadata } from "next";
import Link from "next/link";

import { CHALLENGE_GAME } from "@/lib/games";
import { gameIndex } from "@/lib/server/games";

export const metadata: Metadata = { title: "All games" };
export const dynamic = "force-dynamic";

const HAS = ["tracking", "ball", "events", "possession", "alive", "video"] as const;

export default function GamesPage() {
  const index = gameIndex();
  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6">
      <header>
        <h1 className="text-xl font-semibold text-zinc-100">All games</h1>
        <p className="mt-1 text-sm text-zinc-400">
          The challenge match, plus every open dataset built with <code className="font-mono text-zinc-300">python -m external.build</code>. Each one shows
          whatever its source has.
        </p>
      </header>

      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">Challenge</h2>
        <Link href={CHALLENGE_GAME.href} className="block rounded-lg border border-zinc-700 bg-zinc-900/60 px-4 py-3 hover:bg-zinc-800">
          <div className="font-semibold text-zinc-100">{CHALLENGE_GAME.title}</div>
          <div className="text-xs text-zinc-400">{CHALLENGE_GAME.competition} · tactical + broadcast video, ATD, homography, Smart Tagging, our ball and touches</div>
        </Link>
      </section>

      {Object.entries(index.datasets).map(([key, d]) => {
        const games = index.games.filter((g) => g.dataset === key);
        return (
          <section key={key}>
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                {d.name} <span className="font-normal normal-case text-zinc-500">· {d.licence} · {d.host}</span>
              </h2>
              <a href={d.url} target="_blank" rel="noreferrer" className="text-xs text-zinc-500 hover:text-zinc-200 hover:underline">
                source
              </a>
            </div>
            <p className="mb-2 text-xs text-zinc-500">{d.contents}</p>
            {games.length === 0 ? (
              <div className="rounded-lg border border-dashed border-zinc-800 px-4 py-3 text-sm text-zinc-500">
                Not built yet: <code className="font-mono">python -m external.download {key}</code> then{" "}
                <code className="font-mono">python -m external.build {key}</code> ({d.size}).
              </div>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {games.map((g) => (
                  <Link key={g.id} href={`/games/${g.id}`} className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2 hover:bg-zinc-800">
                    <div className="font-medium text-zinc-100">
                      {g.title} {g.score && <span className="font-mono text-zinc-400">{g.score}</span>}
                    </div>
                    <div className="text-xs text-zinc-500">{[g.competition, g.date].filter(Boolean).join(" · ")}</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {HAS.map((h) => (
                        <span key={h} className={`rounded px-1.5 text-[11px] ${g.has[h] ? "bg-zinc-700 text-zinc-100" : "text-zinc-600 line-through"}`}>
                          {h}
                        </span>
                      ))}
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </section>
        );
      })}
    </main>
  );
}
