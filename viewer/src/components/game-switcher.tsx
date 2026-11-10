"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

import { CHALLENGE_GAME } from "@/lib/games";

export interface SwitcherGroup {
  label: string;
  games: { id: string; label: string }[];
}

// The game picker on every page: the challenge match goes to its own viewer, every other game to
// /games/<id>. On a page that is not a game (the labelling tools, the list) nothing is selected.
export function GameSwitcher({ groups }: { groups: SwitcherGroup[] }) {
  const router = useRouter();
  const path = usePathname();
  const current = path.startsWith("/frame") ? CHALLENGE_GAME.id : (path.match(/^\/games\/([^/]+)/)?.[1] ?? "");

  return (
    <nav className="flex flex-wrap items-center justify-end gap-3 border-b border-zinc-900 px-4 py-1.5 text-xs text-zinc-500">
      <Link href="/games" className="hover:text-zinc-100 hover:underline">
        all games
      </Link>
      <label className="flex items-center gap-2 uppercase tracking-wide">
        Game
        <select
          className="max-w-[28rem] rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm normal-case tracking-normal text-zinc-100"
          value={current}
          onChange={(e) => router.push(e.target.value === CHALLENGE_GAME.id ? CHALLENGE_GAME.href : `/games/${e.target.value}`)}
        >
          {current === "" && (
            <option value="" disabled>
              Pick a game…
            </option>
          )}
          <optgroup label="Challenge">
            <option value={CHALLENGE_GAME.id}>{CHALLENGE_GAME.title}</option>
          </optgroup>
          {groups.map((g) => (
            <optgroup key={g.label} label={g.label}>
              {g.games.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
    </nav>
  );
}
