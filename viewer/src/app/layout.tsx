import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { GameSwitcher, type SwitcherGroup } from "@/components/game-switcher";
import { gameIndex } from "@/lib/server/games";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BAR–ATH tactical frame viewer",
  description: "Frame-by-frame view of tracking, homography, our own ball tracking and touches, and possession",
};

/** Every built game, by dataset, for the picker at the top of each page. */
function switcherGroups(): SwitcherGroup[] {
  const index = gameIndex();
  return Object.entries(index.datasets)
    .map(([key, d]) => ({
      label: `${d.name} · ${d.licence}`,
      games: index.games
        .filter((g) => g.dataset === key)
        .map((g) => ({ id: g.id, label: [g.title + (g.score ? ` ${g.score}` : ""), g.date, g.has.video ? "video" : null].filter(Boolean).join(" · ") })),
    }))
    .filter((g) => g.games.length);
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        <GameSwitcher groups={switcherGroups()} />
        {children}
      </body>
    </html>
  );
}
