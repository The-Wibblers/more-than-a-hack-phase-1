import type { Metadata } from "next";
import { redirect } from "next/navigation";

import { FrameBridge } from "@/components/viewer-context";
import { clampFrame } from "@/lib/match";
import { MissingDataError, getFrame } from "@/lib/server/data";

interface Props {
  params: Promise<{ n: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { n } = await params;
  return { title: `Frame ${n} · BAR–ATH tactical` };
}

export default async function FramePage({ params }: Props) {
  const { n } = await params;
  const parsed = Number(n);
  if (!Number.isFinite(parsed)) redirect("/frame/11185");
  const frame = clampFrame(parsed);
  if (String(frame) !== n) redirect(`/frame/${frame}`);

  try {
    return <FrameBridge data={getFrame(frame)} />;
  } catch (error) {
    if (error instanceof MissingDataError) {
      return (
        <div className="fixed inset-0 z-50 grid place-items-center bg-zinc-950 p-8 text-center text-zinc-300">
          <div>
            <p className="text-lg font-semibold text-zinc-100">Match data not found</p>
            <p className="mt-2 font-mono text-sm">{error.message}</p>
            <p className="mt-2 text-sm">Unpack the tactical bundle into data/tactical/, or set MTAH_DATA_DIR.</p>
          </div>
        </div>
      );
    }
    throw error;
  }
}
