import { redirect } from "next/navigation";

import { FIRST_HALF_KICKOFF } from "@/lib/match";

export default function Home() {
  redirect(`/frame/${FIRST_HALF_KICKOFF}`);
}
