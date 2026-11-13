import type { BroadcastStatus, Label, PeriodKey, Side } from "./match";

export type BallSource = "tactical" | "broadcast"; // Metrica's ball in the tactical ATD, or in the broadcast ATD

export interface Player {
  slot: number; // 1-based ATD track slot -- not a shirt number
  team: Side;
  x: number; // normalised 0-1
  y: number;
  speed: number | null; // m/s; null means the tracker reported no speed (often a stale slot)
  vx: number | null; // normalised displacement per second, from the previous frames
  vy: number | null;
  distToBallM: number | null; // to Metrica's tactical ball, or the broadcast one where the tactical has none
}

export interface Ball {
  x: number;
  y: number;
}

export interface BallPoint extends Ball {
  ago: number; // frames before the current one (0 = now)
  source: BallSource;
}

export interface Tag {
  id: number;
  code: string;
  start: number; // video frame, inclusive
  end: number; // video frame, exclusive
  team: Side | null;
  teamName: string | null;
  labels: Record<string, string>;
}

/** A stoppage (ballcv/stoppages.py): how play stopped, the restart it then waits for, and when that came. */
export interface Stoppage {
  start: number;
  end: number; // inclusive: the last dead frame
  // stoppage whistle in the audio; a tagged set piece; players setting one up (one at a corner, or off the pitch over a
  // touchline) where nothing else says play stopped; or an assistant's flag, restarted before the next set piece's out
  // (an offside or a foul: a free kick)
  cause: "whistle" | "tag" | "set-up" | "flag";
  waits_for: "free kick" | "throw-in" | "goal kick" | "corner" | "kick-off" | "restart"; // restart: a lone flag's, not seen
  restart: number | null; // the restart frame, or null when it was not seen (ended when the players resumed)
  ended_by: "restart" | "players resume";
  tag: number | null; // the set-piece tag's frame, or where the set-up was seen
  // what put the start where it is: the whistle; or, before a tag, the assistant's flag, the ball leaving the pitch,
  // or the fastest players easing off
  timed_by: "whistle" | "assistant flag" | "ball exit" | "speed drop";
  restart_team: Side | null; // who takes the set piece: named on the tag, else the player at the ball
}

/** An assistant referee's flag signalling (ballcv/flag_self.py): level or above from the shoulder -- a goal kick (level), a
 * throw-in (up and out), offside (straight up) -- for two samples or more. */
export interface AssistantFlag {
  start: number;
  end: number; // the last sample it was seen up
  line: "far" | "near";
  x: number; // where along the touchline, m
}

/** An assistant referee where the model finds them (ballcv/assistant_net.py), at the nearest sample. */
export interface Assistant {
  line: "far" | "near";
  x: number; // along the touchline, m
  score: number;
  flagUp: boolean | null; // signalling (level or above); null: the flag was not seen in this sample
}

export interface Run<T extends string = string> {
  start: number;
  end: number; // inclusive
  value: T;
}

export interface HomographyData {
  m: number[];
  reprojectionError: number;
  valid: boolean;
}

/** The broadcast camera at the same instant as the tactical frame: a secondary check, plus a source of
 * ball positions where the tactical tracking has none. */
export interface BroadcastData {
  frame: number | null; // broadcast video frame; null when the broadcast has nothing at this instant
  status: BroadcastStatus;
  // Tracking is only passed on for live frames: a replay shows an earlier moment, so its data is
  // about the wrong time and is disregarded.
  players: { slot: number; team: Side; x: number; y: number }[];
  homography: HomographyData | null;
  runs: Run<Exclude<BroadcastStatus, "none">>[];
}

/** A referee's whistle heard in the broadcast audio, written by alignment/detect_whistles.py. */
export interface Whistle {
  start: number; // tactical frame, inclusive
  end: number; // tactical frame, inclusive
  kind: "live" | "replay" | "wipe" | "break"; // the broadcast picture when it was heard
  score: number; // dB; ~12 is the detection threshold, 20+ is a clear whistle
  hz: number; // pitch of the strongest tone
  // From player speed around it: stoppage = players slow down, restart = they speed up.
  role: "stoppage" | "restart" | "unclear";
}


export interface FrameData {
  frame: number;
  period: { key: PeriodKey; label: string; graded: boolean };
  teams: Record<Side, string>;
  teamSource: "xml" | "inferred";
  players: Player[];
  // Metrica's ball from the tactical ATD: where the tactical camera sees it against the grass (exact on the ground)
  ballInUse: (Ball & { source: BallSource }) | null;
  ballLastSeen: BallPoint | null; // only when the tactical ATD has no ball on this frame
  ballTrail: BallPoint[];
  // Metrica's ball from the broadcast ATD at the same instant (live broadcast frames only): the zoomed-in camera's
  // view of the same ball; off from the tactical one when the ball is in the air
  broadcastBall: (Ball & { source: BallSource }) | null;
  broadcastBallTrail: BallPoint[];
  ballGapM: number | null; // between the two, when both have the ball
  nearest: { slot: number; team: Side; distM: number } | null;
  homography: HomographyData;
  broadcast: BroadcastData | null; // null when the broadcast data or alignment is not available
  whistles: Whistle[] | null; // the whole match; null when the whistle CSV has not been generated
  tagsActive: Tag[];
  window: { start: number; end: number };
  tagsWindow: Tag[];
  gt: Label;
  gtRuns: Run<Label>[];
  touches: TouchData | null; // our own ball track + touches (null when the touch file has not been generated)
  ownTracks: OwnTrack[]; // our ball track (self-trained ball detector, ownball.py link), when its file exists
  pred: Label | null; // our possession label (possession/touch_possession.py public); null when no prediction file exists
  predRuns: Run<Label>[] | null;
  oldRuleRuns: Run<Label>[] | null; // possession/predict.py, the control rule; null when it has not been run
  stoppages: Stoppage[] | null; // in the ±30 s window; null when ballcv/stoppages.py has not been run
  assistants: Assistant[] | null; // null when ballcv/viewer_export.py has not been run
  assistantFlags: AssistantFlag[] | null; // in the ±30 s window; null when ballcv/viewer_export.py has not been run
}

/** A break in our own ball track: a player's touch, a bounce (velocity flips up the vertical while falling), or a
 * deflection (a motion change with nobody in reach). */
export interface TouchEvent {
  f: number;
  kind: "touch" | "bounce" | "deflection";
  slot?: number | null; // touch: the Metrica track slot whose body the ball was at; null: a player Metrica lost
  lost?: boolean; // touch by a player the player detector sees and Metrica has no track for (ballcv/player_det.py)
  team?: Side;
  body?: "foot" | "body" | "head";
  dist_m?: number | null; // ball to that player's body (touch) or to the nearest body (bounce)
  nearest?: number | null; // bounce: nearest slot
  conf?: "break" | "appears" | "disappears"; // break = the ball's motion changed; appears/disappears = track edge at a body
  dv_ms?: number | null; // velocity change, m/s at the ball's ground scale
  // touch only, controlled possession: an opponent's contact changes possession only if his team's next touch is
  // theirs again (or, when no touch is seen, the ball goes to them)
  role?: "control" | "interception" | "block" | "keeper claim";
  claim?: boolean; // keeper claim: the ball vanished and was last / next seen with a goalkeeper in his box
  how?: string;
}

/** A touch's player: "#slot", or "untracked" for a player Metrica lost. */
export const touchSlot = (e: TouchEvent): string => (e.slot != null ? `#${e.slot}` : "untracked");

/** The ball played between two consecutive touches. */
export interface BallPlay {
  f0: number;
  f1: number;
  from_slot: number;
  from_team: Side;
  to_slot: number;
  to_team: Side;
  kind: string; // dribble | ground pass | lofted ball | long ground pass | long lofted ball
  outcome: "kept" | "completed" | "lost";
  length_m: number | null;
  max_h: number; // rough peak height, m
  bounces?: number;
}

/** A stretch of the ball's path between breaks, fitted with p0 + v t + a t^2 / 2 in stabilised pixels. */
export interface BallStretch {
  f0: number;
  f1: number;
  kind: "flight" | "roll" | "unclear"; // flight: accelerates down at ~gravity; roll: barely accelerates
  g_frac: number; // downward image acceleration as a fraction of gravity
}

/** One of our own ball tracks (ownball.py link, from a ball detector's candidates), on the grass in normalised pitch
 * coordinates: each frame's pixel ball through that frame's homography, so trails stay put as the camera moves. */
export type OwnTrackKey = "ball";
export interface OwnTrack {
  key: OwnTrackKey;
  label: string;
  ball: Ball | null; // null on frames the track has no ball
  trail: (Ball & { ago: number })[]; // the last two seconds, the current frame included (ago 0)
}

export interface TouchData {
  covered: { start: number; end: number }; // frames the touch file was computed for
  ownBall: [number, number] | null; // our ball, video pixels
  lastTouch: TouchEvent | null;
  nextTouch: TouchEvent | null;
  play: BallPlay | null; // the ball played that is under way on this frame
  events: TouchEvent[]; // in the ±30 s window
  plays: BallPlay[]; // in the ±30 s window
  stretches: BallStretch[]; // in the ±30 s window
}
