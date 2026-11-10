import { PITCH_M } from "./match";

export type Pt = [number, number];

/** Normalised pitch (0-1) -> image pixels through a row-major 3x3 homography. Null behind the camera. */
export function project(m: number[], x: number, y: number): Pt | null {
  const d = m[6] * x + m[7] * y + m[8];
  if (!(d > 1e-9)) return null;
  return [(m[0] * x + m[1] * y + m[2]) / d, (m[3] * x + m[4] * y + m[5]) / d];
}

/** Image pixels -> normalised pitch (0-1) through the inverse of the same homography: the point on the grass. */
export function unproject(m: number[], u: number, v: number): Pt | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  // inverse = adjugate / determinant; the determinant cancels in the division below, only its sign matters
  const inv = [e * i - f * h, c * h - b * i, b * f - c * e, f * g - d * i, a * i - c * g, c * d - a * f, d * h - e * g, b * g - a * h, a * e - b * d];
  const det = a * inv[0] + b * inv[3] + c * inv[6];
  if (Math.abs(det) < 1e-12) return null;
  const w = (inv[6] * u + inv[7] * v + inv[8]) / det;
  if (!(Math.abs(w) > 1e-12)) return null;
  return [(inv[0] * u + inv[1] * v + inv[2]) / det / w, (inv[3] * u + inv[4] * v + inv[5]) / det / w];
}

const boxW = 40.32;
const sixW = 18.32;
// Penalty arc: the part of the 9.15 m circle around the spot that lies outside the box.
const arcAngle = Math.acos((16.5 - 11) / 9.15);

/** Pitch markings of an L x W m pitch as polylines in normalised coordinates, built in metres. */
export function pitchMarkings(L: number, W: number): { lines: Pt[][]; spots: Pt[] } {
  const n = ([mx, my]: Pt): Pt => [mx / L, my / W];
  const rect = (x0: number, y0: number, x1: number, y1: number): Pt[] =>
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]].map((p) => n(p as Pt));
  const arc = (cx: number, cy: number, r: number, a0: number, a1: number, steps = 48): Pt[] =>
    Array.from({ length: steps + 1 }, (_, i) => {
      const a = a0 + ((a1 - a0) * i) / steps;
      return n([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
    });
  return {
    lines: [
      rect(0, 0, L, W),
      [n([L / 2, 0]), n([L / 2, W])],
      arc(L / 2, W / 2, 9.15, 0, 2 * Math.PI, 72),
      rect(0, (W - boxW) / 2, 16.5, (W + boxW) / 2),
      rect(L - 16.5, (W - boxW) / 2, L, (W + boxW) / 2),
      rect(0, (W - sixW) / 2, 5.5, (W + sixW) / 2),
      rect(L - 5.5, (W - sixW) / 2, L, (W + sixW) / 2),
      arc(11, W / 2, 9.15, -arcAngle, arcAngle),
      arc(L - 11, W / 2, 9.15, Math.PI - arcAngle, Math.PI + arcAngle),
    ],
    spots: [n([L / 2, W / 2]), n([11, W / 2]), n([L - 11, W / 2])],
  };
}

const CHALLENGE_PITCH = pitchMarkings(PITCH_M.length, PITCH_M.width);
export const PITCH_LINES: Pt[][] = CHALLENGE_PITCH.lines;
export const PITCH_SPOTS: Pt[] = CHALLENGE_PITCH.spots;
