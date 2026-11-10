"""Stoppages of the whole match: when play stopped, which restart it then waits for, and when that restart happened.
possession/predict.py reads them for its dead-ball judgement.

That play stopped, and how it restarts: the match's set-piece tags (data/tactical/..._smart_tagging.xml: throw-ins,
corners, goal kicks, free kicks, kick-offs after a goal, with the team taking it where named); a stoppage whistle in
the audio (a free kick); players setting up a
restart where nothing else says play stopped (one standing at a corner: a corner; off the pitch over a touchline: a
throw-in, a substitution, an injury seen to -- players leave the pitch only while the ball is dead).

When it stopped: the whistle; the flag; else, for a tagged or set-up restart, the first out before it (from OUT_S
before to TAG_LEAD_S after): the assistant's flag, or our ball leaving the pitch -- the flag if raised just after;
not where the ball is seen flying with no arc found for it (ball3d.py: the ground point of a ball in the air can lie
past the far touchline or a goal line over the pitch) -- else when the fastest players eased off.

When it restarted: where the ball is reliable -- a goal kick or a corner, the placed ball struck in its goal area or at
its corner arc; a whistle's free kick, on the referee's restart whistle (the kick after it). Everywhere else the players
say it: their pace rises out of its lull just after the restart (RESUME_LAG; the whistle rule's resume).
A restart puts the ball in play: if play stops again within PLACED_S of it, it is the same stoppage. The team named on
the tag takes the set piece; otherwise the team of the player at the ball when it restarts.

  python stoppages.py      -> data/ballcv/BAR-ATH_stoppages.json
"""
import json, os, sys
from collections import Counter
import numpy as np

import ownball as ob
import ball3d
from ownball import L, W, CONTACT_M, BREAK_MS, PLACED_S

sys.path[:0] = [os.path.join(__import__('paths').REPO, 'possession'), __import__('paths').REPO]
import predict

OUT_JSON = os.environ.get('STOPPAGES_OUT') or __import__('paths').DATA + 'ballcv/BAR-ATH_stoppages.json'
TAGS = __import__('paths').DATA + 'tactical/BAR-ATH_tactical_smart_tagging.xml'
TAG_KIND = {'Throw-in': 'throw-in', 'Corner Kick': 'corner', 'Goal Kick': 'goal kick', 'Free Kick': 'free kick',
            'Kick Off (After Goal)': 'kick-off'}
TAG_TEAM = {'Barcelona': 'H', 'Atletic Club': 'A'}
OUT_S = 20                 # an out is tagged within this many seconds of it
TAG_LEAD_S = 5             # ... or up to this long before it
FLAG_LAG_S = 2             # the assistant raises the flag within this long of the ball going out
DROP_S = 3                 # the fastest players' speed is compared over this long before and after
FASTEST = 3                # ... the mean of the fastest this many
RESUME_LAG = 23             # the players' pace rises this many frames (0.9 s) after the restart (measured on the
                           # ground truth's restarts and the verified throw-ins)
CORNER_SETUP_M, CORNER_SETUP_S = 2.5, 1   # a player this close to a corner for this long: a corner being taken
OFF_PITCH_M, OFF_PITCH_S = 0.5, 2         # a player this far off the pitch over a touchline for this long: a throw-in
SETUP_CLEAR_S = 30         # a set-up this close to a known stoppage is part of it; none this close to a half's end or
                           # after a kick-off (players leave the pitch, the kick-off is set up)
GOAL_AREA = (5.5, 9.16)    # goal area: depth, half-width either side of the centre line (m) -- pitch markings
CORNER_M = 1.0             # corner arc radius (m)
MARGIN_M = 1.0             # position error allowed around those markings


def ground_positions(M, track):
    fr = np.array(sorted(track)); uv = np.array([track[f][:2] for f in fr])
    q = np.einsum('nij,nj->ni', np.linalg.inv(M[fr]), np.c_[uv, np.ones(len(fr))])
    return fr, q[:, :2] / q[:, 2:] * [L, W]


def tagged_outs():
    """[(frame, restart kind, team 'H' | 'A' | None)] for every tagged set piece (but the kick-offs that start a half)."""
    import xml.etree.ElementTree as ET
    out = []
    for i in ET.parse(TAGS).getroot().iter('instance'):
        if i.findtext('code') != 'SET PIECES':
            continue
        lab = {l.findtext('group'): l.findtext('text') for l in i.findall('label')}
        if lab.get('Type') in TAG_KIND:
            out.append((round(float(i.findtext('start')) * 25), TAG_KIND[lab['Type']], TAG_TEAM.get(lab.get('Team'))))
    return sorted(out)


def setups(P, V, known, halves, kickoffs):
    """[(frame, restart kind)]: players setting up a restart where nothing else says play stopped -- a player standing
    at a corner (a corner), or off the pitch over a touchline (a throw-in; or a substitution or an injury being seen
    to, also dead ball). Not within SETUP_CLEAR_S of a known stoppage (known), of a half's end or after a kick-off."""
    seen = np.isfinite(V)
    near = lambda a, b: np.minimum(a, b)
    at_corner = seen & (np.hypot(near(P[..., 0], L - P[..., 0]), near(P[..., 1], W - P[..., 1])) < CORNER_SETUP_M)
    off_pitch = seen & (near(P[..., 1], W - P[..., 1]) < -OFF_PITCH_M) & (near(P[..., 0], L - P[..., 0]) > 0)
    clear = SETUP_CLEAR_S * 25
    out = []
    for kind, m, hold in (('corner', at_corner, CORNER_SETUP_S), ('throw-in', off_pitch, OFF_PITCH_S)):
        run = np.zeros(m.shape[1], np.int32); prev = False
        for f in range(len(m)):
            run = np.where(m[f], run + 1, 0); now = bool((run >= hold * 25).any())
            if now and not prev:
                half = next(((a, b) for a, b in halves if a <= f <= b), None)
                if (half and f < half[1] - clear and not any(0 <= f - k < clear for k in kickoffs)
                        and not any(abs(f - k) < clear for k in known) and not any(abs(f - o) < clear for o, _ in out)):
                    out.append((f, kind))
            prev = now
    return sorted(out)


def out_moment(tag, after, fastest, flag_ups, exits):
    """(frame, how) play stopped for a tagged (or set-up) restart, not before frame after: of the assistant's flags and
    the ball's (sure) exits from OUT_S before the tag to TAG_LEAD_S after it (a tag can come just before the out), the
    first -- after it the ball is dead, and moved about off the pitch -- and if the assistant raised the flag within
    FLAG_LAG_S after that exit, the flag (the tag says it went out; the ball only says when). With neither, the largest drop in the fastest players' speed before the tag."""
    lo, hi = max(after, tag - OUT_S * 25), tag + TAG_LEAD_S * 25
    if lo > hi:
        return after - 1, 'within'                                            # inside the stoppage before it
    seen = [(f, f, 'assistant flag') for f in flag_ups if lo <= f <= hi] + [(f, f, 'ball exit') for f in exits if lo <= f <= hi]
    if seen:
        _, f, how = min(seen)                                                 # the first: after it the ball is dead
        if how == 'ball exit':
            flag = [g for g in flag_ups if f <= g <= f + FLAG_LAG_S * 25]
            if flag:
                return flag[0], 'assistant flag'
        return f, how
    if lo > tag:
        return after - 1, 'within'
    k = DROP_S * 25
    cs = np.r_[0, np.cumsum(np.nan_to_num(fastest))]
    mean = lambda a, b: (cs[b] - cs[a]) / (b - a)
    return max(range(lo, tag + 1), key=lambda f: mean(f - k, f) - mean(f, f + k)), 'speed drop'


def cmd_build():
    from cameras import load_h
    M, _, _ = load_h('tactical')
    track = {int(k): np.array(v) for k, v in json.load(open(ball3d.FULL + '/track.json')).items()}
    fr, G = ground_positions(M, track)
    idx = {int(f): i for i, f in enumerate(fr)}
    team_of_track, metrica, players, dead, stops = predict.match_state()
    _, raw = predict.require(['atd_meta', 'atd_raw'], predict.GAME)
    _, _, speeds = predict.read_atd(raw, len(dead), len(team_of_track))
    speed = predict.player_speed(speeds)
    halves = predict.half_video_blocks(predict.GAME)
    half_of = lambda f: next(((a, b) for _h, a, b in halves if a <= f <= b), None)

    def resumed(start, limit):
        """first frame the players' median speed rises RESUME_RISE above its low since start (the whistle rule)."""
        low = np.inf
        for f in range(start, min(limit, start + predict.WHISTLE_CAP)):
            low = min(low, speed[f])
            if f > start + predict.SPEED_HALF and speed[f] > low + predict.RESUME_RISE:
                return f
        return min(limit, start + predict.WHISTLE_CAP)

    # placed balls struck: still for PLACED_S, then off at BREAK_MS or more
    need = int(PLACED_S * 25)
    strikes = []
    for i, f in enumerate(fr):
        prev = [j for j in range(max(0, i - need), i) if fr[j] >= f - need]
        nxt = [j for j in range(i + 1, min(len(fr), i + 6)) if fr[j] <= f + 5]
        if len(prev) < need / 2 or not nxt:
            continue
        P0 = G[prev]
        if np.max(np.linalg.norm(P0 - np.median(P0, 0), axis=1)) >= CONTACT_M:
            continue
        j = nxt[-1]
        if np.linalg.norm(G[j] - G[i - 1]) / (fr[j] - fr[i - 1]) * 25 >= BREAK_MS:
            strikes.append(int(f))
    strikes = sorted(set(s for k_, s in enumerate(strikes) if k_ == 0 or s - strikes[k_ - 1] > need))

    P = np.load(ob.S + 'tactical.npz')['P'][..., :2]
    ob.STALE = ob.stale_mask(P)
    whistles = sorted(int(w) for w in stops if half_of(w))
    _stops, restart_whistles = predict.read_whistles(predict.WHISTLES_CSV.format(game=predict.GAME))
    restart_whistles = sorted(int(r) for r in restart_whistles)
    tac = np.load(ob.S + 'tactical.npz')['P']; spd = tac[..., 2]
    fastest = np.nanmean(-np.sort(-np.nan_to_num(spd, nan=-1), 1)[:, :FASTEST], 1)
    import flag_self
    raised = json.load(open(flag_self.RAISES))['raises']                 # the flag model's raises (flag_self.py raises)
    flag_ups = sorted(r['start'] for r in raised)
    flag_down = {r['start']: r['end'] for r in raised}                   # the flag's last frame up
    # the ball leaving the pitch: seen inside, then past a line by OUT_M on two tracked frames in a row -- but not where
    # ownball sees the ball flying (its image falls at gravity) and ball3d.py found no arc for it: the ground point of a
    # ball in the air can lie past the far touchline or a goal line while the ball is over the pitch. (Over the near
    # touchline a ball in the air only reads further out -- the camera is behind it.)
    arc = np.load(ball3d.OUT)['kind'] >= 1
    flying = np.zeros(len(arc), bool)
    for sg in json.load(open(ob.OUT_JSON))['segments']:
        if sg['kind'] == 'flight':
            flying[sg['f0']:sg['f1'] + 1] = True
    past = -np.minimum.reduce([G[:, 0], L - G[:, 0], G[:, 1], W - G[:, 1]])
    # ... unless the assistant signals just after it (FLAG_LAG_S): they saw it go out
    signalling = np.zeros(len(arc), bool)
    for r in raised:
        signalling[r['start']:r['end'] + 1] = True
    confirmed = np.array([signalling[f:f + FLAG_LAG_S * 25 + 1].any() for f in fr])
    sure = arc[fr] | ~flying[fr] | (G[:, 1] > W) | confirmed
    exits, was_in = [], False
    for i in range(1, len(fr)):
        if past[i] < 0:
            was_in = True
        elif (was_in and past[i - 1] > predict.OUT_M and past[i] > predict.OUT_M and fr[i] - fr[i - 1] <= 3
              and sure[i - 1] and sure[i]):
            exits.append(int(fr[i - 1])); was_in = False
    tags = [t for t in tagged_outs() if half_of(t[0])]
    kickoffs = [a for _h, a, _b in halves] + [t for t, k, _ in tags if k == 'kick-off']
    found = setups(tac[..., :2] * [L, W], spd, [t for t, _, _ in tags] + whistles, [(a, b) for _h, a, b in halves], kickoffs)
    print(f'{len(found)} restart set-ups where nothing else says play stopped: {dict(Counter(k for _, k in found))}')
    def restart_of(kind, s, earliest, limit, rw):
        """the restart we can see: a goal kick or a corner, the placed ball struck in its goal area or at its corner arc;
        a stoppage whistle's free kick, on the referee's restart whistle (the kick after it). Else None: the players
        resuming ends it."""
        if kind in ('goal kick', 'corner'):
            for f in strikes:
                if earliest <= f <= limit:
                    x, y = G[idx[f]]
                    dx = min(x, L - x)
                    if kind == 'goal kick' and dx <= GOAL_AREA[0] + MARGIN_M and abs(y - W / 2) <= GOAL_AREA[1] + MARGIN_M:
                        return f
                    if kind == 'corner' and dx <= CORNER_M + MARGIN_M and min(y, W - y) <= CORNER_M + MARGIN_M:
                        return f
            return None
        if rw is not None and rw <= limit:                                    # on the referee's whistle: the kick after it
            return next((f for f in strikes if rw <= f <= limit), rw)
        return None

    # every stoppage in time order, each from the evidence after the one before it ended: a stoppage whistle; or a
    # tagged set piece / a restart being set up, which began at the assistant's flag, else the ball leaving the pitch,
    # else the players easing off, in the OUT_S before it
    # the flag only confirms: it times a stoppage something else found, or confirms the ball's exit -- a raise alone
    # is not a stoppage (the flag model's raises in live play are mostly false)
    events = sorted([(w, 'whistle', 'free kick', None) for w in whistles] + [(t, 'tag', k, tm) for t, k, tm in tags]
                    + [(f, 'set-up', k, None) for f, k in found])
    out, prev_end = [], -1
    for anchor, cause, kind, team in events:
        while True:
            if cause == 'whistle':
                s, how = anchor, 'whistle'
            else:
                s, how = out_moment(anchor, prev_end + 1, fastest, flag_ups, exits)
            if s <= prev_end:
                break                                                         # part of the stoppage already open
            a, b = half_of(s)
            rw = next((r for r in restart_whistles if r > s), None)           # the referee restarts play on it
            limit = min(b, s + predict.WHISTLE_CAP, next((w for w in whistles if w > s), b), rw + 25 * 3 if rw else b)
            # the restart comes after the out: after the assistant lowers the flag, and not within PLACED_S
            earliest = max(s + int(PLACED_S * 25), flag_down.get(s, s) + 1 if how == 'assistant flag' else 0)
            restart = restart_of(kind, s, earliest, limit, rw); ended = 'restart'
            if restart is None:                                               # the players resume (they react just after)
                restart, ended = max(earliest, resumed(s, limit) - RESUME_LAG), 'players resume'
            out.append(dict(start=int(s), end=int(restart - 1), cause=cause, waits_for='restart' if kind == 'any' else kind,
                            restart=int(restart), ended_by=ended, tag=None if cause == 'whistle' else anchor,
                            timed_by=how, tag_team=team))
            prev_end = restart - 1
            break
    # a restart must put the ball in play: play stopping again within PLACED_S means it was the same stoppage
    joined = []
    for st in out:
        if joined and st['start'] - joined[-1]['end'] - 1 < PLACED_S * 25:
            joined[-1] = dict(st, start=joined[-1]['start'])
        else:
            joined.append(st)
    out = joined
    # who takes the set piece: the player within reach of the ball when it restarts
    team = ['H'] * 24 + ['A'] * 22
    keepers = ob.keeper_slots()
    sides = ob.sides_by_half(P)
    for st in out:
        r, st['restart_team'] = st['restart'], st.pop('tag_team')
        if st['restart_team']:
            continue                                                          # the tag names the team
        g0 = None if r is None else next((g for d in range(4) for g in (r - d, r + d) if g in track), None)
        if g0 is not None:
            near = ob.bodies(M, P, g0, track[g0][:2])
            if near and near[0][0] <= ob.MAX_REACH_M:
                k = near[0][1]
                st['restart_team'] = ob.keeper_team(P, g0, k, sides) if k in keepers else team[k]
    json.dump(dict(stoppages=out, strikes=strikes), open(OUT_JSON, 'w'), indent=1)
    print(f'{len(out)} stoppages {dict(Counter(s["waits_for"] for s in out))}, ended by {dict(Counter(s["ended_by"] for s in out))} '
          f'| {len(strikes)} placed balls struck -> {OUT_JSON}')


if __name__ == '__main__':
    cmd_build()
