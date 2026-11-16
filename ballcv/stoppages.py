"""Stoppages of the whole match: when play stopped, which restart it then waits for, and when that restart happened.
possession/predict.py reads them for its dead-ball judgement.

That play stopped, and how it restarts: the match's set-piece tags (data/tactical/..._smart_tagging.xml: throw-ins,
corners, goal kicks, free kicks, kick-offs after a goal, with the team taking it where named); a stoppage whistle in
the audio (a free kick); players setting up a
restart where nothing else says play stopped (one standing at a corner: a corner; off the pitch over a touchline: a
throw-in, a substitution, an injury seen to -- players leave the pitch only while the ball is dead).

When it stopped: a half whose tracking starts before its kick-off, from its first frame; where nobody ran for longer
than OUT_S before it (RUN_MS: live play has someone running), when the players stopped -- the referee's pip then (a
whistle too short to stand alone), else their pace dropping -- and with nothing else found there, such a stretch is a
stoppage where the broadcast was away on replays for longer than OUT_S inside it or the referee's restart whistle
ends it; a goal (tagged), from the
ball seen in the goal, else the players easing off over the goal's span, until the kick-off; the broadcast's half-time
break, to the end of the half; the whistle; the flag; a ball out over a touchline nobody tagged (a throw-in); else, for a tagged or set-up restart, the first out before it (from OUT_S
before to TAG_LEAD_S after): the assistant's flag, or our ball leaving the pitch (and staying off it) -- the ball when
it crossed, the flag only follows it; not where the ball is seen flying with no arc found for it (ball3d.py: the
ground point of a ball in the air can lie past the far touchline or a goal line over the pitch); a ball first seen out
after it vanished went out unseen, when the players eased off -- else when the fastest players eased off.

When it restarted: where the ball is reliable -- a goal kick or a corner, the placed ball struck in its goal area or at
its corner arc; a whistle's free kick, on the referee's restart whistle (the kick after it). Everywhere else the players
say it: their pace rises out of its lull just after the restart (RESUME_LAG; the whistle rule's resume) -- unless the
ball then comes to rest and is struck from there: that placed ball is the restart.
A restart puts the ball in play: if play stops again within PLACED_S of it, or before anyone ran (the fastest players
never up to a jog, RUN_MS: walking to a set piece while the ball is knocked about is not play), it is the same stoppage;
and if, before anyone runs after the players 'resume', they sink below the stoppage's own lull, that was not the
restart -- the one they then run from is. And a stoppage
the players never pause for (paused_in: their pace never dips by RESUME_RISE before they 'resume') was timed at the
wrong out: the ball cannot be dead and at once live again -- the next evidence after it times it. The team named on
the tag takes the set piece; otherwise the team of the player at the ball when it restarts.

  python stoppages.py      -> data/ballcv/BAR-ATH_stoppages.json
"""
import json, os, sys, warnings
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
BOX_HALF_W = 20.16         # the penalty area's half-width (m)
CORNER_M = 1.0             # corner arc radius (m)
RUN_MS = 3.0               # a jog (m/s): play has players running; walking, as while a set piece is set up, is slower
MARGIN_M = 1.0             # position error allowed around those markings


def tagged_outs():
    """[(frame, restart kind, team 'H' | 'A' | None)] for every tagged set piece (but the kick-offs that start a half:
    half_kickoffs)."""
    import xml.etree.ElementTree as ET
    out = []
    for i in ET.parse(TAGS).getroot().iter('instance'):
        if i.findtext('code') != 'SET PIECES':
            continue
        lab = {l.findtext('group'): l.findtext('text') for l in i.findall('label')}
        if lab.get('Type') in TAG_KIND:
            out.append((round(float(i.findtext('start')) * 25), TAG_KIND[lab['Type']], TAG_TEAM.get(lab.get('Team'))))
    return sorted(out)


def goal_spans():
    """[(start, end, scoring team 'H' | 'A' | None)] of the tagged goals ('GOALS': from the move's start to just after
    the kick-off that follows)."""
    import xml.etree.ElementTree as ET
    out = []
    for i in ET.parse(TAGS).getroot().iter('instance'):
        if i.findtext('code') == 'GOALS':
            lab = {l.findtext('group'): l.findtext('text') for l in i.findall('label')}
            out.append((round(float(i.findtext('start')) * 25), round(float(i.findtext('end')) * 25), TAG_TEAM.get(lab.get('Team'))))
    return sorted(out)


def half_kickoffs():
    """[(frame, team)] of the tagged kick-offs that start a half ('Kick Off (Start)')."""
    import xml.etree.ElementTree as ET
    out = []
    for i in ET.parse(TAGS).getroot().iter('instance'):
        lab = {l.findtext('group'): l.findtext('text') for l in i.findall('label')}
        if i.findtext('code') == 'SET PIECES' and lab.get('Type') == 'Kick Off (Start)':
            out.append((round(float(i.findtext('start')) * 25), TAG_TEAM.get(lab.get('Team'))))
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


LINE_OF = {'throw-in': 'touch', 'goal kick': 'goal', 'corner': 'goal'}   # the line the ball crossed for each restart


def out_moment(tag, after, fastest, flag_ups, exits, inside, last_break, kind, pips=()):
    """(frame, how) play stopped for a tagged (or set-up) restart, not before frame after: of the assistant's flags and
    the ball's (sure) exits seen from OUT_S before the tag to TAG_LEAD_S after it (a tag can come just before the out),
    the first -- after it the ball is dead, and moved about off the pitch; an exit is when the ball crossed, a flag only
    follows it. exits: [(frame seen, frame it went out, the line it crossed)] -- over the restart's own line (LINE_OF:
    a throw-in's touchline, a goal kick's or corner's goal line). An exit before that is the out too when the ball is not seen
    back on the pitch (inside: the frames it is) between it and then: it never came back. With neither, the ball was
    still in play at the last break in its motion we saw before the tag (last_break: a touch, bounce or deflection) --
    it went out after that; then the referee's pip (pips: his whistle too short to stand alone, detect_whistles.py --
    the tag says play stopped, the pip when), else the players easing off (the largest drop in the fastest players'
    speed), between then and the tag."""
    lo, hi = max(after, tag - OUT_S * 25), tag + TAG_LEAD_S * 25
    exits = [(f, d) for f, d, line in exits if LINE_OF.get(kind, line) == line]
    stayed = [(f, d) for f, d in exits if after <= f < lo and not ((inside > f) & (inside <= lo)).any()]
    lo = min([lo] + [f for f, _ in stayed])
    if lo > hi:
        return after - 1, 'within'                                            # inside the stoppage before it
    seen = [(f, f, 'assistant flag') for f in flag_ups if lo <= f <= hi] + [(f, d, 'ball exit') for f, d in exits if lo <= f <= hi]
    if seen:
        _, f, how = min(seen)                                                 # the first: after it the ball is dead
        return f, how
    if lo > tag:
        return after - 1, 'within'
    lo = max(lo, min(last_break, tag))
    pip = next((f for f in pips if lo <= f <= tag), None)              # the referee's pip: play stopped, he whistled
    return (pip, 'referee pip') if pip is not None else (eased_off(fastest, lo, tag), 'speed drop')


def paused_in(fastest, s, r):
    """Did the players pause between an out at s and play going on at r: the fastest players' pace (over DROP_S, as for
    a speed drop) falling at some point by RESUME_RISE below its level over the DROP_S before s?"""
    k = DROP_S * 25
    cs = np.r_[0, np.cumsum(np.nan_to_num(fastest))]
    mean = lambda a, b: (cs[b] - cs[a]) / (b - a)
    low = min(mean(f, f + k) for f in range(s, max(s + 1, r - k + 1)))
    return mean(max(0, s - k), s) - low >= predict.RESUME_RISE


def eased_off(fastest, lo, hi):
    """the frame in lo..hi where the fastest players' speed drops the most (its mean over DROP_S before against after)."""
    k = DROP_S * 25
    cs = np.r_[0, np.cumsum(np.nan_to_num(fastest))]
    mean = lambda a, b: (cs[b] - cs[a]) / (b - a)
    return max(range(lo, hi + 1), key=lambda f: mean(f - k, f) - mean(f, f + k))


def cmd_build():
    from cameras import load_h
    M, _, _ = load_h('tactical')
    track = {int(k): np.array(v) for k, v in json.load(open(ball3d.FULL + '/track.json')).items()}
    fr, G = ob.ground_positions(M, track)
    idx = {int(f): i for i, f in enumerate(fr)}
    mball, _, speeds = predict.match_atd()
    stops, _ = predict.read_whistles(predict.WHISTLES_CSV.format(game=predict.GAME))          # stoppage whistles
    pips_csv = predict.WHISTLES_CSV.format(game=predict.GAME).replace('_whistles.csv', '_whistle_pips.csv')
    import pandas as pd
    pips = sorted(pd.read_csv(pips_csv).tactical_frame.astype(int).tolist()) if os.path.exists(pips_csv) else []
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

    # placed balls struck: still for PLACED_S (placed, or held while the taker waits), then off at BREAK_MS or more
    # within PLACED_S -- timed at the first frame it moves that fast. The rest is the second before it starts to move:
    # a kick takes the ball up to its speed over several frames of the track, and a held ball is put down first
    need = int(PLACED_S * 25)
    still = lambda i: (lambda w: len(w) >= need / 2 and np.max(np.linalg.norm(G[w] - np.median(G[w], 0), axis=1)) < CONTACT_M)(
        [j for j in range(max(0, i - need + 1), i + 1) if fr[j] > fr[i] - need])
    speed5 = lambda i: (lambda j: np.linalg.norm(G[j] - G[i]) / (fr[j] - fr[i]) * 25 if j > i else 0.0)(
        max([j for j in range(i, min(len(fr), i + 6)) if fr[j] <= fr[i] + 5]))
    strikes = []
    for i, f in enumerate(fr[:-1]):
        if not still(i) or still(i + 1):
            continue                                                  # not the last frame at rest
        off = next((fr[j] for j in range(i, len(fr)) if fr[j] <= f + need and speed5(j) >= BREAK_MS), None)
        if off is not None:
            strikes.append(int(off))

    paused = lambda s, r: paused_in(fastest, s, r)

    def placed_after(r, upto):
        """The players seemed to resume at r, but the ball came to rest within PLACED_S of it (placed for the restart)
        and was struck from there (strikes) by upto: that strike is the restart -- the players picking up while the
        ball is knocked about is not play. Unless they picked up on a placed ball struck (within RESUME_LAG, as they
        react to a restart): that one was it. -> its frame, or None."""
        if any(abs(k - r) <= RESUME_LAG for k in strikes):
            return None
        for k in strikes:
            if r < k <= upto:
                j = int(np.searchsorted(fr, k)) - 1                      # back from the strike, over the ball at rest
                while j > 0 and np.linalg.norm(G[j - 1] - G[j]) < CONTACT_M and fr[j] - fr[j - 1] <= ob.FIT_HALF:
                    j -= 1
                if fr[j] <= r + PLACED_S * 25:
                    return int(k)
        return None
    strikes = sorted(set(s for k_, s in enumerate(strikes) if k_ == 0 or s - strikes[k_ - 1] > need))

    P = np.load(ob.S + 'tactical.npz')['P'][..., :2]
    ob.STALE = ob.stale_mask(P)
    whistles = sorted(int(w) for w in stops if half_of(w))
    _stops, restart_whistles = predict.read_whistles(predict.WHISTLES_CSV.format(game=predict.GAME))
    restart_whistles = sorted(int(r) for r in restart_whistles)
    tac = np.load(ob.S + 'tactical.npz')['P']; spd = tac[..., 2]
    fastest = np.nanmean(-np.sort(-np.nan_to_num(spd, nan=-1), 1)[:, :FASTEST], 1)
    kd = DROP_S * 25
    pace = np.convolve(np.nan_to_num(fastest), np.ones(kd) / kd)[kd - 1:]      # pace[f]: their mean over f..f+kd-1
    pace_from = lambda f: pace[min(len(pace) - 1, f)]
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
            # ... and it stays out: a ball followed back onto the pitch within PLACED_S never left it (a dropping ball's
            # ground point runs in from past the line) -- followed, frame to frame no faster than a ball (VMAX_MPF): a
            # jump back in is another object, not the ball
            back = False
            for j in range(i + 1, len(fr)):
                if fr[j] > fr[i] + PLACED_S * 25 or np.linalg.norm(G[j] - G[j - 1]) > ob.VMAX_MPF * (fr[j] - fr[j - 1]):
                    break
                if past[j] < 0:
                    back = True; break
            # ... and it got there: from where it was seen before, no faster than a ball (VMAX_MPF) -- a jump out is the
            # tracker taking another ball, a spare one lying off the pitch
            gone = i >= 2 and fr[i - 1] - fr[i - 2] >= ob.CLAIM_GAP_S * 25
            jumped = i >= 2 and not gone and np.linalg.norm(G[i - 1] - G[i - 2]) > ob.VMAX_MPF * (fr[i - 1] - fr[i - 2])
            # ... and it did not fly on: a ball back in play after an out was held first (a throw-in) or placed (a goal
            # kick, a corner) -- moving at BREAK_MS or more from each sighting past the line to the next, until it is
            # seen on the pitch again, it never stopped: a ball in the air over the line (its ground point runs out
            # and back), however long unseen
            j = next((j for j in range(i + 1, len(fr)) if past[j] < 0), None)
            flew_on = (not back and j is not None and fr[j] - fr[i] <= OUT_S * 25
                       and all(BREAK_MS / 25 <= np.linalg.norm(G[k + 1] - G[k]) / (fr[k + 1] - fr[k]) <= ob.VMAX_MPF
                               for k in range(i - 1, j)))                    # (no faster than a ball: VMAX_MPF)
            if not back and not jumped and not flew_on:
                # seen out only after it vanished in play (for CLAIM_GAP_S or longer, as for ownball's keeper claims):
                # it went out unseen, after it was last seen on the pitch -- when the players eased off
                last_in = fr[:i - 1][past[:i - 1] < 0]
                x, y = G[i - 1]
                line = 'touch' if -min(y, W - y) >= -min(x, L - x) else 'goal'      # the line it is furthest past
                exits.append((int(fr[i - 1]), eased_off(fastest, int(last_in[-1]), int(fr[i - 1])) if gone and len(last_in) else int(fr[i - 1]), line))
            was_in = False
    # a ball struck from where it was placed (strikes) is a restart, not an out, even where its ground point runs on
    # over a line (a corner, a throw-in): no exit within the strike's fit (FIT_HALF) after it
    exits = [(f, d, line) for f, d, line in exits if not any(0 <= f - k <= ob.FIT_HALF for k in strikes)]
    inside = fr[past < 0]                                                     # the frames the ball is seen on the pitch
    seen_breaks = np.array(sorted(e['f'] for e in json.load(open(ob.OUT_JSON))['events']))
    last_break = lambda f: int(seen_breaks[np.searchsorted(seen_breaks, f) - 1]) if np.searchsorted(seen_breaks, f) else f
    tags = [t for t in tagged_outs() if half_of(t[0])]
    kickoffs = [a for _h, a, _b in halves] + [t for t, k, _ in tags if k == 'kick-off']
    found = setups(tac[..., :2] * [L, W], spd, [t for t, _, _ in tags] + whistles, [(a, b) for _h, a, b in halves], kickoffs)
    print(f'{len(found)} restart set-ups where nothing else says play stopped: {dict(Counter(k for _, k in found))}')
    def restart_of(kind, s, earliest, limit, rw):
        """the restart we can see: a goal kick or a corner, the placed ball struck in its goal area or at its corner arc;
        a throw-in, the held ball struck from the body of a player on a touchline (within MARGIN_M: the thrower; held
        overhead, its ground point lies metres inside the line); a stoppage whistle's free kick, on the referee's restart
        whistle (the kick after it). Else None: the players resuming ends it."""
        if kind == 'throw-in':
            on_line = lambda f, d: any(np.isfinite(P[f, k, 1]) and min(P[f, k, 1] * W, W - P[f, k, 1] * W) <= MARGIN_M
                                       for dist, k, *_ in ob.bodies(M, P, f, track[f][:2]) if dist <= d)
            for f in strikes:
                if earliest <= f <= limit:
                    near = ob.bodies(M, P, f, track[f][:2])
                    if near and near[0][0] <= ob.MAX_REACH_M:
                        y = P[f, near[0][1], 1] * W
                        if min(y, W - y) <= MARGIN_M:
                            return f
            # the ball not seen held: the first time it is seen at a touchline (within MARGIN_M of it) and on its way in
            # at BREAK_MS or more, with a player on the line within reach of it -- the thrower; before the players
            # resume (resumed: the throw is what restarts play, and they react to it)
            upto = min(limit, resumed(s, limit))
            for i in range(int(np.searchsorted(fr, earliest)), int(np.searchsorted(fr, upto, 'right'))):
                inside_by = lambda j: min(G[j, 1], W - G[j, 1])
                j = min(len(fr) - 1, i + ob.FIT_HALF)
                if (inside_by(i) <= MARGIN_M and inside_by(j) > inside_by(i) and speed5(i) >= BREAK_MS
                        and on_line(int(fr[i]), ob.MAX_REACH_M)):
                    return int(fr[i])
            return None                                                       # no whistle restarts a throw-in
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
    # a half whose tracking starts before its kick-off: dead from its first frame until the kick-off is taken
    kickoff_tag = {half_of(t)[0]: (t, tm) for t, tm in half_kickoffs() if half_of(t) and t > half_of(t)[0]}
    starts = [(a, 'half start', 'kick-off', tm) for a, (_t, tm) in kickoff_tag.items()]
    # a goal (tagged, with the kick-off after it): dead from the ball going in -- first seen past a goal line within the
    # goal's span, as for an out (by OUT_M on two tracked frames in a row), by our ball (not in flight: the ground point
    # of a cross in the air runs past the line) or Metrica's, whichever is first: inside the penalty area's width (a ball in the air into the goal has
    # its ground point well off to the side of the posts; not out by the corner flag) -- else from the players easing
    # off over the span, until that kick-off; before anything else in between (a review, the celebrations: players
    # running then is not play)
    def in_goal(frs, P, ok):
        out = (np.minimum(P[:, 0], L - P[:, 0]) < -predict.OUT_M) & (np.abs(P[:, 1] - W / 2) <= BOX_HALF_W) & ok
        return next((int(frs[i - 1]) for i in range(1, len(frs)) if out[i - 1] and out[i] and frs[i] - frs[i - 1] <= 3), None)
    mframes = np.flatnonzero(np.isfinite(mball[:, 0]))
    goal_at = {}
    for g0, g1, scorer in goal_spans():
        ko = next((t for t, k, _tm in tags if k == 'kick-off' and g0 <= t <= g1), None)
        if ko is None:
            continue
        k, mk = (fr >= g0) & (fr <= ko), mframes[(mframes >= g0) & (mframes <= ko)]
        inn = [f for f in (in_goal(fr[k], G[k], ~flying[fr[k]]), in_goal(mk, mball[mk] * [L, W], np.ones(len(mk), bool))) if f]
        at, how = (min(inn), 'ball in the goal') if inn else (eased_off(fastest, g0, ko), 'speed drop')
        goal_at[at] = (ko, how, {'H': 'A', 'A': 'H'}.get(scorer))      # the other team kicks off
    tags = [t for t in tags if not (t[1] == 'kick-off' and any(ko == t[0] for ko, *_ in goal_at.values()))]
    starts += [(at, 'goal', 'kick-off', kicks) for at, (_ko, _how, kicks) in goal_at.items()]
    # the broadcast cutting away to replays for longer than OUT_S: nobody misses that much live play -- a stoppage
    import csv
    rows = [r for r in csv.DictReader(open(__import__('paths').DATA + 'broadcast/BAR-ATH_broadcast_to_tactical.csv'))]
    replay_end, i = {}, 0
    while i < len(rows):
        if rows[i]['kind'] in ('replay', 'wipe'):
            j = i
            while j + 1 < len(rows) and rows[j + 1]['kind'] in ('replay', 'wipe'):
                j += 1
            t0 = next((int(float(rows[k]['tactical_frame'])) for k in range(i - 1, -1, -1) if rows[k]['tactical_frame']), None)
            t1 = next((int(float(rows[k]['tactical_frame'])) for k in range(j + 1, len(rows)) if rows[k]['tactical_frame']), None)
            if t0 is not None and t1 is not None and (j - i + 1) > OUT_S * 25 and half_of(t0):
                replay_end[t0] = t1
            i = j + 1
        else:
            i += 1
    events = sorted([(w, 'whistle', 'free kick', None) for w in whistles] + [(t, 'tag', k, tm) for t, k, tm in tags]
                    + [(f, 'set-up', k, None) for f, k in found] + starts)
    def stoppages_of(events):
        """every stoppage in time order from the evidence (events), as below."""
        out, prev_end = [], -1
        for anchor, cause, kind, team in events:
            not_before, first = prev_end + 1, None
            while True:
                if cause == 'whistle':
                    s, how = anchor, 'whistle'
                elif cause == 'half start':
                    s, how = anchor, 'before the kick-off'
                elif cause == 'goal':
                    s, how = anchor, goal_at[anchor][1]
                else:
                    s, how = out_moment(anchor, not_before, fastest, flag_ups, exits, inside, last_break(anchor), kind, pips)
                if first and (how == 'within' or s < not_before):
                    out.append(first); prev_end = first['restart'] - 1           # no other evidence: the first timing stands
                    break
                if s <= prev_end and cause == 'goal' and out:
                    s, how = (lambda o: (o['start'], o['timed_by']))(out.pop())   # a goal takes the open stoppage on to its kick-off
                elif s <= prev_end:
                    break                                                         # part of the stoppage already open
                a, b = half_of(s)
                ko_tag = kickoff_tag[anchor][0] if cause == 'half start' else goal_at[anchor][0] if cause == 'goal' else anchor
                # the referee restarts play on a whistle -- a kick-off on one at its tag (TAG_LEAD_S either side): the whistles
                # before it, over the celebrations or a review, restart nothing
                since = max(s, ko_tag - TAG_LEAD_S * 25) if kind == 'kick-off' else s
                rw = next((r for r in restart_whistles if r > since), None)
                cap = ko_tag + TAG_LEAD_S * 25 if kind == 'kick-off' else s + predict.WHISTLE_CAP   # a kick-off: by its tag --
                if (kind == 'kick-off' and rw is not None and rw > cap
                        and all(pace_from(f) < RUN_MS for f in range(ko_tag, rw - kd))):
                    cap = rw + 3 * 25                    # or on his whistle after it, while the players still wait (RUN_MS)
                limit = min(b, cap, next((w for w in whistles if w > since), b), rw + 25 * 3 if rw else b)
                # the restart comes after the out: after the assistant lowers the flag, and not within PLACED_S
                earliest = max(s + int(PLACED_S * 25), flag_down.get(s, s) + 1 if how == 'assistant flag' else 0)
                if kind == 'kick-off':                                            # only the kick-off restarts it: not before
                    earliest = max(earliest, ko_tag)                                # its tag (players
                    # running in the meantime are celebrating or warming up, not playing)
                # the ball out over a goal line restarts with a corner or a goal kick, whatever the tag says: the placed ball
                # struck at a corner arc or in a goal area says which
                line = next((ln for _f, d, ln in exits if d == s), None) if how == 'ball exit' else None
                seen = [(r, k) for k in ('corner', 'goal kick') if line == 'goal' and kind not in ('corner', 'goal kick')
                        for r in [restart_of(k, s, earliest, limit, None)] if r is not None]
                if seen:
                    restart, kind = min(seen)
                else:
                    restart = restart_of(kind, s, earliest, limit, rw)
                ended = 'restart'
                if restart is None:                                               # the players resume (they react just after)
                    # (a kick-off: from its tag -- before it, the players were celebrating, or warming up)
                    restart, ended = max(earliest, resumed(ko_tag if kind == 'kick-off' else s, limit) - RESUME_LAG), 'players resume'
                    placed = placed_after(restart, limit)
                    if placed:
                        restart, ended = placed, 'restart'
                entry = dict(start=int(s), end=int(restart - 1), cause=cause, waits_for='restart' if kind == 'any' else kind,
                             restart=int(restart), ended_by=ended, tag=None if cause == 'whistle' else anchor, timed_by=how,
                             tag_team=team)
                if ended == 'players resume' and cause in ('tag', 'set-up') and not paused(s, restart):
                    # the players never paused: the ball cannot be dead and at once live again -- that was not the out;
                    # the next evidence after it is
                    first = first or entry
                    not_before = s + 1
                    continue
                out.append(entry)
                prev_end = restart - 1
                break
        return out

    out = stoppages_of(events)
    # the referee restarts a dead ball: his 'restart' whistle (players a little quicker after it) in what we have as
    # live play stopped it -- a stoppage whistle, then
    live = [r for r in restart_whistles if half_of(r) and not any(st['start'] <= r <= st['restart'] + 3 * 25
                                                                   for st in out)]
    if live:
        whistles = sorted(whistles + live)
        restart_whistles = [r for r in restart_whistles if r not in live]
        out = stoppages_of(sorted(events + [(r, 'whistle', 'free kick', None) for r in live]))

    # a sure exit over a touchline that no stoppage covers is a throw-in nobody tagged -- where the players ease off
    # after it (the fastest players' pace over DROP_S, as for a speed drop, falls by RESUME_RISE, the whistle rule's
    # change of pace: the ball out, they stop; play going on, they do not): dead from it until the throw (restart_of:
    # the held ball struck from a player on the touchline), else until the players resume
    kd = DROP_S * 25
    for f, d, line in exits:
        if line != 'touch' or not half_of(d) or any(st['start'] <= d <= st['end'] for st in out):
            continue
        if not np.nanmean(fastest[max(0, d - kd):d]) - np.nanmean(fastest[d:d + kd]) >= predict.RESUME_RISE:
            continue
        a, b = half_of(d)
        limit = min(b, d + predict.WHISTLE_CAP, next((st['start'] for st in out if st['start'] > d), b) - 1)
        earliest = d + int(PLACED_S * 25)
        if earliest > limit:
            continue
        restart = restart_of('throw-in', d, earliest, limit, None); ended = 'restart'
        if restart is None:
            restart, ended = max(earliest, resumed(d, limit) - RESUME_LAG), 'players resume'
            placed = placed_after(restart, limit)
            if placed:
                restart, ended = placed, 'restart'
        out.append(dict(start=int(d), end=int(restart - 1), cause='exit', waits_for='throw-in', restart=int(restart),
                        ended_by=ended, tag=None, timed_by='ball exit', tag_team=None))
    # play stopped when the players stopped running: live play has someone running (RUN_MS), so where nobody runs for
    # longer than OUT_S (the fastest players' pace over DROP_S, as for a speed drop), the play that went before had
    # stopped -- but a player may walk for any reason, so this only times a stoppage other evidence says there was:
    # one we found ending that stretch (a tag, a whistle, a set-up) began when the players stopped; and with none, the
    # broadcast away on replays for longer than OUT_S inside it, or the referee's restart whistle at its end (a dead ball
    # restarted), say it was one. When the players stopped: the referee's pip then, else their pace dropping. Its restart:
    # on the restart whistle (the kick after it), else the placed ball struck as they set off running, else them running
    dead = np.zeros(len(pace), bool)
    for st in out:
        dead[st['start']:st['end'] + 1] = True
    for _h, a, b in halves:
        f = a
        while f <= b:
            if dead[f] or pace_from(f) >= RUN_MS:
                f += 1
                continue
            g = f
            while g <= b and not dead[g] and pace_from(g) < RUN_MS:
                g += 1
            if g - f > OUT_S * 25:
                lo = max([f - kd, a] + [st['end'] + 1 for st in out if st['end'] < f])
                s, how = eased_off(fastest, lo, f + kd), 'players stopped'
                pip = next((p for p in pips if lo <= p <= s + kd), None)        # he blows as they stop
                s, how = (pip, 'referee pip') if pip is not None else (s, how)
                ends = next((st for st in out if st['start'] == g), None)
                if ends and ends['cause'] not in ('goal', 'half start'):
                    ends.update(start=int(s), timed_by=how)
                else:
                    rw = next((r for r in restart_whistles if g - kd <= r <= g + kd), None)
                    replayed = any(f <= t0 < g for t0 in replay_end)
                    if rw is not None or replayed:
                        restart = (next((k for k in strikes if rw <= k <= rw + 3 * 25), rw) if rw is not None
                                   else max((k for k in strikes if g - kd <= k <= g + kd), default=g))
                        out.append(dict(start=int(s), end=int(restart - 1), cause='replay' if replayed else 'restart whistle',
                                        waits_for='restart', restart=int(restart), tag=None, timed_by=how, tag_team=None,
                                        ended_by='players resume' if rw is None and restart == g else 'restart'))
            f = g
    out.sort(key=lambda st: st['start'])
    # play is the players running (RUN_MS): if, before anyone runs after their 'resume', they fall below the stoppage's own
    # lull (the fastest players' pace over DROP_S, as for a speed drop), the ball was still dead -- the restart is the
    # one they then run from: a placed ball struck as they set off, else their pace rising out of that deeper lull
    for k, st in enumerate(out):
        r = st['restart']
        nxt = out[k + 1]['start'] if k + 1 < len(out) else (half_of(st['start']) or (0, len(pace)))[1]
        if st['ended_by'] != 'players resume' or r - st['start'] <= kd:
            continue                                                          # a restart seen played stands
        jog = next((g for g in range(r, nxt - kd) if pace_from(g) >= RUN_MS), None)
        if jog is None or jog == r:
            continue
        lull = min(pace_from(g) for g in range(st['start'], r - kd + 1))
        deeper = min(range(r, jog), key=pace_from)
        if pace_from(deeper) >= lull:
            continue
        restart = next((f for f in strikes if deeper < f <= jog + kd), None); ended = 'restart'
        if restart is None:
            restart, ended = max(deeper, resumed(deeper, nxt) - RESUME_LAG), 'players resume'
        st.update(restart=int(restart), end=int(restart - 1), ended_by=ended)
    # a restart must put the ball in play: play stopping again within PLACED_S means it was the same stoppage (but for a
    # goal: scored, the ball was in play); and so
    # does play stopping again before anyone ran -- the fastest players' pace (over DROP_S, as for a speed drop) never
    # up to a jog (RUN_MS): the ball knocked about while the players walk to a set piece is not play
    kd = DROP_S * 25
    ran = lambda r, s: any(np.nanmean(fastest[f:f + kd]) >= RUN_MS for f in range(r, max(r + 1, s - kd + 1)))
    joined = []
    for st in out:
        if joined and st['cause'] != 'goal' and (st['start'] - joined[-1]['end'] - 1 < PLACED_S * 25
                       or not ran(joined[-1]['end'] + 1, st['start'])):
            joined[-1] = dict(st, **{k: joined[-1][k] for k in ('start', 'cause', 'timed_by')})   # stopped as the first
        else:
            joined.append(st)
    out = joined
    # the broadcast's half-time break (alignment/align_broadcast.py: no match on screen) is dead ball: from its last
    # tactical frame before the break to the end of that half, no restart -- the next half starts at its kick-off. A
    # stoppage still open then (the whistle that ended the half) is the half-time itself
    import csv
    rows = [r for r in csv.DictReader(open(__import__('paths').DATA + 'broadcast/BAR-ATH_broadcast_to_tactical.csv'))]
    for i in range(1, len(rows)):
        if rows[i]['kind'] == 'break' and rows[i - 1]['kind'] != 'break':
            last = next((int(float(rows[j]['tactical_frame'])) for j in range(i - 1, -1, -1) if rows[j]['tactical_frame']), None)
            h = half_of(last) if last is not None else None
            if not h:
                continue
            out = [st for st in out if st['end'] < last and not st['start'] > h[1] or st['start'] > h[1]]
            out = sorted(out + [dict(start=last, end=h[1], cause='half-time', waits_for='half-time', restart=None,
                                     ended_by='half-time', tag=None, timed_by='broadcast break', restart_team=None)],
                         key=lambda st: st['start'])
    # who takes the set piece: the player within reach of the ball when it restarts
    team = ['H'] * 24 + ['A'] * 22
    keepers = ob.keeper_slots()
    sides = ob.sides_by_half(P)
    for st in out:
        r, st['restart_team'] = st['restart'], st.pop('tag_team', st.get('restart_team'))
        if st['restart_team']:
            continue                                                          # the tag names the team
        g0 = None if r is None else next((g for d in range(4) for g in (r - d, r + d) if g in track), None)
        if g0 is not None:
            # a throw-in: the thrower stands on the touchline (within MARGIN_M) -- the nearest player in reach who does,
            # else the nearest in reach
            reach = [(d, k) for d, k, *_ in ob.bodies(M, P, g0, track[g0][:2]) if d <= ob.MAX_REACH_M]
            if not reach and st['ended_by'] == 'restart':
                # a struck ball is timed by the frame it is moving fast: its taker was at it the moment before -- the
                # last player in reach in the PLACED_S before
                for g1 in range(r - 1, r - int(PLACED_S * 25) - 1, -1):
                    if g1 in track:
                        reach = [(d, k) for d, k, *_ in ob.bodies(M, P, g1, track[g1][:2]) if d <= ob.MAX_REACH_M]
                        if reach:
                            g0 = g1
                            break
            on_line = [(d, k) for d, k in reach if min(P[g0, k, 1] * W, W - P[g0, k, 1] * W) <= MARGIN_M]
            near = on_line if st['waits_for'] == 'throw-in' and on_line else reach
            if near:
                k = near[0][1]
                st['restart_team'] = ob.keeper_team(P, g0, k, sides) if k in keepers else team[k]
        if not st['restart_team'] and st['waits_for'] == 'throw-in' and r is not None:
            # the ball not seen at the throw: the thrower stands on or behind a touchline -- the player furthest out
            # over one (or nearest it) in the PLACED_S before, if he is on it (within MARGIN_M)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                y = np.nanmedian(P[max(0, r - int(PLACED_S * 25)):r, :, 1], 0) * W
            inside = np.where(np.isfinite(y), np.minimum(y, W - y), np.inf)
            k = int(np.argmin(inside))
            if inside[k] <= MARGIN_M:
                st['restart_team'] = team[k]
    # a restart we saw played is dead until it is sure whose it is: from the first frame on which only one player could
    # still get to the ball before it is received (the next break in its motion, ownball's events) -- running at
    # RUN_MS to within MAX_REACH_M of where the ball will be (the team taking it keeps it). Sure only once it is
    # received (within the break's fit, FIT_HALF, before it) is not sure while it travels: a ball played into a
    # contest, in play from the strike
    breaks = sorted(e['f'] for e in json.load(open(ob.OUT_JSON))['events'] if e['kind'] in ('touch', 'deflection'))
    for st in out:
        r = st['restart']
        if st['ended_by'] != 'restart':
            continue
        rec = next((f for f in breaks if f > r + ob.FIT_HALF), None)
        if rec is None or (nxt := next((s['start'] for s in out if s['start'] > r), None)) is not None and rec >= nxt:
            continue
        ts = np.array([t for t in range(r, rec + 1) if t in track])
        if not len(ts):
            continue
        sure = ob.only_one_reaches(P, ts, G[[idx[t] for t in ts]], r, rec - ob.FIT_HALF - 1)
        if sure:
            g, k = sure
            # it says when, not whose: a set piece is played to a teammate, so the team taking it keeps the ball (an
            # opponent who gets to it takes it with his touch); only where nobody says who took it, the receiver's.
            # Sure to reach an opponent: it was played into the other team, in play from the strike
            receiver = ob.keeper_team(P, g, k, sides) if k in keepers else team[k]
            if st['restart_team'] in (None, receiver):
                st.update(released=r, restart=g, end=g - 1, restart_team=receiver)
    json.dump(dict(stoppages=out, strikes=strikes), open(OUT_JSON, 'w'), indent=1)
    print(f'{len(out)} stoppages {dict(Counter(s["waits_for"] for s in out))}, ended by {dict(Counter(s["ended_by"] for s in out))} '
          f'| {len(strikes)} placed balls struck -> {OUT_JSON}')


if __name__ == '__main__':
    cmd_build()
