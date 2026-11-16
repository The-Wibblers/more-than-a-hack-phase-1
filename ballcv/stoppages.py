"""Stoppages of the whole match: when play stopped, which restart it then waits for, and when that restart happened.
possession/predict.py reads them for its dead-ball judgement.

That play stopped, and how it restarts: the match's set-piece tags (data/tactical/..._smart_tagging.xml: throw-ins,
corners, goal kicks, free kicks, kick-offs after a goal, with the team taking it where named); a stoppage whistle in
the audio (a free kick); players setting up a
restart where nothing else says play stopped (one standing at a corner: a corner; off the pitch over a touchline: a
throw-in, a substitution, an injury seen to -- players leave the pitch only while the ball is dead).

When it stopped: a half whose tracking starts before its kick-off, from its first frame; a goal (tagged), from the
ball seen in the goal, else the players easing off over the goal's span, until the kick-off; the broadcast's half-time
break, to the end of the half; the whistle; the flag; else, for a tagged or set-up restart, the first out before it (from OUT_S
before to TAG_LEAD_S after): the assistant's flag, or our ball leaving the pitch (and staying off it) -- the ball when
it crossed, the flag only follows it; not where the ball is seen flying with no arc found for it (ball3d.py: the
ground point of a ball in the air can lie past the far touchline or a goal line over the pitch); a ball first seen out
after it vanished went out unseen, when the players eased off -- else when the fastest players eased off.

When it restarted: where the ball is reliable -- a goal kick or a corner, the placed ball struck in its goal area or at
its corner arc; a whistle's free kick, on the referee's restart whistle (the kick after it). Everywhere else the players
say it: their pace rises out of its lull just after the restart (RESUME_LAG; the whistle rule's resume).
A restart puts the ball in play: if play stops again within PLACED_S of it, it is the same stoppage. The team named on
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
GOAL_W = 7.32              # between the goalposts (m), the Laws of the Game
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
    """[(start, end)] of the tagged goals ('GOALS': from the move's start to just after the kick-off that follows)."""
    import xml.etree.ElementTree as ET
    return sorted((round(float(i.findtext('start')) * 25), round(float(i.findtext('end')) * 25))
                  for i in ET.parse(TAGS).getroot().iter('instance') if i.findtext('code') == 'GOALS')


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


def out_moment(tag, after, fastest, flag_ups, exits, inside, last_break):
    """(frame, how) play stopped for a tagged (or set-up) restart, not before frame after: of the assistant's flags and
    the ball's (sure) exits seen from OUT_S before the tag to TAG_LEAD_S after it (a tag can come just before the out),
    the first -- after it the ball is dead, and moved about off the pitch; an exit is when the ball crossed, a flag only
    follows it. exits: [(frame seen, frame it went out)]. An exit before that is the out too when the ball is not seen
    back on the pitch (inside: the frames it is) between it and then: it never came back. With neither, the ball was
    still in play at the last break in its motion we saw before the tag (last_break: a touch, bounce or deflection) --
    it went out after that; the players easing off (the largest drop in the fastest players' speed) only says when,
    between then and the tag."""
    lo, hi = max(after, tag - OUT_S * 25), tag + TAG_LEAD_S * 25
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
    return eased_off(fastest, max(lo, min(last_break, tag)), tag), 'speed drop'


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
            # ... and it stays out: a ball followed back onto the pitch within PLACED_S never left it (a dropping ball's
            # ground point runs in from past the line) -- followed, frame to frame no faster than a ball (VMAX_MPF): a
            # jump back in is another object, not the ball
            back = False
            for j in range(i + 1, len(fr)):
                if fr[j] > fr[i] + PLACED_S * 25 or np.linalg.norm(G[j] - G[j - 1]) > ob.VMAX_MPF * (fr[j] - fr[j - 1]):
                    break
                if past[j] < 0:
                    back = True; break
            if not back:
                # seen out only after it vanished in play (for CLAIM_GAP_S or longer, as for ownball's keeper claims):
                # it went out unseen, after it was last seen on the pitch -- when the players eased off
                gone = i >= 2 and fr[i - 1] - fr[i - 2] >= ob.CLAIM_GAP_S * 25
                last_in = fr[:i - 1][past[:i - 1] < 0]
                exits.append((int(fr[i - 1]), eased_off(fastest, int(last_in[-1]), int(fr[i - 1])) if gone and len(last_in) else int(fr[i - 1])))
            was_in = False
    # a ball struck from where it was placed (strikes) is a restart, not an out, even where its ground point runs on
    # over a line (a corner, a throw-in): no exit within the strike's fit (FIT_HALF) after it
    exits = [(f, d) for f, d in exits if not any(0 <= f - k <= ob.FIT_HALF for k in strikes)]
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
            for f in strikes:
                if earliest <= f <= limit:
                    near = ob.bodies(M, P, f, track[f][:2])
                    if near and near[0][0] <= ob.MAX_REACH_M:
                        y = P[f, near[0][1], 1] * W
                        if min(y, W - y) <= MARGIN_M:
                            return f
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
    # a goal (tagged, with the kick-off after it): dead from the ball going in -- seen past the goal line between the
    # posts within the goal's span -- else from the players easing off over the span, until that kick-off; before
    # anything else in between (a review, the celebrations)
    goal_at = {}
    for g0, g1 in goal_spans():
        ko = next((t for t, k, _tm in tags if k == 'kick-off' and g0 <= t <= g1), None)
        if ko is None:
            continue
        inn = [int(f) for f in fr[(fr >= g0) & (fr <= ko)] if past[idx[int(f)]] > predict.OUT_M
               and abs(G[idx[int(f)], 1] - W / 2) <= GOAL_W / 2 and min(G[idx[int(f)], 0], L - G[idx[int(f)], 0]) < 0]
        at, how = (inn[0], 'ball in the goal') if inn else (eased_off(fastest, g0, ko), 'speed drop')
        goal_at[at] = (ko, how)
    tags = [t for t in tags if not (t[1] == 'kick-off' and any(ko == t[0] for ko, _h in goal_at.values()))]
    starts += [(at, 'goal', 'kick-off', None) for at in goal_at]
    events = sorted([(w, 'whistle', 'free kick', None) for w in whistles] + [(t, 'tag', k, tm) for t, k, tm in tags]
                    + [(f, 'set-up', k, None) for f, k in found] + starts)
    out, prev_end = [], -1
    for anchor, cause, kind, team in events:
        while True:
            if cause == 'whistle':
                s, how = anchor, 'whistle'
            elif cause == 'half start':
                s, how = anchor, 'before the kick-off'
            elif cause == 'goal':
                s, how = anchor, goal_at[anchor][1]
            else:
                s, how = out_moment(anchor, prev_end + 1, fastest, flag_ups, exits, inside, last_break(anchor))
            if s <= prev_end:
                break                                                         # part of the stoppage already open
            a, b = half_of(s)
            rw = next((r for r in restart_whistles if r > s), None)           # the referee restarts play on it
            ko_tag = kickoff_tag[anchor][0] if cause == 'half start' else goal_at[anchor][0] if cause == 'goal' else anchor
            cap = ko_tag + TAG_LEAD_S * 25 if kind == 'kick-off' else s + predict.WHISTLE_CAP   # a kick-off: by its tag
            limit = min(b, cap, next((w for w in whistles if w > s), b), rw + 25 * 3 if rw else b)
            # the restart comes after the out: after the assistant lowers the flag, and not within PLACED_S
            earliest = max(s + int(PLACED_S * 25), flag_down.get(s, s) + 1 if how == 'assistant flag' else 0)
            if kind == 'kick-off':                                            # only the kick-off restarts it: not before
                earliest = max(earliest, ko_tag)                                # its tag (players
                # running in the meantime are celebrating or warming up, not playing)
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
            near = ob.bodies(M, P, g0, track[g0][:2])
            if near and near[0][0] <= ob.MAX_REACH_M:
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
    # RUN_MS to within MAX_REACH_M of where the ball will be -- and that player's team has it. Sure only once it is
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
            st.update(released=r, restart=g, end=g - 1, restart_team=ob.keeper_team(P, g, k, sides) if k in keepers else team[k])
    json.dump(dict(stoppages=out, strikes=strikes), open(OUT_JSON, 'w'), indent=1)
    print(f'{len(out)} stoppages {dict(Counter(s["waits_for"] for s in out))}, ended by {dict(Counter(s["ended_by"] for s in out))} '
          f'| {len(strikes)} placed balls struck -> {OUT_JSON}')


if __name__ == '__main__':
    cmd_build()
