import pickle, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import data

H = 8
XI_T, PREV_T, RIDGE = 0.03, 0.5, 0.3          # team model: per gw decay, last season weight, ridge
DEC_MIN, PREV_MIN, A_MIN = 0.9, 0.25, 2.0       # minutes model
DEC_RATE, PREV_RATE, A_RATE = 0.97, 0.5, 3.0    # per 90 rates, prior weight in 90s
ASSIST = 0.8                                    # assists per goal
GOAL = np.array([6, 6, 5, 4])
CS = np.array([4, 4, 1, 0])
P60 = np.array([0.35, 0.35, 0.35, 0.30])
P1 = np.array([0.02, 0.10, 0.15, 0.20])
G90 = np.array([0.0, 0.05, 0.13, 0.32])
A90 = np.array([0.0, 0.06, 0.13, 0.13])
SV90, B90, YC = 2.8, 0.3, 0.12
DCR = np.array([0.0, 0.35, 0.12, 0.05])
DC_THR = np.array([99, 10, 12, 12])
XG_FROM = '2022-23'
DC_FROM = '2025-26'
CACHE = 'data/cache'

_b = {}


def bundle(season):
    if season in _b:
        return _b[season]
    p, rows, f = data.load(season)
    ps = data.PREV.get(season)
    if ps:
        _, prows, pf = data.load(ps)
        prows = prows.assign(prev=True, has_xg=ps >= XG_FROM)
        pf = pf.assign(gw=pf.GW - 38, prev=True)
    else:
        prows = rows.iloc[:0].assign(prev=True, has_xg=False)
        pf = f.iloc[:0].assign(gw=f.GW, prev=True)
    rows = rows.assign(prev=False, has_xg=season >= XG_FROM)
    fx = pd.concat([f.assign(gw=f.GW, prev=False), pf])
    names = sorted(set(fx.hn) | set(fx.an))
    _b[season] = (p, pd.concat([rows, prows]), f, fx, names)
    return _b[season]


def fit_teams(fx, t, names):
    d = fx[fx.gw < t]
    w = np.exp(-XI_T * (t - d.gw.values)) * np.where(d.prev.values, PREV_T, 1.0)
    idx = {nm: k for k, nm in enumerate(names)}
    hi, ai = d.hn.map(idx).values, d.an.map(idx).values
    hg, ag = d.hg.values.astype(float), d.ag.values.astype(float)
    n = len(names)

    def nll(th):
        att, dfn, mu, hm = th[:n], th[n:2 * n], th[2 * n], th[2 * n + 1]
        lh = np.exp(mu + hm + att[hi] - dfn[ai])
        la = np.exp(mu + att[ai] - dfn[hi])
        val = np.sum(w * (lh - hg * np.log(lh) + la - ag * np.log(la))) + RIDGE * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - hg), w * (la - ag)
        g = np.zeros_like(th)
        g[:n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + 2 * RIDGE * att
        g[n:2 * n] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + 2 * RIDGE * dfn
        g[2 * n] = rh.sum() + ra.sum()
        g[2 * n + 1] = rh.sum()
        return val, g

    th = minimize(nll, np.zeros(2 * n + 2), jac=True, method='L-BFGS-B').x
    return dict(zip(names, th[:n])), dict(zip(names, th[n:2 * n])), th[2 * n], th[2 * n + 1]


def fixture_rates(f, att, dfn, mu, hm):
    fr = f.copy()
    fr['lh'] = np.exp(mu + hm + f.hn.map(att) - f.an.map(dfn))
    fr['la'] = np.exp(mu + f.an.map(att) - f.hn.map(dfn))
    return fr


def efloor2(lam):
    k = np.arange(12)
    return (np.floor(k / 2)[None] * poisson.pmf(k[None], np.atleast_1d(lam)[:, None])).sum(1)


def player_rates(rows, t, pos):
    # weighted sums of past stats per player code, current season decayed, last season flat
    d = rows[(rows.GW < t) | rows.prev].copy()
    age = np.where(d.prev, np.inf, t - d.GW)
    d['wm'] = np.where(d.prev, PREV_MIN, DEC_MIN ** age)
    d['wr'] = np.where(d.prev, PREV_RATE, DEC_RATE ** age)
    mn = d.minutes.values
    d['s60'] = d.wm * (mn >= 60)
    d['s1'] = d.wm * ((mn > 0) & (mn < 60))
    d['m90'] = d.wr * mn / 90
    d['g'] = d.wr * np.where(d.has_xg, d.xg, d.goals)
    d['a'] = d.wr * np.where(d.has_xg, d.xa, d.assists)
    d['sv'] = d.wr * d.saves
    d['bn'] = d.wr * d.bonus
    d['y'] = d.wr * d.yc
    d['ap'] = d.wr * (mn > 0)
    d['ap60'] = d.wr * (mn >= 60)
    pcode = pos.reindex(d.code).fillna(0).astype(int).values   # players gone from the pool dont matter here
    d['dci'] = d.wr * (d.dc.values >= DC_THR[pcode]) * (~d.prev)
    S = d.groupby('code')[['wm', 'wr', 's60', 's1', 'm90', 'g', 'a', 'sv', 'bn', 'y', 'ap', 'ap60', 'dci']].sum()
    S = S.reindex(pos.index).fillna(0)
    q = pos.values
    r = pd.DataFrame(index=pos.index)
    r['p60'] = (A_MIN * P60[q] + S.s60) / (A_MIN + S.wm)
    r['p1'] = (A_MIN * P1[q] + S.s1) / (A_MIN + S.wm)
    r['g90'] = (A_RATE * G90[q] + S.g) / (A_RATE + S.m90)
    r['a90'] = (A_RATE * A90[q] + S.a) / (A_RATE + S.m90)
    r['sv90'] = (A_RATE * SV90 + S.sv) / (A_RATE + S.m90) * (q == 0)
    r['b90'] = (A_RATE * B90 + S.bn) / (A_RATE + S.m90)
    r['yc'] = (A_RATE * YC + S.y) / (A_RATE + S.ap)
    r['dc'] = (A_RATE * DCR[q] + S.dci) / (A_RATE + S.ap60)
    return r


def state(season, t):
    p, rows, f, fx, names = bundle(season)
    att, dfn, mu, hm = fit_teams(fx, t, names)
    fr = fixture_rates(f, att, dfn, mu, hm)
    pool = p[p.GW == t].set_index('element')
    pos = pd.Series(pool.pos.values, index=pool.code.values)
    pos = pos[~pos.index.duplicated()]
    r = player_rates(rows, t, pos).reindex(pool.code.values)
    r.index = pool.index
    if season < DC_FROM:
        r['dc'] = 0.0
    r['emin'] = 90 * r.p60 + 30 * r.p1
    return pool, r, fr


def sides(fr, gw):
    # (team, own rate, opponent rate) for every team fixture in the gameweek
    d = fr[fr.GW == gw]
    return [(int(x.h), x.lh, x.la) for x in d.itertuples()] + [(int(x.a), x.la, x.lh) for x in d.itertuples()]


def fixture_points(rr, q, lf, la):
    # expected points for one fixture for the players of one team; the teams poisson goals are
    # split among players by thinning with shares (rate x expected minutes), see docs/math.md sec 2
    em = rr.emin / 90
    wg, wa = rr.g90 * em, rr.a90 * em
    xg = lf * wg / wg.sum()
    xa = ASSIST * lf * wa / wa.sum()
    pts = 2 * rr.p60 + rr.p1 + GOAL[q] * xg + 3 * xa
    pts = pts + CS[q] * rr.p60 * np.exp(-la) - (q <= 1) * rr.p60 * efloor2(la)[0]
    pts = pts + rr.sv90 * em / 3 + rr.b90 * em - rr.yc * (rr.p60 + rr.p1) + 2 * rr.dc * rr.p60
    return pts.values


def forecast(season, t, H=H):
    pool, r, fr = state(season, t)
    q, team = pool.pos.values, pool.team_id.values
    xp = pd.DataFrame(0.0, index=pool.index, columns=range(t, min(t + H, 38) + 1))
    for gw in xp.columns:
        for tid, lf, la in sides(fr, gw):
            idx = np.where(team == tid)[0]
            if len(idx):
                xp.iloc[idx, xp.columns.get_loc(gw)] += fixture_points(r.iloc[idx], q[idx], lf, la)
    info = pd.DataFrame(dict(price=pool.value, pos=pool.pos, team=pool.team_id, pplay=r.p60 + r.p1,
                             p60=r.p60, name=pool.name), index=pool.index)
    return xp, info


def simulate(season, t, n=2000, seed=0):
    # joint draw of gameweek t points for the whole pool: team goals first, then player events,
    # so teammates and opponents are correlated, which the captaincy race needs
    pool, r, fr = state(season, t)
    rng = np.random.default_rng(seed)
    q, team = pool.pos.values, pool.team_id.values
    Y = np.zeros((n, len(pool)))
    for x in fr[fr.GW == t].itertuples():
        gh, ga = rng.poisson(x.lh, n), rng.poisson(x.la, n)
        for tid, G, Go in [(int(x.h), gh, ga), (int(x.a), ga, gh)]:
            idx = np.where(team == tid)[0]
            if len(idx) == 0:
                continue
            rr, qq = r.iloc[idx], q[idx]
            u = rng.random((n, len(idx)))
            m60 = u < rr.p60.values
            m1 = (u >= rr.p60.values) & (u < (rr.p60 + rr.p1).values)
            mins = 90 * m60 + 30 * m1
            wg = rr.g90.values * mins + 1e-9
            wa = rr.a90.values * mins + 1e-9
            goals = np.array([rng.multinomial(G[i], wg[i] / wg[i].sum()) for i in range(n)])
            na = rng.binomial(G, ASSIST)
            assists = np.array([rng.multinomial(na[i], wa[i] / wa[i].sum()) for i in range(n)])
            cs = (Go == 0)[:, None] & m60
            gc = np.floor(Go / 2)[:, None] * m60 * (qq <= 1)
            saves = rng.poisson(rr.sv90.values * mins / 90)
            bonus = rng.poisson(rr.b90.values * mins / 90)
            yc = rng.random((n, len(idx))) < rr.yc.values * (mins > 0)
            dc = rng.random((n, len(idx))) < rr.dc.values * m60
            Y[:, idx] += 2 * m60 + m1 + GOAL[qq] * goals + 3 * assists + CS[qq] * cs - gc + saves // 3 + bonus - yc + 2 * dc
    return pd.DataFrame(Y, columns=pool.index)


def forecasts(season, H=H):
    os.makedirs(CACHE, exist_ok=True)
    fn = f'{CACHE}/xp_{season}_{H}.pkl'
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    out = {t: forecast(season, t, H) for t in range(1, 39)}
    pickle.dump(out, open(fn, 'wb'))
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    xp, info = forecast('2023-24', 10)
    print(round(time.time() - t0, 2), 's')
    Y = simulate('2023-24', 10, n=3000)
    top = xp[10].nlargest(6).index
    print(info.name[top].str[:12].tolist())
    print(np.round(xp.loc[top, 10].values, 2), np.round(Y[top].mean().values, 2), np.round(Y[top].std().values, 2))
    print('mc vs closed form: mean abs diff', round((xp[10] - Y.mean()).abs().mean(), 3))
    for s in data.SEASONS:
        t0 = time.time(); forecasts(s); print(s, round(time.time() - t0, 1))
