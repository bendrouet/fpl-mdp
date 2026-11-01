import numpy as np
import pandas as pd
import data, model, milp

HIT = 4
FORMS = [(1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 4, 2), (1, 4, 3, 3), (1, 4, 5, 1), (1, 5, 4, 1), (1, 5, 3, 2), (1, 5, 2, 3)]
XI_MIN = np.array([1, 3, 2, 1])
POOL = (12, 40, 45, 25)     # candidates per position handed to the milp each week


def lineup(xp, pos):
    # exact for an additive objective: best formation is the best of 8 top-k picks
    best = None
    for fm in FORMS:
        xi = []
        for q in range(4):
            xi += list(xp[pos == q].sort_values(ascending=False).index[:fm[q]])
        v = xp[xi].sum()
        if best is None or v > best[0]:
            best = (v, xi)
    xi = best[1]
    order = xp[xi].sort_values(ascending=False).index
    bench = [i for i in xp.index if i not in xi]
    gk = [i for i in bench if pos[i] == 0]
    outf = sorted([i for i in bench if pos[i] != 0], key=lambda i: -xp[i])
    return xi, order[0], order[1], gk + outf


def valid(xi, pos):
    cnt = np.bincount([pos[i] for i in xi], minlength=4)
    return cnt[0] == 1 and (cnt >= XI_MIN).all()


def score(xi, cap, vice, bench, pts, mins, pos, bb=False, tc=False):
    played = lambda i: mins.get(i, 0) > 0
    if bb:
        final = list(xi) + list(bench)
    else:
        final = list(xi)
        for b in bench:
            if not played(b):
                continue
            for o in [i for i in final if not played(i)]:
                if pos[o] != pos[b] and (pos[o] == 0 or pos[b] == 0):
                    continue
                cand = [i for i in final if i != o] + [b]
                if valid(cand, pos):
                    final = cand
                    break
    c = cap if played(cap) else (vice if played(vice) else None)
    total = sum(pts.get(i, 0) for i in final)
    if c is not None:
        total += pts.get(c, 0) * (2 if tc else 1)
    return total


def run(season, H=3, w=None, eps=milp.EPS, freeze=False, site=False, menu=False, noise=0.0, shrink=False, seed=0, pool=POOL, time_limit=20, gap=0.005, verbose=False):
    p, rows, f, *_ = model.bundle(season)
    fc = model.forecasts(season)
    pts, mins = data.wide(p, 'pts'), data.wide(p, 'minutes')
    sxp = data.wide(p, 'xP')
    K = data.RULES[season]['K']
    squad, buy, bank, ft = None, {}, 1000, 1
    rng = np.random.default_rng(seed)
    log = []
    for t in range(1, 39):
        xp, info = fc[t]
        xp0 = xp
        if site:
            xp = xp.copy()
            xp[t] = sxp[t].reindex(xp.index).fillna(0)
        if noise:      # E4: the policy acts on a corrupted forecast, rho below still uses the model's
            xp = (xp + rng.normal(0, noise, xp.shape)).clip(lower=0)
            if shrink:     # bivariate normal shrinkage, b = tau^2 / (tau^2 + sigma^2), tau from the clean spread
                m, tau2 = xp0.mean(), xp0.var()
                xp = m + tau2 / (tau2 + noise ** 2) * (xp - m)
        cols = list(xp.columns[:H])
        if squad is not None and freeze:
            ins, outs, hits, gm = [], [], 0, [np.nan] * 3
        else:
            cand = set(squad or [])
            for q in range(4):
                cand |= set(xp[info.pos == q][cols].max(axis=1).nlargest(pool[q]).index)
            cand = sorted(cand)
            X = xp.loc[cand, cols].values.T
            pr = np.tile(info.price[cand].values, (len(cols), 1))
            pos, club = info.pos[cand].values, info.team[cand].values
            ww = None if w is None else np.asarray(w)[:len(cols)]
            if squad is None:
                r = milp.plan(X, pr, pos, club, free_first=True, bank0=1000, K=K, w=ww, eps=eps, time_limit=time_limit, gap=gap)
            else:
                s0 = np.isin(cand, squad)
                b0 = np.array([buy.get(i, info.price[i]) for i in cand])
                r = milp.plan(X, pr, pos, club, squad0=s0, buy0=b0, bank0=bank, ft0=ft, K=K, w=ww, eps=eps, time_limit=time_limit, gap=gap)
            if menu and squad is not None:
                # G(j): best one week value with exactly j transfers and no hit charged, the menu of math.md sec 1
                G = [milp.plan(X[:1], pr[:1], pos, club, squad0=s0, buy0=b0, bank0=bank, ft0=ft, K=K, eps=eps, nfix=j,
                               time_limit=time_limit, gap=0)['obj'] for j in range(4)]
                gm = np.diff(G)
            else:
                gm = [np.nan] * 3
            ins = [cand[i] for i in np.where(r['tin'][0])[0]]
            outs = [cand[i] for i in np.where(r['tout'][0])[0]]
            hits = 0 if squad is None else int(r['hits'][0])
        for i in outs:
            bank += int(milp.sell_price(buy[i], info.price[i]))
            del buy[i]
        for i in ins:
            bank -= int(info.price[i])
            buy[i] = int(info.price[i])
        squad = sorted(set(squad or []) - set(outs) | set(ins))
        n = len(ins)
        ft = 1 if squad is None or t == 1 else int(np.clip(ft - n + 1, 1, K))
        xi, cap, vice, bench = lineup(xp.loc[squad, t], info.pos[squad])
        g = score(xi, cap, vice, bench, pts[t].to_dict(), mins[t].to_dict(), info.pos) - HIT * hits
        rho = xp0.loc[xi, t].sum() + xp0.loc[cap, t] - HIT * hits    # forecast of the chosen lineup, math.md sec 5
        tv = bank + sum(milp.sell_price(buy[i], info.price[i]) for i in squad)
        log.append(dict(gw=t, pts=g, rho=rho, hits=hits, n_in=n, ft=ft, bank=bank, tv=tv, g1=gm[0], g2=gm[1], g3=gm[2],
                        squad=' '.join(map(str, squad)), xi=' '.join(map(str, xi)), cap=cap))
        if verbose:
            print(t, g, n, hits, ft, bank)
    return pd.DataFrame(log)


def path_bound(season, eps=0.0, pool=(8, 30, 35, 18), time_limit=600, gap=0.002, chips=None, realized=False, extra=(), start_log=None, msg=0):
    # hindsight problem on the whole seasons forecasts (math.md sec 5), selling at current price so every real path stays feasible.
    # the pool is pruned, so this is exact only if the optimum lives inside the pool
    p, *_ = model.bundle(season)
    fc = model.forecasts(season)
    K = data.RULES[season]['K']
    pts = data.wide(p, 'pts')
    obj = lambda t: pts[t].reindex(fc[t][0].index).fillna(0) if realized else fc[t][0][t]   # realized = wait and see value
    cand = set()
    for t in range(1, 39):
        xp, info = fc[t]
        for q in range(4):
            cand |= set(obj(t)[info.pos == q].nlargest(pool[q]).index)
    cand = sorted(cand | set(extra))    # extra: players held by the policies being compared, so their paths are feasible here
    n = len(cand)
    price = data.wide(p, 'value').reindex(cand)
    X = np.array([obj(t).reindex(cand).fillna(0).values for t in range(1, 39)])
    pr = price[range(1, 39)].fillna(999).astype(int).values.T
    last = pd.concat([fc[t][1].reindex(cand) for t in range(1, 39)]).groupby(level=0).last()
    clubs = np.array([fc[t][1].team.reindex(cand).fillna(last.team).astype(int).values for t in range(1, 39)])
    start = None
    if start_log is not None:    # a sim.run log with squad/xi/cap columns, used as the warm start
        pos_of = {e: k for k, e in enumerate(cand)}
        start = {k: np.zeros((38, n), int) for k in ['squad', 'xi', 'cap', 'tin', 'tout']}
        prev = set()
        for t, row in enumerate(start_log.itertuples()):
            sq = set(map(int, row.squad.split()))
            for e in sq: start['squad'][t, pos_of[e]] = 1
            for e in map(int, row.xi.split()): start['xi'][t, pos_of[e]] = 1
            start['cap'][t, pos_of[int(row.cap)]] = 1
            for e in sq - prev: start['tin'][t, pos_of[e]] = 1
            for e in prev - sq: start['tout'][t, pos_of[e]] = 1
            prev = sq
    r = milp.plan(X, pr, last.pos.astype(int).values, clubs, free_first=True, bank0=1000,
                  K=K, eps=eps, sell_cur=True, chips=chips, start=start, time_limit=time_limit, gap=gap, msg=msg)
    return r, cand


if __name__ == '__main__':
    import time
    t0 = time.time()
    lg = run('2023-24', H=1, verbose=True)
    print(lg.pts.sum(), lg.hits.sum(), lg.n_in.sum(), round(time.time() - t0), 's')
