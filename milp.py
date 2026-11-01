import numpy as np
import pulp

HIT = 4
QUOTA = np.array([2, 5, 5, 3])
XI_MIN = np.array([1, 3, 2, 1])
XI_MAX = np.array([1, 5, 5, 3])
CLUB_MAX = 3
EPS = 0.05     # bench weight in the objective, rough stand in for autosub probability
BIGM = 15


def sell_price(buy, cur):
    # official rule: keep half the profit, rounded down to 0.1m
    return np.where(cur > buy, buy + (cur - buy) // 2, cur)


def V(v):
    return 0 if v is None else pulp.value(v)


def plan(xp, price, pos, club, squad0=None, buy0=None, bank0=1000, ft0=1, K=5, w=None, eps=EPS,
         chips=None, sell_cur=False, free_first=False, nfix=None, start=None, time_limit=60, gap=0.0, msg=0):
    # xp, price: (H, n); club (n,) or (H, n) when players move mid season.  squad0 None means build from scratch.
    # chips: dict wc/bb/tc -> list of week indices where that chip may be played
    H, n = xp.shape
    w = np.ones(H) if w is None else np.asarray(w, float)
    chips = chips or {}
    R, N = range(H), range(n)
    bvar = lambda nm: pulp.LpVariable(nm, cat='Binary')

    m = pulp.LpProblem('fpl', pulp.LpMaximize)
    s = [[bvar(f's_{t}_{i}') for i in N] for t in R]
    x = [[bvar(f'x_{t}_{i}') for i in N] for t in R]
    c = [[bvar(f'c_{t}_{i}') for i in N] for t in R]
    tin = [[bvar(f'in_{t}_{i}') for i in N] for t in R]
    tout = [[bvar(f'out_{t}_{i}') for i in N] for t in R]
    f = [pulp.LpVariable(f'f_{t}', 1, K, cat='Integer') for t in range(H + 1)]
    u = [pulp.LpVariable(f'u_{t}', 0, K, cat='Integer') for t in R]
    h = [pulp.LpVariable(f'h_{t}', 0, cat='Integer') for t in R]
    hp = [pulp.LpVariable(f'hp_{t}', 0, cat='Integer') for t in R]
    bank = [pulp.LpVariable(f'bank_{t}', 0) for t in range(H + 1)]
    wc = [bvar(f'wc_{t}') if t in chips.get('wc', []) else None for t in R]
    bb = [bvar(f'bb_{t}') if t in chips.get('bb', []) else None for t in R]
    tc = [bvar(f'tc_{t}') if t in chips.get('tc', []) else None for t in R]
    z = [[bvar(f'z_{t}_{i}') if tc[t] is not None else None for i in N] for t in R]
    y = [[bvar(f'y_{t}_{i}') if bb[t] is not None else None for i in N] for t in R]
    o = lambda v: 0 if v is None else v

    if squad0 is None:
        s_prev = np.zeros(n, int)
        sell0 = price[0]
    else:
        s_prev = np.asarray(squad0, int)
        sell0 = sell_price(np.asarray(buy0), price[0])

    m += pulp.lpSum(w[t] * xp[t, i] * (x[t][i] + c[t][i] + o(z[t][i]) + eps * (s[t][i] - x[t][i]) + (1 - eps) * o(y[t][i]))
                    for t in R for i in N) - HIT * pulp.lpSum(hp)

    m += f[0] == ft0
    m += bank[0] == bank0
    for t in R:
        m += pulp.lpSum(s[t]) == 15
        m += pulp.lpSum(x[t]) == 11
        m += pulp.lpSum(c[t]) == 1
        for p in range(4):
            idx = [i for i in N if pos[i] == p]
            m += pulp.lpSum(s[t][i] for i in idx) == QUOTA[p]
            m += pulp.lpSum(x[t][i] for i in idx) >= XI_MIN[p]
            m += pulp.lpSum(x[t][i] for i in idx) <= XI_MAX[p]
        ct = club[t] if np.ndim(club) == 2 else club
        for cl in np.unique(ct):
            m += pulp.lpSum(s[t][i] for i in N if ct[i] == cl) <= CLUB_MAX
        for i in N:
            m += x[t][i] <= s[t][i]
            m += c[t][i] <= x[t][i]
            prev = int(s_prev[i]) if t == 0 else s[t - 1][i]
            m += s[t][i] == prev + tin[t][i] - tout[t][i]
            m += tin[t][i] + tout[t][i] <= 1
            if z[t][i] is not None:
                m += z[t][i] <= c[t][i]
                m += z[t][i] <= tc[t]
            if y[t][i] is not None:
                m += y[t][i] <= s[t][i] - x[t][i]
                m += y[t][i] <= bb[t]
        # initial squad sells at the rule price, in-window buys at their (frozen) buy price;
        # sell_cur sells everything at the current price, only used for the hindsight bound
        sell = price[t] if sell_cur else np.where(s_prev == 1, sell0, price[t])
        m += bank[t + 1] == bank[t] + pulp.lpSum(sell[i] * tout[t][i] for i in N) - pulp.lpSum(price[t][i] * tin[t][i] for i in N)
        # transfer bank rule as inequalities
        nt = pulp.lpSum(tin[t])
        m += u[t] <= f[t]
        m += u[t] <= nt
        m += h[t] == nt - u[t]
        m += f[t + 1] <= f[t] - u[t] + 1
        free = 1 if (t == 0 and (free_first or nfix is not None)) else o(wc[t])   # nfix reads the menu, no hit charge
        m += hp[t] >= h[t] - BIGM * free
        if free_first and t == 0:
            m += u[t] == 0
            m += f[1] <= 1
        m += o(wc[t]) + o(bb[t]) + o(tc[t]) <= 1
    if nfix is not None:     # exactly nfix transfers in week 0, used to read off the menu G(j) for T1
        m += pulp.lpSum(tin[0]) == nfix
    for k, v in [('wc', wc), ('bb', bb), ('tc', tc)]:
        if chips.get(k):
            m += pulp.lpSum(v[t] for t in chips[k]) <= 1

    if start is not None:
        # warm start from a known feasible path (squad, xi, cap, tin, tout arrays); the rest is implied
        ft_, bk = ft0, bank0
        for t in R:
            for i in N:
                for var, arr in [(s, 'squad'), (x, 'xi'), (c, 'cap'), (tin, 'tin'), (tout, 'tout')]:
                    var[t][i].setInitialValue(int(start[arr][t, i]))
            nt = int(start['tin'][t].sum())
            ut = 0 if (free_first and t == 0) else min(nt, ft_)
            u[t].setInitialValue(ut); h[t].setInitialValue(nt - ut); hp[t].setInitialValue(0 if (free_first and t == 0) else nt - ut)
            f[t].setInitialValue(ft_)
            ft_ = 1 if (free_first and t == 0) else int(np.clip(ft_ - nt + 1, 1, K))
            sell = price[t] if sell_cur else np.where(s_prev == 1, sell0, price[t])
            bank[t].setInitialValue(bk)
            bk = bk + int((sell * start['tout'][t]).sum()) - int((price[t] * start['tin'][t]).sum())
        f[H].setInitialValue(ft_); bank[H].setInitialValue(bk)
    m.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap, warmStart=start is not None, timeMode='cpu'))
    arr = lambda a: np.array([[V(v) for v in row] for row in a]).round().astype(int)
    vec = lambda a: np.array([V(v) for v in a]).round().astype(int)
    return dict(obj=pulp.value(m.objective), status=pulp.LpStatus[m.status],
                squad=arr(s), xi=arr(x), cap=arr(c), tin=arr(tin), tout=arr(tout),
                ft=vec(f), hits=vec(hp), bank=np.array([V(v) for v in bank]),
                wc=vec(wc), bb=vec(bb), tc=vec(tc))


if __name__ == '__main__':
    rng = np.random.default_rng(1)
    n, H = 120, 3
    pos = np.repeat([0, 1, 2, 3], [15, 40, 45, 20])
    club = rng.integers(0, 20, n)
    price = np.tile(rng.integers(40, 130, n), (H, 1))
    xp = np.clip(rng.normal(3, 1.5, (H, n)) * price[0] / 70, 0, None)
    r = plan(xp, price, pos, club, free_first=True, K=5)
    print(r['status'], round(r['obj'], 2), r['ft'], r['hits'], r['bank'].round(1), r['tin'].sum(1))
    sq = r['squad'][-1].astype(bool)
    r2 = plan(xp, price, pos, club, squad0=sq, buy0=price[0], bank0=r['bank'][-1], ft0=2, K=5,
              chips=dict(wc=[1], tc=[0, 1, 2]))
    print(r2['status'], round(r2['obj'], 2), r2['ft'], r2['hits'], r2['tin'].sum(1), r2['wc'], r2['tc'])
