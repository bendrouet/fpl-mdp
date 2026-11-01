import gzip, json, urllib.request
import numpy as np
import pandas as pd

RAW = 'data/raw'
SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
PREV = dict(zip(SEASONS, ['2020-21'] + SEASONS[:-1]))
POS = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
QUOTA = np.array([2, 5, 5, 3])
BUDGET = 1000

# per season rules; verify these against the archived rules pages
RULES = {
    '2021-22': dict(K=2, wc=2, fh=1, bb=1, tc=1),
    '2022-23': dict(K=2, wc=2, fh=1, bb=1, tc=1),
    '2023-24': dict(K=2, wc=2, fh=1, bb=1, tc=1),
    '2024-25': dict(K=5, wc=2, fh=1, bb=1, tc=1),
    '2025-26': dict(K=5, wc=2, fh=2, bb=2, tc=2),
}

SUM = ['total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       'saves', 'bonus', 'yellow_cards', 'red_cards', 'own_goals', 'penalties_missed',
       'penalties_saved', 'xP', 'expected_goals', 'expected_assists', 'defensive_contribution']

# end of season copies of the fpl api on the wayback machine, only used for the average manager score
SNAP = {'2021-22': '20220530102612', '2022-23': '20230602070006', '2023-24': '20240602000008',
        '2024-25': '20250602000010', '2025-26': '20260602000036'}


def get_snap(season):
    url = f'https://web.archive.org/web/{SNAP[season]}id_/https://fantasy.premierleague.com/api/bootstrap-static/'
    raw = urllib.request.urlopen(url).read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    open(f'{RAW}/{season}/bootstrap.json', 'wb').write(raw)


def avg_score(season):
    ev = json.load(open(f'{RAW}/{season}/bootstrap.json'))['events']
    return pd.Series({e['id']: e['average_entry_score'] for e in ev})


def load(season):
    g = pd.read_csv(f'{RAW}/{season}/merged_gw.csv')
    tm = pd.read_csv(f'{RAW}/{season}/teams.csv')
    g['position'] = g.position.replace({'GKP': 'GK'})
    g = g[g.position.isin(POS)].copy()
    g['team_id'] = g.team.map(dict(zip(tm.name, tm.id)))
    pr = pd.read_csv(f'{RAW}/{season}/players_raw.csv')
    g['code'] = g.element.map(dict(zip(pr.id, pr.code)))
    for c in SUM:
        if c not in g:
            g[c] = 0.0
    g['n_fix'] = 1
    g['home'] = g.was_home.astype(int)

    fix = g.groupby(['element', 'GW']).agg(
        {**{c: 'sum' for c in SUM}, 'n_fix': 'sum', 'value': 'first', 'selected': 'first',
         'team_id': 'first', 'position': 'first', 'name': 'first', 'code': 'first'}).reset_index()

    els = fix.element.unique()
    full = pd.MultiIndex.from_product([els, range(1, 39)], names=['element', 'GW']).to_frame(index=False)
    p = full.merge(fix, how='left', on=['element', 'GW']).sort_values(['element', 'GW'])
    for c in SUM + ['n_fix']:
        p[c] = p[c].fillna(0)
    # price and club carry over blank weeks; before a players first row he isnt in the pool
    for c in ['value', 'team_id', 'position', 'name', 'selected', 'code']:
        p[c] = p.groupby('element')[c].ffill()
    p = p.dropna(subset=['value']).copy()
    p['pos'] = p.position.map(POS).astype(int)
    p['team_id'] = p.team_id.astype(int)
    p['value'] = p.value.astype(int)
    p = p.rename(columns={'total_points': 'pts', 'goals_scored': 'goals', 'clean_sheets': 'cs',
                          'goals_conceded': 'gc', 'yellow_cards': 'yc', 'red_cards': 'rc',
                          'expected_goals': 'xg', 'expected_assists': 'xa',
                          'defensive_contribution': 'dc'})

    # one row per (player, fixture) for the minutes model and per-fixture stats
    rows = g[['element', 'code', 'GW', 'team_id', 'opponent_team', 'home', 'minutes', 'goals_scored',
              'assists', 'saves', 'bonus', 'yellow_cards', 'clean_sheets', 'expected_goals',
              'expected_assists', 'defensive_contribution', 'kickoff_time']].copy()
    rows = rows.rename(columns={'goals_scored': 'goals', 'yellow_cards': 'yc', 'clean_sheets': 'cs',
                                'expected_goals': 'xg', 'expected_assists': 'xa',
                                'defensive_contribution': 'dc', 'opponent_team': 'opp'})

    f = pd.read_csv(f'{RAW}/{season}/fixtures.csv')
    names = dict(zip(tm.id, tm.name))
    f = f[['event', 'team_h', 'team_a', 'team_h_score', 'team_a_score', 'kickoff_time']].copy()
    f['hn'] = f.team_h.map(names)
    f['an'] = f.team_a.map(names)
    f = f.rename(columns={'event': 'GW', 'team_h': 'h', 'team_a': 'a', 'team_h_score': 'hg',
                          'team_a_score': 'ag'})
    f['GW'] = f.GW.astype(int)
    return p.reset_index(drop=True), rows, f


def wide(p, col):
    return p.pivot(index='element', columns='GW', values=col)


if __name__ == '__main__':
    for s in SEASONS:
        p, rows, f = load(s)
        print(s, p.element.nunique(), len(p), p.pts.sum(), f.shape, p.groupby('GW').element.nunique().min())
