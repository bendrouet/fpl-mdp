# Math notes

Short guide to what each file computes.

## 1. Banking game (`bank.py`)

State is $f \in \{1,\dots,5\}$, the number of banked free transfers. Each week there's a random menu: the best $u$ transfers gain $G(u)$ and the $j$-th adds $g(j) = G(j) - G(j-1)$. Each transfer past $f$ costs 4. Backward induction:

$$V_t(f) = \mathbb{E}\Big[\max_u \big\{ G(u) - 4(u-f)^+ + V_{t+1}(\min(\max(f-u+1, 1), 5)) \big\}\Big], \qquad V_{39} = 0.$$

`bank.week` does the max for a batch of sampled menus, `bank.solve` loops $t = 38, \dots, 1$.

For decreasing menus ($g(1) \ge g(2) \ge \dots$) the numbers show each extra banked transfer is worth less than the last. This isn't proved here. The two-week example with $K = 3$ and $g = (1, 6)$ (sell two to fund one) is an increasing menu, and there the pattern doesn't hold: $V_1 = (7, 10, 14)$. About half the real menus aren't decreasing (`sim.run(menu=True)` reads them off).

## 2. Points model (`model.py`)

Team goals are Poisson with $\log \lambda = \mu + \text{home} + \text{attack} - \text{defence}_{\text{opp}}$, fit each week on past matches only by weighted max likelihood with a ridge penalty (0.3, otherwise adding a constant to every attack and every defence leaves every $\lambda$ unchanged). Weights decay by 0.03 per gameweek, and last season's matches are weighted 0.5 on top. Each goal goes to player $i$ with probability $p_i$ (goals per 90, or expected goals per 90 from 2022-23, times expected minutes, normalized), so by thinning player $i$ scores $\text{Poisson}(\lambda p_i)$. Assists are the same with 0.8 per goal.

Clean sheet is $e^{-\lambda'}$ and the conceded penalty is $\mathbb{E}\lfloor N'/2 \rfloor$ for $N' \sim \text{Poisson}(\lambda')$. Expected points are the scoring weights times these expectations, plus saves, bonus, yellow cards and defensive contributions (`fixture_points`). Rates use pseudo-counts, $(w \cdot \text{prior} + \text{count}) / (w + \text{exposure})$, so new players start at their position's average.

`simulate` draws team goals first and then the scorers, so teammates' points are correlated.

## 3. Forecast quality (`calibration.ipynb`)

Slope of realized points on the forecast made $k$ weeks out, $\beta_k$. Relative to $k = 1$ it goes 1.00 down to 0.86 over 9 weeks, much flatter than the usual $0.85^{k-1}$. The `b` strategies weight week $k$ by $\beta_k$.

## 4. Squad picker (`milp.py`)

Integer program for squad, lineup, captain and transfers over the next $H$ weeks, maximizing forecast points minus hits. Solved with PuLP/CBC. Selling prices follow the real rule, and bench players get a small weight `EPS`.

## 5. Backtest and hindsight run (`sim.py`)

Each week: forecast from past data, solve, apply week 1 of the plan, score with real points (auto-subs and captain included).

Let $\rho_t$ be the forecast for the lineup actually picked (column `rho`). Let $U$ be the best forecast total over a pruned candidate pool, with every week's forecasts known in advance (`path_bound`).

Real managers: the game's average score per gameweek (`data.avg_score`) summed over the season, roughly 1900 to 2030.

## Names

- `xp`: forecast points
- `rho`: forecast of the chosen lineup
- `ft`: banked free transfers
- `g1, g2, g3`: menu
- `hK`: plan $K$ weeks, equal weights; `hKd`: $0.85^{k-1}$; `hKb`: $\beta_k$
