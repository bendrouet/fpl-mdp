import numpy as np

K = 5      # free transfer cap
U = 4      # most transfers considered in a week
T = 38
HIT = 4


def week(G, Vn, K=K):
    # one week of the banking game for a batch of menus G (n, U+1), G[:,0]=0
    # returns H (n, K) and smallest maximizer u* (n, K), f indexed 1..K
    n, U1 = G.shape
    f = np.arange(1, K + 1)[:, None]
    u = np.arange(U1)[None, :]
    nxt = np.clip(f - u + 1, 1, K) - 1
    phi = G[:, None, :] - HIT * np.maximum(u - f, 0)[None] + Vn[nxt][None]
    return phi.max(2), phi.argmax(2)


def solve(sample, K=K, T=T, n=2000, seed=0):
    # sample(rng, n) -> (n, U) sorted nonincreasing gains and returns V (T+1, K) and menus used
    rng = np.random.default_rng(seed)
    V = np.zeros((T + 2, K))
    for t in range(T, 0, -1):
        g = sample(rng, n)
        G = np.concatenate([np.zeros((n, 1)), np.cumsum(g, 1)], 1)
        H, _ = week(G, V[t + 1], K)
        V[t] = H.mean(0)
    return V


def kappa(Vn, K=K):
    # price of the j-th transfer at stock f, for decreasing menus
    k = np.zeros((K + 1, U + 1))
    for f in range(1, K + 1):
        for j in range(1, U + 1):
            d = f - j
            if d <= -1:
                k[f, j] = HIT
            elif d <= K - 2:
                k[f, j] = Vn[d + 1] - Vn[d]
            else:
                k[f, j] = 0
    return k[1:, 1:]


def expo_menu(mean=2.0, U=U):
    def sample(rng, n):
        return -np.sort(-rng.exponential(mean, (n, U)), 1)
    return sample


def counterexample():
    # T=2, K=3, deterministic non-concave menu g=(1,6) vs concave g=(6,1)
    for g in [(1, 6), (6, 1)]:
        G = np.array([[0, g[0], g[0] + g[1]]], float)
        V2, _ = week(G, np.zeros(3), 3)
        V1, u1 = week(G, V2[0], 3)
        print(g, V2[0], V1[0], u1[0])


if __name__ == "__main__":
    counterexample()
    V = solve(expo_menu())
    print(np.round(V[1], 2), np.round(np.diff(V[1]), 3))
    print(np.round(kappa(V[2]), 3))
