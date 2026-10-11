"""Statistics shared by the kill tests and the mission analysis: paired bootstrap, Holm, AUROC."""
import numpy as np

N_BOOT = 10_000


def boot_idx(n, n_boot=N_BOOT, seed=0):
    """[n_boot, n] resample indices over paired units (seeds). Same seed -> same resamples."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def paired_bootstrap(stat, *arrays, n_boot=N_BOOT, seed=0, alpha=0.05):
    """stat(*resampled_arrays) -> scalar. Arrays share axis 0 (paired units, e.g. test seeds).
    Returns point estimate, percentile CI and two-sided bootstrap p (2 * min tail mass at 0)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays)
    idx = boot_idx(n, n_boot, seed)
    est = float(stat(*arrays))
    bs = np.array([stat(*(a[i] for a in arrays)) for i in idx])
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return {"est": est, "lo": float(lo), "hi": float(hi), "p": float(p), "n": n, "n_boot": n_boot}


def paired_diff(a, b, **kw):
    """Mean of a - b with a paired-bootstrap CI."""
    return paired_bootstrap(lambda x, y: np.mean(x - y), a, b, **kw)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney), exact via ranks."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 1, 300)
    r = paired_diff(a, a - 0.5, n_boot=2000)
    assert abs(r["est"] - 0.5) < 1e-12 and r["lo"] == r["hi"] == 0.5          # constant shift -> degenerate CI
    b = a + rng.normal(0, 1, 300)
    r = paired_diff(a, b, n_boot=2000)
    assert r["lo"] < 0 < r["hi"] and r["p"] > 0.05                             # null difference
    r = paired_diff(a + 0.5, b, n_boot=2000)
    assert r["lo"] > 0 and r["p"] < 0.01
    assert paired_diff(a, b, n_boot=500, seed=3) == paired_diff(a, b, n_boot=500, seed=3)  # seeded
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    assert auroc([1, 2, 3], [0, 0.5]) == 1.0 and auroc([0], [1]) == 0.0 and auroc([1, 0], [1, 0]) == 0.5
    print("stats self-check OK")
