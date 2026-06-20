#!/usr/bin/env python3
"""van-Krevelen-style additive group-contribution Tg estimator.

Tg = (sum_i  count_i * w_i) / M_repeat     (van Krevelen: Tg = Yg/M)

Features are physically-additive structural groups counted on the repeat unit
(the two '*' are stripped and replaced by H to make a neutral fragment). The
weights w_i are fit by ridge regression on the BASICS set only; the OXYGENATED
set is held out so its use as a window-centerer is genuinely out-of-sample.

Goal: not a precise predictor, just a coarse centerer good to ~+/-50 K.
"""
import json, sys, math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---- group SMARTS (van-Krevelen-style structural increments) ----
SMARTS = {
    "ch2":        "[CH2;!$(C=O)]",                 # methylene
    "ch":         "[CX4H1]",                        # methine (branch point)
    "cq":         "[CX4H0]",                        # quaternary C
    "ch3":        "[CH3]",                          # methyl
    "ester":      "[OX2][CX3]=[OX1]",               # -O-C(=O)-
    "carbonate":  "[OX2][CX3](=[OX1])[OX2]",        # -O-C(=O)-O-
    "ether":      "[OX2;!$([OX2][CX3]=O)]",         # ether O (not ester/carbonate)
    "amide":      "[NX3][CX3]=[OX1]",               # -N-C(=O)-
    "hydroxyl":   "[OX2H]",                         # -OH
    "arom_ring":  "c1ccccc1",                       # benzene ring
    "ali_ring_at":"[R;!a;#6]",                      # aliphatic ring carbon (rigidity)
    "halogen":    "[F,Cl,Br,I]",
    "nitrile":    "[CX2]#[NX1]",
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
KEYS = list(SMARTS.keys())


def repeat_mol(psmiles):
    """Strip the two '*' -> cap with H, return neutral repeat-unit Mol."""
    s = psmiles.replace("[*]", "*")
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    # replace dummy atoms (*) with hydrogens
    rw = Chem.RWMol(m)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(dummies, reverse=True):
        rw.GetAtomWithIdx(idx).SetAtomicNum(1)
    try:
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
        return m2
    except Exception:
        return None


def features(psmiles):
    m = repeat_mol(psmiles)
    if m is None:
        return None, None
    M = Descriptors.MolWt(m)
    if M < 1:
        return None, None
    counts = []
    for k in KEYS:
        p = PATT[k]
        n = len(m.GetSubstructMatches(p, uniquify=True)) if p is not None else 0
        counts.append(n)
    # carbonate also matches ester+ether; subtract overlap so groups are disjoint
    c = dict(zip(KEYS, counts))
    c["ester"] = max(0, c["ester"] - c["carbonate"])
    c["ether"] = max(0, c["ether"] - 2 * c["carbonate"] - c["ester"])
    x = np.array([c[k] for k in KEYS], dtype=float) / M  # van Krevelen: /M
    return x, M


def load_dataset():
    """Return list of (name, psmiles, tg_K, domain)."""
    rows = []
    # basics: PSMILES in 'smiles', tg_exp in Kelvin
    for f in ["../tg_ml_academic_archive/outputs/tgcli/_p30_results.json",
              "../tg_ml_academic_archive/outputs/tgcli/_night16_results.json",
              "../tg_ml_academic_archive/outputs/tgcli/_batch5_results.json"]:
        try:
            d = json.load(open(f))
        except FileNotFoundError:
            continue
        for e in (d if isinstance(d, list) else d.get("results", [])):
            ps, tg = e.get("smiles"), e.get("tg_exp")
            if ps and tg and "*" in ps:
                rows.append((e.get("name"), ps, float(tg), "basic"))
    # oxygenated: polymer_smiles, Tg_C
    d = json.load(open("oxygenated_polymers.json"))["polymers"]
    for e in d:
        ps, tgc = e.get("polymer_smiles"), e.get("Tg_C")
        if ps and tgc is not None and e.get("polymer_smiles_verified"):
            rows.append(("oxy%02d" % e["entry"], ps, float(tgc) + 273.15, "oxy"))
    # dedup by name keeping first
    seen, out = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0]); out.append(r)
    return out


def main():
    rows = load_dataset()
    X, Y, names, dom = [], [], [], []
    for name, ps, tg, d in rows:
        x, M = features(ps)
        if x is None:
            print("  skip (parse):", name, ps, file=sys.stderr); continue
        X.append(x); Y.append(tg); names.append(name); dom.append(d)
    X = np.array(X); Y = np.array(Y); dom = np.array(dom)
    bi = dom == "basic"; oi = dom == "oxy"
    print(f"basics n={bi.sum()}  oxygenated n={oi.sum()}  features={len(KEYS)}")

    lam = 0.3  # ridge in standardized space
    def fit(Xtr, Ytr):
        # standardize columns (van Krevelen /M features have tiny scale)
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-12] = 1.0
        Z = (Xtr - mu) / sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])  # intercept
        pen = np.full(Z.shape[1], lam); pen[-1] = 0.0  # don't penalize intercept
        A = Z.T @ Z + np.diag(pen)
        wz = np.linalg.solve(A, Z.T @ Ytr)
        return mu, sd, wz
    def pred(mdl, Xq):
        mu, sd, wz = mdl
        Z = (Xq - mu) / sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return Z @ wz

    # --- LOO within basics ---
    Xb, Yb = X[bi], Y[bi]
    loo = []
    for i in range(len(Xb)):
        mask = np.ones(len(Xb), bool); mask[i] = False
        mdl = fit(Xb[mask], Yb[mask])
        loo.append(pred(mdl, Xb[i:i+1])[0] - Yb[i])
    loo = np.array(loo)
    print(f"\n[basics LOO]  MAE={np.abs(loo).mean():5.1f} K  med={np.median(np.abs(loo)):5.1f}  "
          f"within50={np.mean(np.abs(loo)<50)*100:.0f}%  within25={np.mean(np.abs(loo)<25)*100:.0f}%")

    # --- fit on ALL basics, test out-of-sample on oxygenated ---
    mdl = fit(Xb, Yb)
    Xo, Yo = X[oi], Y[oi]
    err = pred(mdl, Xo) - Yo
    print(f"[oxy  OOS ]  MAE={np.abs(err).mean():5.1f} K  med={np.median(np.abs(err)):5.1f}  "
          f"within50={np.mean(np.abs(err)<50)*100:.0f}%  within25={np.mean(np.abs(err)<25)*100:.0f}%")

    mu, sd, wz = mdl
    print("\n  standardized weights (importance):")
    for k, wi in sorted(zip(KEYS, wz[:-1]), key=lambda t: -abs(t[1])):
        print(f"    {k:12} {wi:8.1f}")
    print(f"    {'intercept':12} {wz[-1]:8.1f}")

    print("\n  oxygenated out-of-sample predictions:")
    onames = np.array(names)[oi]
    predo = pred(mdl, Xo)
    order = np.argsort(Yo)
    for j in order:
        pr = predo[j]
        flag = "" if abs(pr-Yo[j]) < 50 else "  <-- >50K"
        print(f"    {onames[j]:8} exp={Yo[j]-273.15:6.1f}C  vk_pred={pr-273.15:6.1f}C  err={pr-Yo[j]:+6.1f}K{flag}")

    # save model
    model = {"keys": KEYS, "smarts": SMARTS, "mu": mu.tolist(), "sd": sd.tolist(),
             "wz": wz.tolist(), "lambda": lam,
             "fit_on": "basics", "loo_basics_mae": float(np.abs(loo).mean()),
             "oos_oxy_mae": float(np.abs(err).mean())}
    json.dump(model, open("vk_centerer_model.json", "w"), indent=2)
    print("\n  saved -> vk_centerer_model.json")


if __name__ == "__main__":
    main()
