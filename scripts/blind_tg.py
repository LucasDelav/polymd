#!/usr/bin/env python3
"""Blind Tg pipeline driver — end-to-end, no experimental Tg used.

  VK group-contribution seed  →  3-window pooled wide scan (ρ, D, ⟨u²⟩, U per palier)
  →  ⟨u²⟩ shape drives FONDU re-centering (window placement)
  →  ★ THE RECIPE (blind_tg_recipe): Tg VALUE from the density ρ(T) coude (q_melt 0.45);
     CONFIDENCE tier from the diffusion D(T) branch ANGLE (≥48°=haute). Validated 45 mols:
     haute MAE 28/méd 20K; basse catches the window-misplacement catastrophes. Replaces the
     old contrast-based confidence() flag (kept only as legacy fallback when ρ/D are absent).

Usage:
  blind_tg.py "*OCCCOC(*)=O"                 # one PSMILES
  blind_tg.py --batch polymers.json          # [{name, polymer_smiles}, ...]

This is the reference orchestration of src/tg_ml/tg_blind.py over the cluster; it
reuses tg_ml.cli.submit / wait_for_jobs. See reference-tg-prediction-method memory.
"""
import argparse, json, sys, time, glob
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tg_ml.tg_blind import (parse_curves, blind_estimate, recenter_target,
                            blind_tg_recipe, confidence, CAL)
from tg_ml.cli import submit, _ssh, REMOTE_ROOT
from vk_centerer import features as vk_features

VK = json.load(open("vk_centerer_model.json"))
_MU, _SD, _WZ = np.array(VK["mu"]), np.array(VK["sd"]), np.array(VK["wz"])


def vk_seed_K(psmiles):
    x, _ = vk_features(psmiles)
    if x is None:
        return 373.0
    z = np.append((x - _MU) / _SD, 1.0)
    return float(z @ _WZ)


def _wait(jobids):
    ids = list(jobids)
    while ids:
        time.sleep(30)
        q = _ssh("squeue -h -o %i -j " + ",".join(ids), check=False).split()
        ids = [j for j in ids if j in q]


def _bracket(psmiles, center_sim, tag):
    """Submit a 3-window bracket (±50K, WIN140) with MSD; return pooled curves."""
    jobs = []
    for sub, c in zip("dum", [center_sim - 50, center_sim, center_sim + 50]):
        env = {"SMILES": psmiles, "TG_EXP": round(center_sim / 1.5), "TG_SIM_PRIOR": round(c),
               "WIN_HI": 140, "WIN_LO": 140, "MSD_TG": 1, "MECH": 0, "CP_DOS": 0,
               "TG_DESC": f"BLIND {tag} center={round(c)}K"}
        jid, out = submit(env, f"bt_{tag}_{sub}", "01:45:00", "gpu_all")
        jobs.append((jid, out))
    _wait([j for j, _ in jobs])
    merged, fondu, target = {}, False, None      # pool ALL observables (u2, U, rho, D)
    for jid, out in jobs:
        txt = _ssh(f"cat {REMOTE_ROOT}/{out}", check=False)
        pc = parse_curves(txt)
        for k, (T, Y) in pc["curves"].items():
            merged.setdefault(k, ([], [])); merged[k][0].extend(T); merged[k][1].extend(Y)
        fondu = fondu or pc["fondu"]
        if pc["fondu_target"]:
            target = pc["fondu_target"]
    curves = {k: (np.array(v[0]), np.array(v[1])) for k, v in merged.items() if v[0]}
    return curves, fondu, target


def run_one(name, psmiles, max_recenter=2):
    seed = vk_seed_K(psmiles)
    center = 1.5 * seed
    fondu_flagged = False
    curves, info, it, Tg_sim = {}, {}, 0, None
    for it in range(max_recenter + 1):
        # recenter is driven by the ⟨u²⟩ SHAPE (blind_estimate) — best for window placement
        curves, fondu, target = _bracket(psmiles, center, f"{name}_{it}")
        fondu_flagged = fondu_flagged or fondu
        Tg_sim, info = blind_estimate(curves)
        if Tg_sim is None:
            return {"name": name, "tg_pred_C": None, "confidence": "échec", "reasons": ["no fit"]}
        do_rc, tgt, _ = recenter_target(info, fondu_target=target)
        if not do_rc or it == max_recenter:
            break
        center = tgt
    # ★ FINAL value + confidence via THE RECIPE: density coude value + diffusion-angle tier
    tg, rinfo = blind_tg_recipe(curves)
    if tg is None:                                   # density failed → fall back to ⟨u²⟩ estimate
        tg = Tg_sim / CAL - 273.15
    return {"name": name, "vk_seed_C": round(seed - 273.15, 1),
            "tg_pred_C": round(tg, 1),
            "confidence": rinfo["tier"], "conf_reasons": rinfo["reasons"],
            "value_obs": rinfo["value_obs"],
            "angle_deg": round(rinfo["angle_deg"], 1) if rinfo["angle_deg"] is not None else None,
            "contrast": round(info["contrast"], 1), "iters": it + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="?")
    ap.add_argument("--batch")
    a = ap.parse_args()
    items = (json.load(open(a.batch)) if a.batch
             else [{"name": "poly", "polymer_smiles": a.smiles}])
    results = []
    for e in items:
        r = run_one(e.get("name", "poly"), e["polymer_smiles"])
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
    json.dump(results, open("blind_tg_results.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
