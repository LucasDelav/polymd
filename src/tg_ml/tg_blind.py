"""Blind Tg extraction: window-center-independent pooled asymptote-intersection
+ two self-consistency layers that flag/repair a mis-placed window WITHOUT ever
using the experimental Tg.

WHY: a single MD cooling window gives a Tg estimate that TRACKS the window center
(pull ~0.9) — the kinetically-broadened transition has no sharp local breakpoint.
Pooling several offset windows into one wide scan and intersecting the glassy and
melt asymptotes (fit far from Tg) removes that center-dependence. Two mechanisms
then handle a window placed so far off that the transition is at/outside it:

  Layer 1 (shape + FONDU recenter): a weak pooled slope contrast and/or the
    intersection jammed at a scan edge => the knee is outside the window. When the
    per-window engine emits a FONDU recommendation ("relance --tg-prior XXX") we
    re-center on that concrete target. Validated: recovers 5/6 gross mis-placements.
  confidence(): a blind self-consistency score (contrast, ⟨u²⟩-vs-U agreement, knee
    position, FONDU). It does NOT change the value — it flags which answers to
    distrust. This surfaces the residual hard case (a knee sitting shape-invisibly
    on the cold edge, e.g. oxy33) rather than silently mis-reporting it.

REJECTED (do not re-add): an invariance auto-fix that re-centers a blind scan on
its OWN estimate. Tested — it cannot reach a far-away knee, and a cold-heavy probe
biases every estimate low (corrupted good cases, MAE 43K vs 18.6K). There is no
blind auto-correction for a shape-invisible edge knee; confidence() is the answer.

All temperatures here are SIMULATION temperatures (Tg_sim); divide by the kinetic
factor CAL=1.50 for the experimental prediction. See reference-tg-prediction-method.
"""
from __future__ import annotations
import json
import re
import numpy as np

CAL = 1.50  # universal kinetic correction Tg_sim -> Tg_exp
# Prigogine-Defay decoupling correction: the Tg seen by VOLUME (ρ) and by ENTHALPY (U) differ;
# their gap (U_Tg − ρ_Tg) ∝ the V/H response decoupling ∝ fragility. Nudging the volumetric Tg
# k=0.20 toward the caloric Tg removes part of the per-molecule kinetic residual. ONE universal
# physical constant (like CAL), LOO-validated: blind median 19→15K. NOT a per-family fit.
K_DECOUPLE = 0.20


def _grab_array(txt, key):
    """Bracket-match the JSON array value for `key` in a pipeline .out dump."""
    i = txt.find(f'"{key}"')
    if i < 0:
        return None
    i = txt.find("[", i)
    depth, j = 0, i
    while j < len(txt):
        if txt[j] == "[":
            depth += 1
        elif txt[j] == "]":
            depth -= 1
            if depth == 0:
                break
        j += 1
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def parse_curves(text):
    """Extract the pooled-able observables from one pipeline run's stdout.
    Returns dict {'u2': (T[], <u2>[]), 'U': (T[], U[])} plus a 'fondu' flag and
    any engine re-center target. ⟨u²⟩ per palier = the cage plateau (last MSD lag)."""
    out = {"curves": {}, "fondu": False, "fondu_target": None}
    msd = _grab_array(text, "MSD_T")
    if msd:
        Ts, U2 = [], []
        for rec in msd:
            c = rec.get("curve")
            if c:
                Ts.append(rec["T"]); U2.append(c[-1][1])
        if len(Ts) >= 6:
            out["curves"]["u2"] = (np.array(Ts), np.array(U2))
    ut = _grab_array(text, "U_T")
    if ut and len(ut) >= 6:
        out["curves"]["U"] = (np.array([r[0] for r in ut]), np.array([r[1] for r in ut]))
    rho = _grab_array(text, "RHO_T")          # densité par palier → coude dilatométrique (VALEUR)
    if rho and len(rho) >= 6:
        out["curves"]["rho"] = (np.array([r[0] for r in rho]), np.array([r[1] for r in rho]))
    dt = _grab_array(text, "D_T")             # diffusion par palier → angle (CONFIANCE)
    if dt:
        pts = [(r[0], r[1]) for r in dt if r[1] is not None]
        if len(pts) >= 6:
            out["curves"]["D"] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    if "SOUS la fenêtre" in text or "FONDU" in text:
        out["fondu"] = True
        m = re.search(r"--tg-prior\s+(\d+)", text)
        if m:
            out["fondu_target"] = float(m.group(1))
    return out


def _branches(T, y, q=0.30, q_melt=None):
    """Average duplicate-T points, fit glass (cold q) & melt (warm q_melt) lines.
    q_melt defaults to q (symmetric). For DENSITY ρ(T), q_melt≈0.45 is better: the
    melt/rubbery branch is longer & cleaner, so giving the red line more points sharpens
    the crossing (validated: density median MAE 38.5→25.4K). Glass branch stays at q."""
    qm = q if q_melt is None else q_melt
    T = np.asarray(T, float); y = np.asarray(y, float)
    o = np.argsort(T); T, y = T[o], y[o]
    uT = np.unique(T)
    uy = np.array([y[T == t].mean() for t in uT])
    n = len(uT)
    kg = max(3, int(round(q * n)))
    km = max(3, int(round(qm * n)))
    cl = np.polyfit(uT[:kg], uy[:kg], 1)   # glassy asymptote (cold q)
    ch = np.polyfit(uT[-km:], uy[-km:], 1)  # melt asymptote (warm q_melt)
    return cl, ch, uT, uy, kg


def pooled_intersection(T, y, q=0.30, q_melt=None):
    """Window-independent Tg_sim from the glass/melt asymptote crossing.
    Returns (Tg_sim, info) or (None, info). info carries shape diagnostics.
    q_melt: fraction of warm points for the melt branch (default = q; use ≈0.45 for ρ)."""
    if len(np.unique(T)) < 8:
        return None, {"reason": "too_few_points"}
    cl, ch, uT, uy, k = _branches(T, y, q, q_melt)
    gslope, mslope = float(cl[0]), float(ch[0])
    if abs(gslope - mslope) < 1e-12:
        return None, {"reason": "parallel_branches"}
    Tx = float((ch[1] - cl[1]) / (cl[0] - ch[0]))
    span = float(uT.max() - uT.min())
    frac = (Tx - uT.min()) / span if span > 0 else 0.5
    contrast = mslope / gslope if abs(gslope) > 1e-12 else float("inf")
    # ANGLE between the two branches in NORMALIZED [0,1] coords (axis-scale-free): 0°=parallel
    # (no coude → distrust), 90°=orthogonal (sharp coude → confident). Validated as the best
    # confidence signal on the DIFFUSION observable (corr(angle,err)=−0.60). See confidence_angle().
    qm = q if q_melt is None else q_melt
    n = len(uT); kg = max(3, int(round(q * n))); km = max(3, int(round(qm * n)))
    span_y = float(uy.max() - uy.min()) or 1.0
    Tn = (uT - uT.min()) / (span if span > 0 else 1.0)
    yn = (uy - uy.min()) / span_y
    mg = float(np.polyfit(Tn[:kg], yn[:kg], 1)[0])
    mm = float(np.polyfit(Tn[-km:], yn[-km:], 1)[0])
    angle = float(abs(np.degrees(np.arctan(mm) - np.arctan(mg))))
    info = {
        "Tg_sim": Tx, "contrast": float(contrast), "frac": float(frac),
        "glass_slope": gslope, "melt_slope": mslope, "angle_deg": angle,
        "n_glass": int((uT < Tx).sum()), "n_melt": int((uT > Tx).sum()),
        "t_lo": float(uT.min()), "t_hi": float(uT.max()),
    }
    if not (uT.min() - 40 <= Tx <= uT.max() + 40):
        return None, {**info, "reason": "intersection_out_of_range"}
    return Tx, info


def blind_estimate(curves, q=0.30):
    """curves: {observable: (T[], y[])}, e.g. {'u2': (...), 'U': (...)}.
    Average the per-observable intersections. Returns (Tg_sim, info)."""
    ests, infos = {}, {}
    for name, (T, y) in curves.items():
        if name not in ("u2", "U"):          # ρ/D handled by blind_tg_recipe (own q_melt) — don't mix
            continue
        x, inf = pooled_intersection(T, y, q)
        if x is not None:
            ests[name] = x
            infos[name] = inf
    if not ests:
        return None, {"reason": "no_fit", "per_obs": infos}
    Tg = float(np.mean(list(ests.values())))
    disagree = (max(ests.values()) - min(ests.values())) if len(ests) > 1 else 0.0
    # shape flags are judged on the MOBILITY observable (u2): its glass/melt slope
    # contrast genuinely reflects whether a real knee was bracketed. U(T)'s contrast
    # is intrinsically weak (enthalpy is near-linear) and would flag everything.
    prim = "u2" if "u2" in infos else next(iter(infos))
    s = infos[prim]
    return Tg, {"Tg_sim": Tg, "ests": ests, "disagree": float(disagree),
                "contrast": s["contrast"], "frac": s["frac"],
                "n_glass": s["n_glass"], "n_melt": s["n_melt"],
                "shape_obs": prim, "per_obs": infos}


def blind_tg_recipe(curves, q_rho=0.45, q_diff=0.50):
    """★ THE VALIDATED RECIPE (2026-06): Tg VALUE from the density ρ(T) coude (q_melt q_rho),
    CONFIDENCE TIER from the diffusion D(T) branch ANGLE (q_melt q_diff). On the 20-mol
    convergence run: converged (angle≥48°) → density MAE 30/médiane 16K; non-converged → 72K.
    Falls back to ⟨u²⟩ for the value and contrast-confidence if ρ/D are absent (legacy runs).
    Returns (Tg_pred_C | None, info) with info['tier','angle_deg','value_obs','Tg_sim','reasons'].
    """
    info = {"reasons": [], "value_obs": None, "angle_deg": None}
    # ── VALUE: density coude (+ Prigogine-Defay correction), else ⟨u²⟩ ──
    val_sim = None
    if "rho" in curves:
        x, _ = pooled_intersection(*curves["rho"], q_melt=q_rho)
        if x is not None:
            val_sim, info["value_obs"] = x, "rho"
            # nudge the volumetric Tg toward the caloric (U) Tg by k (decoupling ∝ fragility)
            if "U" in curves:
                u, _ = pooled_intersection(*curves["U"])
                if u is not None:
                    val_sim = val_sim + K_DECOUPLE * (u - val_sim)
                    info["pdefay"] = round(float(u - x) / CAL, 1)   # U−ρ gap (pred K), diagnostic
                    info["reasons"].append(f"correction Prigogine-Defay U−ρ ×{K_DECOUPLE}")
    if val_sim is None and "u2" in curves:
        x, _ = pooled_intersection(*curves["u2"])
        if x is not None:
            val_sim, info["value_obs"] = x, "u2"
            info["reasons"].append("valeur: repli ⟨u²⟩ (pas de densité)")
    # ── CONFIDENCE: diffusion branch angle preferred, else contrast on ⟨u²⟩ ──
    if "D" in curves:
        _, idd = pooled_intersection(*curves["D"], q_melt=q_diff)
        info["angle_deg"] = idd.get("angle_deg")
    if info["angle_deg"] is not None:
        tier, reason = confidence_angle(info["angle_deg"])
    elif "u2" in curves:
        _, iu = pooled_intersection(*curves["u2"])
        tier, _, _ = confidence(iu)
        reason = "confiance: repli contraste (pas de diffusion)"
    else:
        tier, reason = "moyenne", "aucun signal de confiance"
    info["tier"] = tier; info["reasons"].append(reason)
    info["Tg_sim"] = round(val_sim, 1) if val_sim is not None else None
    tg = (val_sim / CAL - 273.15) if val_sim is not None else None
    info["Tg_pred_C"] = round(tg, 1) if tg is not None else None
    return tg, info


# ── Layer 1 — shape flags (cheap, single pooled bracket) ───────────────────────
def shape_flags(info, contrast_min=3.2, frac_lo=0.20, frac_hi=0.85, n_branch_min=4):
    """Return (suspect: bool, direction: 'down'|'up'|None, reasons: list)."""
    reasons, direction = [], None
    if info.get("contrast", 9.9) < contrast_min:
        reasons.append(f"contrast {info['contrast']:.1f}<{contrast_min} (no real knee bracketed)")
    frac = info.get("frac", 0.5)
    if frac < frac_lo:
        reasons.append(f"intersection jammed at COLD edge (frac {frac:.2f})"); direction = "down"
    elif frac > frac_hi:
        reasons.append(f"intersection jammed at WARM edge (frac {frac:.2f})"); direction = "up"
    if info.get("n_glass", 9) < n_branch_min:
        reasons.append(f"glass branch starved (n={info.get('n_glass')})"); direction = "down"
    elif info.get("n_melt", 9) < n_branch_min:
        reasons.append(f"melt branch starved (n={info.get('n_melt')})"); direction = "up"
    suspect = bool(reasons)
    # if contrast-only suspect (no edge signal), default to probing DOWN: the FF/VK
    # bias places windows HIGH, so the knee is most often below.
    if suspect and direction is None:
        direction = "down"
    return suspect, direction, reasons


# ── Re-center controller: follow the engine's FONDU recommendation ─────────────
# NB. A re-center is only trustworthy when it has a CONCRETE TARGET (the engine's
# FONDU "relance --tg-prior XXX", computed from the melt-slope) or a shape-edge
# direction. Re-centering a blind scan on its OWN (possibly wrong) estimate was
# TESTED and REJECTED: an estimate that is wrong because the knee is far away
# cannot reach that far knee, and a cold-heavy probe biases every estimate low
# (validated: it corrupted good cases, MAE 43K vs 18.6K). So there is NO blind
# auto-fix for a knee that sits shape-invisibly at a scan edge (e.g. oxy33);
# such cases are surfaced via confidence() instead of silently "corrected".

def recenter_target(info, fondu_target=None):
    """Return (should_recenter, center_sim, reasons). Acts ONLY on a concrete
    signal: the engine's FONDU target, or a shape edge/starvation direction."""
    suspect, direction, reasons = shape_flags(info)
    if fondu_target is not None:
        return True, float(fondu_target), ["FONDU: knee outside window → " + str(round(fondu_target))]
    if suspect and direction == "down":
        return True, info["Tg_sim"] - 90.0, reasons
    if suspect and direction == "up":
        return True, info["Tg_sim"] + 90.0, reasons
    return False, None, reasons


# ── Confidence: 3-tier blind trust label (does NOT modify the value) ───────────
# Tier strings stay {haute, moyenne, basse} for back-compat (cli.py maps them to
# ✅ confiance / 🟡 à vérifier / 🔴 rejeter). The LOGIC is the HONEST design below.
TIER_LABEL = {"haute": "confiance", "moyenne": "à vérifier", "basse": "rejeter"}

def confidence(info, fondu_flagged=False, n_seeds=1, converged=None):
    """3-tier blind trust label for a Tg estimate (no experimental Tg used).
    Returns (tier, score 0-1, reasons); tier ∈ {haute, moyenne, basse} mapped by
    TIER_LABEL to confiance / à vérifier / rejeter.

    ★ HONEST DESIGN (validated exhaustively 2026-06; see project-overflag-floor-explained).
    No blind self-consistency signal separates a GOOD estimate from a systematically
    mis-placed one — proven over 8 observables, rigorous UQ (Davies/Muggeo/Patrone),
    a group-contribution consensus, and a 45-molecule window-scan study. So we do NOT
    pretend to predict correctness:
      • REJETER (basse) ⇐ ONLY positive, physically-grounded failure evidence — the knee
        sits at an extreme window edge (extrapolation, not interpolation) or FONDU fires
        with the knee on the cold side (transition provably below the window). NOT weak
        contrast: contrast measures transition SHARPNESS, not accuracy, and condemning low
        contrast wrongly rejects broad-transition low-Tg polymers whose Tg is dead-on
        (oxy24/26/48, err≈0 — the false-positive bug this redesign fixes).
      • CONFIANCE (haute) ⇐ positive multi-signal quality: sharp contrast AND ⟨u²⟩/U(T)
        agreement AND centered knee AND no FONDU (+ multi-seed / window-scan convergence).
      • À VÉRIFIER (moyenne) ⇐ the inseparable middle, the honest DEFAULT. A weak but
        otherwise-clean estimate lands here, NOT in rejeter — so its false-positive cost ≈0.
    `converged`: optional window-scan verdict ('convergé' | 'non résolu'). An ABSTAIN
    signal — 'non résolu' caps the tier at à-vérifier (never forces rejeter by itself);
    'convergé' nudges score up. KNOWN irreducible limit: 'chemistry-deceptive' polymers
    (structure implies high Tg, real Tg low) can be wrong yet pass every signal — documented,
    not flag-able.
    """
    c = info.get("contrast", 9.9)
    d = info.get("disagree", 0.0)
    f = info.get("frac", 0.5)
    # ── REJETER — hard, positive failure evidence only ──
    # NB: FONDU is NOT a reject signal — empirically a window that saw FONDU got
    # re-centered and ended up MORE reliable (n_fondu≥2 → MAE 15K vs 30K), so FONDU
    # firing means "repaired", not "broken". Only an out-of-range knee is hard evidence.
    hard = []
    if f <= 0.12 or f >= 0.90:
        hard.append(f"coude au bord extrême (frac {f:.2f}) → extrapolation hors transition")
    if hard:
        return "basse", 0.2, hard
    # ── CONFIANCE — the only signals with VALIDATED predictive value: a sharp pooled
    # contrast (≥8, the production cut: MAE 21K vs 32K) AND ⟨u²⟩/U(T) agreement. (frac is
    # ~0.5 by construction for a pooled bracket; FONDU is ambiguous post-recenter.) ──
    # disagree≤70 (⟨u²⟩/U must genuinely agree) is the cut that keeps the lone catastrophe
    # oxy33 (disagree 82) OUT of confiance → confiance MAE 20K vs 29K rest, 0 catastrophe.
    ok_c = c >= 8.0
    ok_d = d <= 70.0
    ok_v = converged != "non résolu"
    if ok_c and ok_d and ok_v:
        score = 0.75 + 0.05 * (c >= 11.0) + 0.10 * (n_seeds >= 3) + 0.10 * (converged == "convergé")
        return "haute", round(min(1.0, score), 2), ["contraste net + ⟨u²⟩/U cohérents"]
    # ── À VÉRIFIER — the inseparable middle (honest default; FPs land here, cost ≈0) ──
    reasons = []
    if not ok_c: reasons.append(f"contraste modéré {c:.1f} (transition douce ≠ erreur)")
    if not ok_d: reasons.append(f"⟨u²⟩ vs U écart {d:.0f}K")
    if not ok_v: reasons.append("scan fenêtre: point fixe non résolu (abstention)")
    return "moyenne", 0.5, reasons or ["signaux mitigés"]


# ── Angle-based confidence + convergence (the user's criterion, calibrated on diffusion) ──
# The angle between the glass & melt asymptotes (normalized coords, 0-90°) measures how SHARP
# the coude is. Bigger angle → more orthogonal → real, well-defined transition → more reliable.
# Calibrated on D(T): corr(angle, |error|)=−0.60; angle≥48° → MAE 33K vs <48° → 68K.
ANGLE_HAUTE, ANGLE_MOY = 48.0, 35.0

def confidence_angle(angle_deg):
    """3-tier trust from the branch angle alone. Returns (tier, reason)."""
    if angle_deg >= ANGLE_HAUTE:
        return "haute", f"angle {angle_deg:.0f}°≥{ANGLE_HAUTE:.0f}° — coude net/orthogonal"
    if angle_deg >= ANGLE_MOY:
        return "moyenne", f"angle {angle_deg:.0f}° — coude modéré"
    return "basse", f"angle {angle_deg:.0f}°<{ANGLE_MOY:.0f}° — coude mou (≈parallèle)"

def converged(angle_deg, n_calc, angle_thresh=ANGLE_HAUTE, max_calc=5):
    """Convergence rule for an iterative (add-a-seed) loop:
      • DONE+confident  when angle ≥ angle_thresh (sharp coude → trust the value), OR
      • DONE (give up)  when n_calc ≥ max_calc (won't iterate forever; flag low-confidence).
    Returns (done: bool, confident: bool, reason: str)."""
    if angle_deg >= angle_thresh:
        return True, True, f"convergé : angle {angle_deg:.0f}°≥{angle_thresh:.0f}° après {n_calc} calcul(s)"
    if n_calc >= max_calc:
        return True, False, f"arrêt à {n_calc} calculs (non convergé, angle {angle_deg:.0f}°<{angle_thresh:.0f}°)"
    return False, False, f"angle {angle_deg:.0f}°<{angle_thresh:.0f}° — continuer ({n_calc}/{max_calc})"
