"""Analyse cinétique de la Tg : extraction par fit HYPERBOLIQUE (Patrone) de ρ(T) à
chaque vitesse de refroidissement, puis EXTRAPOLATION vers la vitesse expérimentale.

Ancré sur :
  - Afzal et al., ACS Appl. Polym. Mater. 2021, 3, 620 (fit hyperbolique, densité/CTE).
  - Soldera & Metatla, Phys. Rev. E 74, 061803 (2006) (extrapolation WLF, Éq. 2).
  - Buchholz, Paul, Binder, J. Chem. Phys. 117, 7364 (2002) (dépendance log(vitesse)).

La Tg est cinétique : Tg(q) ↑ avec la vitesse de refroidissement q. La MD (~10¹¹-10¹³ K/s)
surestime de ~80-120 K vs l'expérience (~0.17 K/s). On corrige via WLF.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit

# Constantes WLF "universelles" (Ferry ; Soldera retrouve 16.7 ± 0.9 / 48 ± 8 par fit).
C1_WLF = 17.44
C2_WLF = 51.6  # K
Q_EXP_DEFAULT = 10.0 / 60.0  # 10 K/min en K/s (vitesse DSC standard)


def hyperbola(T, rho_g, slope, dslope, Tg, width):
    """ρ(T) en deux asymptotes linéaires raccordées par une hyperbole (Patrone et al.).
    Pentes asymptotiques = slope ∓ dslope ; intersection (le coude) = Tg. `width` = douceur
    de la transition. ρ_g = densité au coude. Robuste et automatisable (pas de régions à choisir).
    """
    x = T - Tg
    return rho_g - slope * x - dslope * np.sqrt(x * x + width * width)


def fit_tg_hyperbola(temps, rho):
    """Ajuste l'hyperbole sur ρ(T) ; renvoie (Tg, dict de paramètres + CTE vitreux/caoutchouc).
    CTE volumétrique α = -(1/ρ)(dρ/dT) ; vitreux = côté basse-T (T<Tg), caoutchouc = haute-T.
    Le dict porte aussi la COVARIANCE du fit (pcov), l'incertitude 1σ sur Tg (tg_std), le RMSE
    des résidus et les bornes de Tg — de quoi calculer des intervalles de confiance + diagnostiquer.
    """
    T = np.asarray(temps, float); R = np.asarray(rho, float)
    order = np.argsort(T); T, R = T[order], R[order]
    # Estimations initiales : pentes des deux moitiés, coude au milieu.
    n = len(T); mid = n // 2
    s_lo = np.polyfit(T[:mid], R[:mid], 1)[0]
    s_hi = np.polyfit(T[mid:], R[mid:], 1)[0]
    slope0 = -(s_lo + s_hi) / 2; dslope0 = (s_lo - s_hi) / 2
    rng = T[-1] - T[0]
    # Tg borné à l'INTÉRIEUR (15-85% de la fenêtre) : évite que le fit s'emballe vers les
    # extrêmes où une seule asymptote domine (sinon Tg aberrant sur données bruitées).
    tg_lo, tg_hi = T[0] + 0.15 * rng, T[-1] - 0.15 * rng
    p0 = [R[mid], slope0, abs(dslope0) + 1e-4, np.clip(T[mid], tg_lo, tg_hi), rng / 10]
    bounds = ([R.min() - 0.2, -1, 0, tg_lo, 1e-3],
              [R.max() + 0.2, 1, 1, tg_hi, rng])
    popt, pcov = curve_fit(hyperbola, T, R, p0=p0, bounds=bounds, maxfev=20000)
    rho_g, slope, dslope, Tg, width = popt
    rho_at_Tg = hyperbola(Tg, *popt)
    # Asymptotes : pente basse-T (verre) dρ/dT = dslope−slope ; haute-T (caoutchouc) = −(slope+dslope).
    # CTE = −(1/ρ)(dρ/dT).
    cte_glass = (slope - dslope) / rho_at_Tg * 1e6
    cte_rubber = (slope + dslope) / rho_at_Tg * 1e6
    pcov_ok = np.all(np.isfinite(pcov))
    perr = np.sqrt(np.diag(pcov)) if pcov_ok else np.full(5, np.nan)
    rmse = float(np.sqrt(np.mean((R - hyperbola(T, *popt)) ** 2)))
    cte_glass_std = propagate_std(
        lambda p: (p[1] - p[2]) / hyperbola(p[3], *p) * 1e6, popt, pcov if pcov_ok else None)
    return float(Tg), {"rho_Tg": float(rho_at_Tg), "cte_glass": float(cte_glass),
                       "cte_rubber": float(cte_rubber), "width": float(width),
                       "params": popt.tolist(), "pcov": pcov.tolist() if pcov_ok else None,
                       "tg_std": float(perr[3]), "cte_glass_std": cte_glass_std,
                       "rmse": rmse, "tg_bounds": [float(tg_lo), float(tg_hi)]}


def fit_tg_breakpoint(temps, rho):
    """Estimateur ROBUSTE de Tg : rupture de pente à 2 segments (split à résidu minimal).
    Ne latche JAMAIS sur une borne (contrairement à l'hyperbole) → bien meilleur quand le coude est
    faible/en bordure. Quantifié au pas de température (résolution ~step/2). Renvoie (Tg, dict)."""
    T = np.asarray(temps, float); R = np.asarray(rho, float)
    idx = np.argsort(T); T, R = T[idx], R[idx]
    n = len(T)
    best_t, best_res = float(np.median(T)), np.inf
    for i in range(2, n - 2):                       # split entre i et i+1 ; chaque segment ≥3 pts
        res = 0.0
        for a, b in [(0, i + 1), (i, n)]:
            c = np.polyfit(T[a:b], R[a:b], 1)
            res += float(np.sum((R[a:b] - np.polyval(c, T[a:b])) ** 2))
        if res < best_res:
            best_res, best_t = res, float(T[i])
    step = float(np.median(np.abs(np.diff(np.unique(T))))) if n > 2 else 20.0
    return best_t, {"rmse": float((best_res / n) ** 0.5), "step": step}


def propagate_std(func, popt, pcov):
    """Incertitude 1σ de func(params) par propagation de la covariance (jacobien numérique).
    func prend le vecteur de paramètres et renvoie un scalaire. Renvoie None si pcov indéfinie."""
    popt = np.asarray(popt, float)
    if pcov is None:
        return None
    pcov = np.asarray(pcov, float)
    if not np.all(np.isfinite(pcov)):
        return None
    J = np.zeros(len(popt))
    f0 = float(func(popt))
    for i in range(len(popt)):
        h = 1e-6 * (abs(popt[i]) + 1e-6)
        dp = popt.copy(); dp[i] += h
        J[i] = (float(func(dp)) - f0) / h
    var = float(J @ pcov @ J)
    return float(np.sqrt(var)) if (np.isfinite(var) and var > 0) else None


def rho_at_T(T, popt, pcov=None):
    """Densité prédite par le fit à la température T (K, frame réel/simulé), + incertitude 1σ.
    Évalue l'hyperbole : pour T ≪ Tg, suit l'asymptote VITREUSE (extrapolation propre sous la fenêtre)."""
    val = float(hyperbola(T, *popt))
    return val, propagate_std(lambda p: hyperbola(T, *p), popt, pcov)


def assess_fit(temps, Tg, det, t_step):
    """Diagnostique la fiabilité du fit hyperbolique. Renvoie {reliable: bool, reasons: [str]}.
    Détecte les modes d'échec connus : coude collé à une borne (latch), incertitude Tg élevée,
    covariance indéfinie, coude peu marqué (transition lavée), résidu fort."""
    T = np.asarray(temps, float)
    t_lo, t_hi = float(T.min()), float(T.max())
    reasons = []
    edge = max(2 * t_step, 0.10 * (t_hi - t_lo))
    if (Tg - t_lo) < edge or (t_hi - Tg) < edge:
        reasons.append(f"coude (Tg_sim={Tg:.0f} K) trop près d'une borne [{t_lo:.0f},{t_hi:.0f}] K "
                       f"→ élargis la fenêtre (plage ou marge)")
    tg_std = det.get("tg_std", float("nan"))
    if not np.isfinite(tg_std):
        reasons.append("covariance du fit non définie (fit instable)")
    elif tg_std > 25:
        reasons.append(f"incertitude sur Tg élevée (±{tg_std:.0f} K)")
    params = det.get("params")
    if params:
        _, _, dslope, _, width = params
        if abs(dslope) < 1e-4 or width > 0.5 * (t_hi - t_lo):
            reasons.append("coude peu marqué (faible contraste de pente) → signal Tg noyé dans le bruit")
    if det.get("rmse", 0.0) > 0.01:
        reasons.append(f"résidu du fit élevé (RMSE={det['rmse']:.4f} g/cm³)")
    return {"reliable": len(reasons) == 0, "reasons": reasons}


def recommend_window(temps, rho, Tg, t_step, contrast_min=1.3, cte_melt=4.0e-4):
    """Si le fit est douteux, recommande quoi faire — à partir de la STRUCTURE DE PENTE de ρ(T),
    PAS de la position où le Tg ajusté a latché (cette position est bidon quand le fit est instable :
    un coude bas peut faire latcher le fit en haut → une reco basée dessus part dans le mauvais sens).

    Logique :
      • Contraste de pente fort (|pente_fondu|/|pente_verre| ≥ contrast_min) → un VRAI coude est
        PRÉSENT dans la fenêtre. Ce n'est pas un problème de placement mais de stabilité du fit →
        recommander MULTI-SEED / plus d'échantillonnage. NE PAS déplacer la fenêtre.
      • Pente quasi uniforme (pas de coude détectable) → direction selon le RÉGIME : pente raide de
        type FONDU (CTE > cte_melt) → coude EN DESSOUS → 'plus bas' ; pente molle de type VERRE →
        coude AU-DESSUS → 'plus haut'.
    Renvoie {direction, message, suggested_prior_sim}."""
    T = np.asarray(temps, float); R = np.asarray(rho, float)
    order = np.argsort(T)[::-1]; T, R = T[order], R[order]      # haut → bas
    t_hi, t_lo = float(T[0]), float(T[-1])
    n = len(T); k = max(2, n // 3)
    m_hot = float(np.polyfit(T[:k], R[:k], 1)[0])               # pente côté haute-T (fondu attendu)
    m_cold = float(np.polyfit(T[-k:], R[-k:], 1)[0])            # pente côté basse-T (verre attendu)
    rho_mid = float(R.mean())
    contrast = abs(m_hot) / abs(m_cold) if m_cold else 99.0
    cte_hot = abs(m_hot) / rho_mid

    if contrast >= contrast_min:                               # coude réel présent dans la fenêtre
        return {"direction": "stabiliser",
                "message": f"un coude est PRÉSENT dans la fenêtre (contraste de pente {contrast:.1f}) "
                           f"mais le fit est instable — ce n'est PAS un problème de placement. "
                           f"Relance en MULTI-SEED (--seeds 3-5) et/ou augmente --sample-ps pour le stabiliser.",
                "suggested_prior_sim": None}
    if cte_hot > cte_melt:                                     # tout fondu → coude en dessous
        prior = round(t_lo, 0)
        return {"direction": "plus bas",
                "message": f"pente uniforme de type FONDU (CTE≈{cte_hot*1e4:.1f}e-4/K) → pas de coude, "
                           f"il est SOUS la fenêtre. Relance PLUS BAS : --tg-prior {prior:.0f} (≈ --tg-exp {prior/1.5:.0f}).",
                "suggested_prior_sim": prior}
    prior = round(t_hi, 0)                                     # tout verre → coude au-dessus
    return {"direction": "plus haut",
            "message": f"pente uniforme de type VERRE (CTE≈{cte_hot*1e4:.1f}e-4/K) → pas de coude, "
                       f"il est AU-DESSUS. Relance PLUS HAUT : --tg-prior {prior:.0f} (≈ --tg-exp {prior/1.5:.0f}).",
            "suggested_prior_sim": prior}


def wlf_shift(tg_sim, q_sim, q_exp=Q_EXP_DEFAULT, c1=C1_WLF, c2=C2_WLF):
    """Décale une Tg simulée (à la vitesse q_sim) vers la vitesse q_exp via WLF (Soldera Éq. 2).
    ΔTg = Tg_sim − Tg_exp = −C2·L/(C1+L), L = log10(q_exp/q_sim) < 0. Renvoie Tg extrapolée.
    Valide tant que |L| < C1 (~17 décades) — couvre le cas MD→DSC (~11-13 décades).
    """
    L = np.log10(q_exp / q_sim)
    if C1_WLF + L <= 0:
        return np.nan   # hors domaine WLF
    dTg = -c2 * L / (c1 + L)
    return tg_sim - dTg


def vft_tg_of_q(q, T0, B, A):
    """Forme VFT (Soldera Éq. 3) : Tg(q) = T0 − B/log10(A·q). T0 = Tg à vitesse nulle."""
    return T0 - B / np.log10(A * q)


def fit_vft_extrapolate(q_list, tg_list, q_exp=Q_EXP_DEFAULT):
    """Fit VFT sur l'échelle (q, Tg) puis extrapole à q_exp. Nécessite ≥4 points. Renvoie
    (Tg_extrapolée, params). Plus physique que WLF mais demande plus de vitesses."""
    q = np.asarray(q_list, float); tg = np.asarray(tg_list, float)
    if len(q) < 4:
        return np.nan, None
    # T0 ~ Tg_lent − marge ; B, A initiaux raisonnables.
    p0 = [tg.min() - 50, 500.0, 1e-12]
    try:
        popt, _ = curve_fit(vft_tg_of_q, q, tg, p0=p0, maxfev=20000)
        return float(vft_tg_of_q(q_exp, *popt)), popt.tolist()
    except Exception:
        return np.nan, None


def predict_tg(q_list, tg_list, q_exp=Q_EXP_DEFAULT):
    """Combine les deux extrapolations. WLF appliqué à CHAQUE vitesse (devrait être cohérent
    → moyenne ± écart-type), + VFT si ≥4 vitesses. Renvoie un dict de prédictions."""
    q = np.asarray(q_list, float); tg = np.asarray(tg_list, float)
    wlf_preds = np.array([wlf_shift(t, qi, q_exp) for t, qi in zip(tg, q)])
    wlf_preds = wlf_preds[np.isfinite(wlf_preds)]
    vft_pred, vft_par = fit_vft_extrapolate(q, tg, q_exp)
    return {
        "tg_wlf": float(np.mean(wlf_preds)) if len(wlf_preds) else np.nan,
        "tg_wlf_std": float(np.std(wlf_preds)) if len(wlf_preds) else np.nan,
        "tg_vft": vft_pred,
        "vft_params": vft_par,
        "tg_sim_per_rate": list(zip([float(x) for x in q], [float(x) for x in tg])),
    }
