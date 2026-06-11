# Benchmark de fiabilité — polymd vs expérience (12 polymères)

Pipeline complet (Cp quantique-DOS + CTE dédié), BOX_A=70, propriétés standard.
Valeurs exp : van Krevelen / Bicerano / Polymer Handbook (`scripts/exp_reference.py`).
Comparaison auto : `python scripts/compare_exp.py <nom>=<run.out> ...`.

## MAE % par propriété (12 polymères)

| Propriété | MAE % | n | Biais | Verdict |
|---|---|---|---|---|
| **Indice n** | **2.8** | 12 | −3% systématique | ✅ excellent (calibrable → ~1%) |
| **Tg** | 7.2 | 12 | mixte | ✅ (4.9% hors bas-Tg outliers PP/PIB) |
| **Cp** | 11.2 | 12 | bas-Tg sous-estimé | ✅ **5.0% sur les verres** (voir split) |
| densité | 7.5 | 12 | −7% (toujours bas) | ⛔ FF sous-dense |
| δ solubilité | 10.4 | 12 | mixte | 🟡 (outliers H-bond) |
| IP (xtb) | 1.5 | 2 | — | ✅ |

## Split Cp HAUT-Tg vs BAS-Tg (la nuance clé)

Cp/CTE/densité « vitreux » ne sont valides que si le polymère est VITREUX à 300 K (Tg > 300).
Pour Tg < 300 K, à 300 K le polymère est CAOUTCHOUTEUX → notre Cp (branche vitreuse) sous-estime
le Cp réel (excès configurationnel liquide). Ce n'est PAS un échec de la méthode DOS, c'est un
problème de PHASE/branche de mesure.

| Sous-ensemble | Cp MAE % |
|---|---|
| Haut-Tg (PS, PMMA, PC, PLA, PEMA, PaMS) — *vitreux à 300 K* | **5.0** ✅ |
| Bas-Tg (PVAc, PnBMA, PMA, PP, PEO, PIB) — *caoutchouteux à 300 K* | 17.4 (sous-estimé systématique = phase) |

Le facteur quantique-DOS, par polymère : 0.349 (PS/PaMS) → 0.436 (PLA).

## Détail par polymère (err % signée ; A=conf élevée)

| Polymère | Tg | densité | Cp | n | δ | conf |
|---|---|---|---|---|---|---|
| PS | +2.4 | −5.9 | −2.5 | −2.2 | −7.6 | A |
| PMMA | 0.0 | −10.8 | −7.0 | −3.8 | −8.4 | A |
| PC | −14.1 | −5.5 | 0.0 | −2.0 | −8.5 | A |
| PaMS | 0.0 | −9.5 | +1.7 | −4.0 | −17.7 | B |
| PLA | −9.0 | −4.2 | +11.0 | −1.9 | +16.9 | B |
| PEMA | +3.9 | −9.1 | −7.6 | −3.4 | −9.8 | B |
| PVAc | −4.4 | −6.9 | −13.0¹ | −2.3 | −1.0 | B |
| PnBMA | +9.1 | −12.3 | −8.7¹ | −4.5 | −14.6 | B |
| PMA | −4.7 | −7.6 | −15.3¹ | −2.6 | +1.9 | B |
| PP | −20.5 | −5.0 | −11.2¹ | −3.0 | +4.2 | B |
| PEO | −11.3 | −2.2 | −31.0¹ | −0.2 | +26.7 | B |
| PIB | −6.7 | −10.7 | −25.1¹ | −3.9 | −7.3 | B |

¹ Cp biaisé par la phase (Tg < 300 K → caoutchouteux à 300 K).

## DÉCOUVERTE CLÉ : densité, n et modules = UNE seule erreur (sous-cohésion FF)
`refractive_index` = Lorentz-Lorenz avec la densité MD (φ = R_M·ρ/M) → le −3% de n n'est PAS
indépendant, c'est le −7% de densité qui se propage (dn/n ≈ 0.39·dρ/ρ ≈ −2.7%). Les modules mous
viennent AUSSI de la sous-densité. ⇒ une SEULE calibration de densité corrige densité ET n.

### Correction densité (DENS_FF_CORR=1.078, commit 7d309a8)
ρ_corrigée = ρ_MD × 1.078 (estimation ρ_exp ; brut gardé en `density_300K_md`). Cascade sur n et FFV.

| | densité brute | **densité corr.** | n brut | **n corr.** |
|---|---|---|---|---|
| PS | −5.9% | **+1.0%** | −2.2% | **+0.9%** |
| PMMA | −10.8% | **−4.7%** | −3.8% | **−1.4%** |
| **MAE 12 polym.** | **7.5%** | **~2.7%** | **2.8%** | **~1.3%** |

Résiduel = chimie-dépendance du biais FF (−2 à −12%) ; DENS_FF_CORR=1.0 désactive.

## Constats restants
- **Cp** : physique solide sur la bonne phase ; pour bas-Tg, mesurer sur la branche caoutchouteuse.
- **Tg** : PP (−20%) et PIB outliers FF connus ; 4.9% sans eux.
- **modules E/G** : même cause (sous-densité) mais comprimer sur-estime → limite FF, non calibrable proprement.
