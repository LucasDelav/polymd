"""Matrice de référence EXPÉRIMENTALE pour valider/calibrer le pipeline polymd.

Valeurs de littérature (van Krevelen "Properties of Polymers", Bicerano "Prediction of
Polymer Properties", Polymer Handbook (Brandrup/Immergut), Mark "Physical Properties of
Polymers Handbook"). Toutes à T ambiante (~298 K) sauf indication. AMORPHE quand la
distinction cristallin/amorphe compte (densité, κ, Cp dépendent de la cristallinité).

⚠ Niveaux de confiance (conf) :
  'A' = bien établi, faible dispersion inter-sources (Tg, densité, n, δ).
  'B' = correct mais dispersion notable (Cp, κ, ε — sensibles méthode/cristallinité).
  'C' = très dispersé / dépendant procédé (E module, CTE) ou rare (IP solide).
Unités : density g/cm³ ; Tg K ; Cp J/g/K ; kappa W/m/K ; n sans ; E GPa ; poisson sans ;
delta MPa^0.5 ; eps_static sans (~1 kHz, sec) ; cte_lin_glass 1e-4/K (linéaire, vitreux) ;
ip_eV (potentiel d'ionisation, ~onset photoémission / 1ère IP, souvent gaz — peu fiable solide).

Clés = noms de md_build.REFERENCE_POLYMERS. None = valeur non renseignée (à compléter).
"""

EXP = {
    # name:            density  Tg    Cp     kappa  n      E      poisson delta  eps   cte_lin  ip
    "polyethylene":   dict(density=0.855, Tg=195,  Cp=1.90, kappa=0.20, n=1.51, E=0.30, poisson=0.45, delta=16.2, eps=2.3,  cte_lin_glass=2.0, ip=8.7,  conf="B"),  # amorphe; Tg ambiguë
    "polypropylene":  dict(density=0.855, Tg=260,  Cp=1.70, kappa=0.17, n=1.49, E=1.30, poisson=0.42, delta=16.8, eps=2.2,  cte_lin_glass=1.0, ip=None, conf="B"),  # atactique amorphe
    "polyisobutylene":dict(density=0.918, Tg=200,  Cp=1.95, kappa=0.13, n=1.51, E=None, poisson=None, delta=16.4, eps=2.2,  cte_lin_glass=None,ip=None, conf="B"),
    "polybutadiene":  dict(density=0.900, Tg=170,  Cp=1.89, kappa=0.13, n=1.52, E=None, poisson=None, delta=17.0, eps=2.5,  cte_lin_glass=None,ip=None, conf="B"),  # cis-1,4
    "polyisoprene":   dict(density=0.910, Tg=203,  Cp=1.88, kappa=0.13, n=1.52, E=None, poisson=0.49, delta=16.5, eps=2.4,  cte_lin_glass=None,ip=None, conf="B"),  # caoutchouc naturel
    "polystyrene":    dict(density=1.050, Tg=373,  Cp=1.22, kappa=0.15, n=1.59, E=3.20, poisson=0.33, delta=18.5, eps=2.5,  cte_lin_glass=0.7, ip=8.45, conf="A"),
    "PaMS":           dict(density=1.075, Tg=443,  Cp=1.20, kappa=0.15, n=1.61, E=3.50, poisson=0.34, delta=18.6, eps=2.6,  cte_lin_glass=None,ip=None, conf="B"),  # poly(α-méthylstyrène)
    "PMMA":           dict(density=1.180, Tg=378,  Cp=1.42, kappa=0.19, n=1.49, E=2.90, poisson=0.37, delta=19.0, eps=3.3,  cte_lin_glass=0.6, ip=None, conf="A"),
    "PEMA":           dict(density=1.115, Tg=338,  Cp=1.45, kappa=0.18, n=1.485,E=2.50, poisson=0.37, delta=18.4, eps=3.0,  cte_lin_glass=None,ip=None, conf="B"),  # poly(éthacrylate de méthyle)→méthacrylate d'éthyle
    "PnBMA":          dict(density=1.055, Tg=293,  Cp=1.50, kappa=0.18, n=1.483,E=1.80, poisson=0.40, delta=17.8, eps=3.0,  cte_lin_glass=None,ip=None, conf="B"),  # poly(méthacrylate de n-butyle)
    "PMA":            dict(density=1.220, Tg=283,  Cp=1.50, kappa=0.18, n=1.479,E=None, poisson=None, delta=20.7, eps=3.5,  cte_lin_glass=None,ip=None, conf="B"),  # poly(acrylate de méthyle)
    "PVAc":           dict(density=1.190, Tg=305,  Cp=1.46, kappa=0.16, n=1.467,E=2.00, poisson=0.40, delta=21.0, eps=3.2,  cte_lin_glass=None,ip=None, conf="B"),  # poly(acétate de vinyle)
    "polycarbonate":  dict(density=1.200, Tg=418,  Cp=1.20, kappa=0.20, n=1.585,E=2.30, poisson=0.37, delta=20.0, eps=3.0,  cte_lin_glass=0.65,ip=None, conf="A"),  # BPA-PC
    "PEO":            dict(density=1.130, Tg=213,  Cp=2.00, kappa=0.20, n=1.46, E=None, poisson=None, delta=20.2, eps=5.0,  cte_lin_glass=None,ip=10.0, conf="B"),  # poly(oxyde d'éthylène), amorphe ~1.06-1.13
    "PLA":            dict(density=1.250, Tg=328,  Cp=1.18, kappa=0.16, n=1.46, E=3.30, poisson=0.39, delta=19.5, eps=3.0,  cte_lin_glass=None,ip=None, conf="B"),  # poly(acide lactique)
}

# Propriétés DÉRIVÉES utiles à la comparaison (calculées depuis EXP, pas saisies)
def cte_vol_glass_ppmK(name):
    """CTE volumétrique (ppm/K) ≈ 3× linéaire, pour comparer à CTE_glass_ppmK du pipeline."""
    v = EXP.get(name, {}).get("cte_lin_glass")
    return None if v is None else round(v * 1e-4 * 3 * 1e6, 1)

# Mapping clé_exp → clé_pipeline (props.json) pour l'auto-comparaison
PIPELINE_KEY = {
    "density": "density_300K", "Tg": "Tg_pred", "Cp": "Cp_JgK",
    "kappa": "thermal_conductivity_WmK", "n": "refractive_index", "E": "E_GPa",
    "poisson": "poisson", "delta": "solubility_delta", "eps": "static_dielectric",
    "ip": "ionization_potential_eV",
}
