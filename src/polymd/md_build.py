"""Construction d'un bulk polymère amorphe périodique pour la MD.

Étapes :
  1. `build_oligomer` : polymérise un motif (PSMILES à 2 `*`) en oligomère linéaire
     de N unités tête-à-queue, bouts cappés H, conformère 3D (RDKit).
  2. `pack_box` : réplique M chaînes sur une grille 3D dans une boîte cubique
     périodique à basse densité (orientations aléatoires) → ASE Atoms (PBC).

La basse densité initiale sera compressée ensuite par MD NPT vers la densité réelle.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RANDOM_SEED = 42        # graine de l'embedding du conformère initial (reproductibilité)

RDLogger.DisableLog("rdApp.*")


def build_oligomer(repeat_smiles: str, n_units: int, tacticity: str = "atactic") -> Chem.Mol:
    """Polymérise un motif (2 atomes `*`) en oligomère linéaire de n_units unités.

    tacticity : 'atactic' (stéréo aléatoire, défaut), 'iso' (tous les centres chiraux du
    squelette de même configuration) ou 'syndio' (configurations alternées). Permet de
    tester l'effet ÉNORME de la tacticité sur Tg (ex. PMMA iso 45°C vs syndio 130°C).
    """
    unit = Chem.MolFromSmiles(repeat_smiles)
    if unit is None:
        raise ValueError(f"SMILES invalide : {repeat_smiles}")
    dummies = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise ValueError(f"motif attendu avec 2 points d'attache `*`, trouvé {len(dummies)}")

    # Voisin (atome du squelette) de chaque dummy.
    def neighbor(mol, d):
        return mol.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx()

    head_d, tail_d = dummies              # head = bout gauche, tail = bout droit
    nat = unit.GetNumAtoms()

    # Concatène n copies du motif.
    combo = unit
    for _ in range(n_units - 1):
        combo = Chem.CombineMols(combo, unit)
    rw = Chem.RWMol(combo)

    to_remove = []
    for i in range(n_units):
        off = i * nat
        if i < n_units - 1:  # lier la queue de l'unité i à la tête de l'unité i+1
            tail_nb = neighbor(unit, tail_d) + off
            head_nb = neighbor(unit, head_d) + (i + 1) * nat
            rw.AddBond(tail_nb, head_nb, Chem.BondType.SINGLE)
            to_remove += [tail_d + off, head_d + (i + 1) * nat]  # dummies internes

    # Dummies terminaux restants (tête unité 0, queue unité n-1) → hydrogènes.
    for term in (head_d, tail_d + (n_units - 1) * nat):
        rw.GetAtomWithIdx(term).SetAtomicNum(1)

    for idx in sorted(set(to_remove), reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    # Tacticité : assigne une configuration aux centres chiraux NON DÉFINIS, ORDONNÉS le long de
    # la chaîne (par indice d'atome ≈ ordre de construction séquentiel). iso = tous CW ; syndio =
    # CW/CCW alternés ; atactic = aléatoire mais DÉFINI (graine fixe → reproductible).
    #
    # ON PRÉSERVE la stéréo DÉJÀ DÉFINIE (ex. dessinée dans Ketcher → @/@@ dans le PSMILES, conservés
    # par cli.normalize_psmiles) : seuls les centres laissés indéterminés ('?') sont remplis. Ainsi
    # un motif stéréo-spécifié garde sa chiralité ; un motif sans stéréo est complété (atactic).
    #
    # PIÈGE CORRIGÉ (hang infini) : tout centre INDÉFINI restant ferait échantillonner la chiralité
    # par ETKDG ; sur un motif à cycles FUSIONNÉS (ex. acétal bicyclique *OCC1OC2OC(C)(C)OC2C1*) il
    # tire alors des jonctions de cycle *trans* géométriquement impossibles à chacune des ~N unités
    # → la distance-geometry échoue et re-tente sans fin → embed bloqué à 100 % CPU en C++ (le timeout
    # SIGALRM de l'appelant est impuissant face à un appel C++ qui ne rend pas la main). En remplissant
    # tous les '?', on supprime cet échantillonnage interne → embed déterministe et fini.
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    undefined = [i for i, lab in centers if lab == "?"]
    has_defined = any(lab in ("R", "S") for _, lab in centers)   # stéréo dessinée → à préserver
    # Cycles FUSIONNÉS (atome dans ≥2 cycles, ex. acétal bicyclique) : pour ceux-là, l'embed SANS
    # imposer la chiralité est PLUS lent/instable (ETKDG explore librement des jonctions infaisables)
    # → on garde l'embed imposé (avec _assign_stereo) qui leur donne une cible cis claire. Le fast-path
    # ne s'applique qu'aux chaînes à cycles ISOLÉS/pendants (phényl du PS) ou sans cycle.
    uri = unit.GetRingInfo()
    fused_rings = any(uri.NumAtomRings(i) >= 2 for i in range(unit.GetNumAtoms()))

    def _assign_stereo(attempt: int) -> None:
        rng = np.random.default_rng(RANDOM_SEED + attempt)
        for k, ci in enumerate(sorted(undefined)):
            if tacticity == "iso":
                cw = True
            elif tacticity == "syndio":
                cw = (k % 2 == 0)
            else:                                  # atactic : aléatoire mais défini
                cw = bool(rng.integers(2))
            mol.GetAtomWithIdx(ci).SetChiralTag(
                Chem.ChiralType.CHI_TETRAHEDRAL_CW if cw else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Conformère initial COMPACT (pelote globulaire) : useRandomCoords + MMFF. (Cf compact-builder :
    # un ETKDG étendu → boîte énorme quasi-vide → MD lente ; seule la COMPACITÉ compte, la MD relaxe.)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.useRandomCoords = True
    params.maxIterations = 400

    if tacticity == "atactic" and not has_defined and not fused_rings:
        # ── FAST PATH (défaut : atactique, aucune stéréo dessinée, pas de cycles fusionnés) ──
        # IMPOSER la chiralité sur 40+ centres de la chaîne rend l'embed PATHOLOGIQUE (PS 40-mère :
        # >5 min, jusqu'à ~50 min avec les retries). Or pour de l'atactique sans stéréo à préserver,
        # inutile de l'imposer : enforceChirality=False → embed ~25× plus rapide (PS ~2 min) et ETKDG
        # produit une chiralité ALÉATOIRE par centre = atactique par nature. On re-perçoit la stéréo
        # depuis la 3D (tags ↔ géométrie). BONUS : supprime aussi le hang des cycles fusionnés (plus
        # d'échantillonnage de jonctions trans impossibles).
        params.enforceChirality = False
        if AllChem.EmbedMolecule(mol, params) != 0:        # très rare → repli imposé
            params.enforceChirality = True
            _assign_stereo(0); AllChem.EmbedMolecule(mol, params)
        else:
            Chem.AssignStereochemistryFrom3D(mol)
    else:
        # ── iso/syndio OU stéréo dessinée : on IMPOSE les tags (avec retries + repli iso tout-CW) ──
        # Un tirage atactic+défini peut échouer (rc != 0) → on retente avec une autre graine stéréo.
        n_try = 1 if tacticity in ("iso", "syndio") else 4
        embedded = False
        for attempt in range(n_try):
            _assign_stereo(attempt)
            if AllChem.EmbedMolecule(mol, params) == 0:
                embedded = True
                break
        if not embedded:
            for ci in undefined:
                mol.GetAtomWithIdx(ci).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _mol_to_arrays(mol: Chem.Mol):
    conf = mol.GetConformer()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = conf.GetPositions()
    return syms, pos - pos.mean(0)  # centré


def pack_box(mol: Chem.Mol, n_chains: int, margin: float = 3.0,
             init_density: float | None = None, seed: int = 42) -> Atoms:
    """Place n_chains chaînes sur une grille, une par maille, SANS chevauchement.

    L'espacement de grille = diamètre de la chaîne (2·rayon) + marge → chaque
    chaîne reste confinée dans sa maille. La densité initiale émerge de cette
    géométrie (basse) et sera compressée par NPT. `init_density` est ignoré
    (conservé pour compatibilité d'appel).
    """
    syms, pos = _mol_to_arrays(mol)            # pos déjà centré
    radius = float(np.sqrt((pos ** 2).sum(1)).max())  # rayon englobant
    spacing = 2 * radius + margin
    grid = int(np.ceil(n_chains ** (1 / 3)))
    box_A = grid * spacing
    rng = np.random.default_rng(seed)

    all_sym, all_pos = [], []
    placed = 0
    for ix in range(grid):
        for iy in range(grid):
            for iz in range(grid):
                if placed >= n_chains:
                    break
                center = (np.array([ix, iy, iz]) + 0.5) * spacing
                q = rng.normal(size=4); q /= np.linalg.norm(q)
                all_pos.append(pos @ _quat_to_mat(q).T + center)
                all_sym += syms
                placed += 1
    atoms = Atoms(symbols=all_sym, positions=np.vstack(all_pos),
                  cell=[box_A] * 3, pbc=True)
    return atoms


def estimate_density(mol: Chem.Mol, packing: float = 0.64) -> float:
    """Densité estimée (g/cm³) à partir de la STRUCTURE seule : volume de van der Waals
    (grille RDKit) divisé par une compacité typique des organiques amorphes (~0.64).

    Sert à dimensionner la boîte et la cible de compression SANS donnée expérimentale
    (l'expérience ne sert plus qu'à *comparer* le résultat). 1 uma/Å³ = 1.66054 g/cm³.
    """
    v_vdw = AllChem.ComputeMolVolume(mol)                  # Å³ (inclut les H)
    m_amu = sum(a.GetMass() for a in mol.GetAtoms())
    return 1.66053907 * packing * m_amu / v_vdw


def chain_mass(mol: Chem.Mol) -> float:
    """Masse d'une chaîne (uma, H compris)."""
    return float(sum(a.GetMass() for a in mol.GetAtoms()))


def n_chains_for_box(mol: Chem.Mol, box_target_A: float, rho_est: float) -> int:
    """Nombre de chaînes pour qu'une boîte cubique d'arête `box_target_A` ait la densité
    `rho_est`. ρ = 1.66054·n·m_chain/L³ → n = ρ·L³/(1.66054·m_chain). DÉRIVE la taille
    du système de l'entrée : plus le motif est lourd, moins de chaînes — boîte constante.
    """
    n = rho_est * box_target_A ** 3 / (1.66053907 * chain_mass(mol))
    return max(1, int(round(n)))


def apply_hmr(atoms: Atoms, h_mass: float = 3.0) -> Atoms:
    """Hydrogen Mass Repartitioning : alourdit les H (→ h_mass uma) en prélevant la
    masse sur l'atome lourd voisin (masse totale conservée). Permet un pas de temps
    2-4× plus grand. N'affecte PAS les propriétés d'équilibre (densité, Rg, modules).

    Affectation H→lourd par distance (liaison la plus proche < 1.3 Å). À appeler sur
    la structure fraîchement construite (non repliée par PBC).
    """
    masses = atoms.get_masses().copy()
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols())
    h_idx = np.where(sym == "H")[0]
    heavy_idx = np.where(sym != "H")[0]
    if len(heavy_idx) == 0:
        return atoms
    heavy_pos = pos[heavy_idx]
    for i in h_idx:
        d = np.linalg.norm(heavy_pos - pos[i], axis=1)
        j = heavy_idx[int(d.argmin())]
        add = h_mass - masses[i]
        masses[i] = h_mass
        masses[j] -= add
    atoms.set_masses(masses)
    return atoms


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# Motifs de référence pour la validation (Tg expérimentale connue). 7 polymères très
# étudiés, diversité chimie + gamme de Tg (170-418 K), tous C/H/O (compatibles OpenFF).
REFERENCE_POLYMERS = {
    "polybutadiene": {"smiles": "*CC=CC*", "Tg_exp_C": -103},        # cis-1,4, ~170 K
    "polyisobutylene": {"smiles": "*CC(C)(C)*", "Tg_exp_C": -73},    # ~200 K
    "polypropylene": {"smiles": "*CC(*)C", "Tg_exp_C": -13},         # atactique, ~260 K
    "PVAc": {"smiles": "*CC(*)OC(C)=O", "Tg_exp_C": 32},             # acétate de vinyle, ~305 K
    "polystyrene": {"smiles": "*CC(*)c1ccccc1", "Tg_exp_C": 100},    # ~373 K
    "PMMA": {"smiles": "*CC(*)(C)C(=O)OC", "Tg_exp_C": 105},         # ~378 K
    "polycarbonate": {"smiles": "*OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(*)cc1", "Tg_exp_C": 145},  # BPA, ~418 K
    "polyethylene": {"smiles": "*CC*", "Tg_exp_C": -120},            # gardé (Tg ambiguë, hors set)
    # --- 2e lot (14 polymères au total) : diversité chimique pour corréler le facteur ÷1.50 ---
    "polyisoprene": {"smiles": "*CC(C)=CC*", "Tg_exp_C": -70},       # diène + méthyle, ~203 K
    "PEO": {"smiles": "*CCO*", "Tg_exp_C": -60},                     # éther flexible, ~213 K
    "PMA": {"smiles": "*CC(*)C(=O)OC", "Tg_exp_C": 10},              # acrylate (sans α-méthyle), ~283 K
    "PnBMA": {"smiles": "*CC(*)(C)C(=O)OCCCC", "Tg_exp_C": 20},      # méthacrylate chaîne longue, ~293 K
    "PLA": {"smiles": "*OC(C)C(=O)*", "Tg_exp_C": 55},               # ester de squelette, ~328 K
    "PEMA": {"smiles": "*CC(*)(C)C(=O)OCC", "Tg_exp_C": 65},         # méthacrylate éthyle, ~338 K
    "PaMS": {"smiles": "*CC(*)(C)c1ccccc1", "Tg_exp_C": 170},        # α-méthylstyrène, aromatique haut-Tg ~443 K
}
