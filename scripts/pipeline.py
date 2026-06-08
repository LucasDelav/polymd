"""Pipeline UNIFIÉ MD-FF : un SMILES de monomère → propriétés thermiques + mécaniques.
Un seul GPU, ~15-18 min (boîte 48k atomes — voir le plancher SNR ci-dessous).

SORTIES :
  Tg_pred   : Tg expérimentale prédite = Tg_sim / 1.50  (correction CINÉTIQUE universelle,
              MAE ~26 K sur 14 polymères, sans dépendance chimique ; cf project-md-pipeline).
  density_300K, CTE_glass : densité et dilatation thermique (gratuit, depuis ρ(T)).
  K_GPa     : module de compression (fluctuations de volume — FIABLE).
  E/ν/G     : traction uniaxiale stress-strain (latéral relaxé) — MECH_TENSILE=1 (défaut). Cross-check
              interne K=E/(3(1−2ν)) vs K_fluct. (Remplace l'ancien energy-strain cassé.)

MÉTHODE : build compact + OpenFF Sage + HMR 4fs → compression → fonte → refroidissement PAR PALIERS
(P1 : 20 K/palier, 100+50 ps) sur fenêtre étroite centrée sur 1.5×Tg_exp → fit hyperbolique de ρ(T)
→ Tg_sim → ÷1.50. PLANCHER : ~48k atomes nécessaires (le coude ρ(T) est un signal faible, SNR
∝ √(atomes×temps) ; sous 48k le fit latche). Donc pas plus rapide sans perdre la robustesse.

USAGE :
  SMILES='*CC(*)c1ccccc1' TG_EXP=373 FF_CUDA=1 conda_openff_gpu/bin/python -u scripts/pipeline.py
  (CRIANN : via run_pipeline.slurm + sbatch --export ; cf project-md-pipeline pour le quoting SMILES.)
  Pour un polymère INCONNU : donner TG_EXP = estimation (contribution de groupes) ; la fenêtre
  s'auto-centre sur 1.5×TG_EXP. Le SMILES doit avoir 2 points d'attache `*` (PSMILES).

PARAMS (env, défauts entre parenthèses) : BOX_A(80=~48k at), TG_SIM_PRIOR(1.5×TG_EXP),
WIN_HI(80)/WIN_LO(140), T_STEP(20), EQUIL_PS(100), SAMPLE_PS(50), N_UNITS(40), MECH(1=K),
SHEAR(0 ; =1 active G/E/ν expérimentaux), COOL_INDEP(0 ; =1 paliers indépendants).
Profiler intégré : breakdown temporel par étape en fin de run.
"""
import os, sys, time, json
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tg_ml import md_build, tg_kinetics


# ───────────────────────── Profiler ─────────────────────────
class Profiler:
    def __init__(self):
        self.t = defaultdict(float); self.n = defaultdict(int); self.t0 = time.time()

    def block(self, label):
        prof = self
        class _C:
            def __enter__(self): self.s = time.time()
            def __exit__(self, *a):
                prof.t[label] += time.time() - self.s; prof.n[label] += 1
        return _C()

    def report(self):
        tot = time.time() - self.t0
        print("\n" + "=" * 60); print(f"  PROFIL TEMPOREL (total {tot:.1f}s = {tot/60:.1f}min)")
        print(f"  {'étape':28s} {'appels':>7s} {'temps(s)':>9s} {'%':>6s}")
        for lab in sorted(self.t, key=lambda k: -self.t[k]):
            print(f"  {lab:28s} {self.n[lab]:7d} {self.t[lab]:9.1f} {100*self.t[lab]/tot:5.0f}%")
        print("=" * 60)


P = Profiler()


def compute_chain_dims(pos, boxL, omm_top, n_chains):
    """Dimensions de chaîne depuis une config MD (chaînes dépliées à travers la PBC) :
    Rg (rayon de giration, nm), Ree (distance bout-à-bout, nm) et C∞ (ratio caractéristique).
    Les bouts = extrémités du DIAMÈTRE du graphe de liaisons (2 BFS) ; C∞ = ⟨Ree²⟩/(N_bb·l²)
    avec N_bb et l (longueur de liaison) mesurés le long du chemin squelette. Boîte cubique."""
    adj = defaultdict(list)
    for b in omm_top.bonds():
        adj[b.atom1.index].append(b.atom2.index)
        adj[b.atom2.index].append(b.atom1.index)
    N = len(pos); apc = N // n_chains
    rgs, rees, cinfs = [], [], []

    def bfs(src, lo, hi, parent=None):
        dist = {src: 0}; par = {src: None}; q = [src]; far = src
        while q:
            nq = []
            for u in q:
                for v in adj[u]:
                    if lo <= v < hi and v not in dist:
                        dist[v] = dist[u] + 1; par[v] = u; nq.append(v)
                        if dist[v] > dist[far]:
                            far = v
            q = nq
        return (far, dist, par) if parent else (far, dist)

    for c in range(n_chains):
        lo, hi = c * apc, (c + 1) * apc
        unwrapped = {lo: pos[lo].copy()}; stack = [lo]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if lo <= j < hi and j not in unwrapped:
                    d = pos[j] - pos[i]
                    d -= np.round(d / boxL) * boxL          # image minimale (cubique)
                    unwrapped[j] = unwrapped[i] + d
                    stack.append(j)
        arr = np.array([unwrapped.get(k, pos[k]) for k in range(lo, hi)])
        cm = arr.mean(axis=0)
        rgs.append(float(np.sqrt(((arr - cm) ** 2).sum(axis=1).mean())))
        # diamètre (2 BFS) → bouts a,b
        a, _ = bfs(lo, lo, hi)
        b, _, par = bfs(a, lo, hi, parent=True)
        path = []; v = b
        while v is not None:
            path.append(v); v = par[v]
        ree = float(np.linalg.norm(unwrapped[a] - unwrapped[b]))
        n_bb = len(path) - 1
        rees.append(ree)
        if n_bb > 0:
            l = float(np.mean([np.linalg.norm(unwrapped[path[k]] - unwrapped[path[k + 1]])
                               for k in range(n_bb)]))           # longueur de liaison squelette
            cinfs.append(ree ** 2 / (n_bb * l ** 2))
    return (float(np.mean(rgs)), float(np.mean(rees)),
            float(np.mean(cinfs)) if cinfs else None)


def compute_ced(pos, boxL, omm_top, n_chains, single_sim, e_bulk_kjmol, V_m3):
    """CED (MPa) et δ=√CED (MPa^0.5) par la méthode CONFORMATION GELÉE : E_coh = Σ E_molécule(coords
    du bulk, vide) − E_bulk. On évalue chaque chaîne SEULE à sa conformation du bulk (pas relaxée →
    pas d'effondrement en pelote qui sous-estimerait E_coh, le piège de l'ancien ced_diag). Chaînes
    dépliées (PBC) avant l'éval en vide."""
    from openmm import unit as ou
    adj = defaultdict(list)
    for b in omm_top.bonds():
        adj[b.atom1.index].append(b.atom2.index)
        adj[b.atom2.index].append(b.atom1.index)
    apc = len(pos) // n_chains
    e_mol_sum = 0.0
    for c in range(n_chains):
        lo, hi = c * apc, (c + 1) * apc
        unwrapped = {lo: pos[lo].copy()}; stack = [lo]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if lo <= j < hi and j not in unwrapped:
                    d = pos[j] - pos[i]; d -= np.round(d / boxL) * boxL
                    unwrapped[j] = unwrapped[i] + d; stack.append(j)
        coords = np.array([unwrapped.get(k, pos[k]) for k in range(lo, hi)])
        single_sim.context.setPositions(coords * ou.nanometer)
        e_mol_sum += single_sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            ou.kilojoule_per_mole)
    e_coh_J = (e_mol_sum - e_bulk_kjmol) * 1000.0 / 6.02214076e23     # J par boîte (>0)
    ced_mpa = e_coh_J / V_m3 / 1e6                                    # J/m³ = Pa → MPa
    return ced_mpa, (max(ced_mpa, 0.0)) ** 0.5


def main():
    smiles = os.environ.get("SMILES", "*CC(*)c1ccccc1")
    tg_exp = float(os.environ.get("TG_EXP", "373"))
    n_units = int(os.environ.get("N_UNITS", "40"))
    box_target = float(os.environ.get("BOX_A", "80"))
    do_mech = os.environ.get("MECH", "1") == "1"
    seed = int(os.environ.get("SEED", "1"))   # graine aléatoire (packing + vitesses + intégrateur)
    prior = float(os.environ.get("TG_SIM_PRIOR", str(1.5 * tg_exp)))
    t_high = prior + float(os.environ.get("WIN_HI", "80"))
    t_low = max(prior - float(os.environ.get("WIN_LO", "140")), 120.0)
    t_anneal = t_high + 150.0
    t_step = float(os.environ.get("T_STEP", "20"))
    equil_ps = float(os.environ.get("EQUIL_PS", "100"))
    sample_ps = float(os.environ.get("SAMPLE_PS", "50"))
    # TG_DESC : libellé lisible fourni par l'appelant (ex. "plage Tg_exp 400-500 K"). À défaut,
    # on retombe sur Tg_exp= (cohérent en mode estimation ponctuelle).
    tg_label = os.environ.get("TG_DESC") or f"Tg_exp={tg_exp:.0f}K"
    print(f"=== PIPELINE | {smiles} | {tg_label} | prior={prior:.0f}K | "
          f"fenêtre {t_high:.0f}→{t_low:.0f}K ===", flush=True)

    with P.block("01_build"):
        # GARDE-FOU : timeout sur la construction de l'oligomère (RDKit ETKDG/MMFF peut être très lent
        # sur gros groupes latéraux, voire hang infini sur certains motifs type nitrile). Évite de
        # brûler tout le walltime GPU sur un build pathologique → échec propre en ~BUILD_TIMEOUT s.
        import signal
        def _bto(s, f):
            raise TimeoutError(f"build_oligomer > {os.environ.get('BUILD_TIMEOUT', '600')}s")
        signal.signal(signal.SIGALRM, _bto)
        signal.alarm(int(os.environ.get("BUILD_TIMEOUT", "600")))
        mol = md_build.build_oligomer(smiles, n_units, "atactic")
        signal.alarm(0)
        rho_est = md_build.estimate_density(mol)
        n_chains = md_build.n_chains_for_box(mol, box_target, rho_est)
        v_vdw_chain = float(md_build.AllChem.ComputeMolVolume(mol))   # Å³ (pour FFV)
        m_chain = md_build.chain_mass(mol)                            # uma
        from rdkit.Chem import Crippen
        mr_chain = float(Crippen.MolMR(mol))                          # réfraction molaire (pour l'indice)
    with P.block("02_pack_box"):
        atoms = md_build.pack_box(mol, n_chains, margin=3.0, seed=seed)
        box_A = float(atoms.cell[0, 0]); pos_A = atoms.get_positions()
    print(f"  {n_chains} chaînes, {len(atoms)} atomes, ρ_est={rho_est:.3f}", flush=True)

    from openff.toolkit import Molecule, ForceField, Topology
    from openff.units import unit as offunit
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
    with P.block("03_charges_nagl"):
        offmol = Molecule.from_rdkit(md_build.Chem.AddHs(mol), allow_undefined_stereo=True)
        offmol.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt", toolkit_registry=NAGLToolkitWrapper())
    with P.block("04_parametrize"):
        ff = ForceField("openff-2.2.0.offxml")
        top = Topology.from_molecules([offmol] * n_chains)
        top.box_vectors = np.eye(3) * box_A * offunit.angstrom
        omm_system = ff.create_interchange(top, charge_from_molecules=[offmol]).to_openmm()
        omm_top = top.to_openmm()

    import openmm
    from openmm import unit, LangevinMiddleIntegrator, MonteCarloBarostat, NonbondedForce
    from openmm.app import Simulation
    with P.block("05_hmr"):
        for bond in omm_top.bonds():
            a1, a2 = bond.atom1, bond.atom2
            s1 = a1.element.symbol if a1.element else ""; s2 = a2.element.symbol if a2.element else ""
            if (s1 == "H") != (s2 == "H"):
                h, hv = (a1, a2) if s1 == "H" else (a2, a1)
                mh = omm_system.getParticleMass(h.index).value_in_unit(unit.dalton)
                mv = omm_system.getParticleMass(hv.index).value_in_unit(unit.dalton)
                omm_system.setParticleMass(h.index, 4.0 * unit.dalton)
                omm_system.setParticleMass(hv.index, (mv - (4.0 - mh)) * unit.dalton)
    plat = openmm.Platform.getPlatformByName("CUDA") if os.environ.get("FF_CUDA", "1") == "1" \
        else openmm.Platform.getPlatformByName("CPU")
    mass_g = sum(a.element.mass.value_in_unit(unit.dalton) for a in omm_top.atoms()) / 6.02214076e23
    DT = 4.0; spp = int(round(1000.0 / DT))

    def Lnm(s):
        return float(s.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)[0, 0])

    def rho(s):
        V = s.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) * 1e-21
        return mass_g / V

    # Compression staged au recuit.
    with P.block("06_compress"):
        rho_c = 0.82 * rho_est
        ia = LangevinMiddleIntegrator(t_anneal * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds)
        ia.setRandomNumberSeed(seed)
        sA = Simulation(omm_top, omm_system, ia, plat)
        sA.context.setPositions(pos_A * 0.1 * unit.nanometer)
        sA.context.setPeriodicBoxVectors(*(np.eye(3) * box_A * 0.1))
        sA.minimizeEnergy(maxIterations=2000)
        Ltar = (mass_g / rho_c * 1e21) ** (1 / 3)
        L0 = Lnm(sA); ratio = (Ltar / L0) ** (1 / 30)
        for s in range(1, 31):
            Ls = L0 * ratio ** s
            p = sA.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer) * (Ls / Lnm(sA))
            sA.context.setPeriodicBoxVectors(*(np.eye(3) * Ls)); sA.context.setPositions(p * unit.nanometer)
            sA.minimizeEnergy(maxIterations=200); sA.step(300)
        pc = sA.context.getState(getPositions=True).getPositions(asNumpy=True)
        bc = sA.context.getState().getPeriodicBoxVectors()

    baro = MonteCarloBarostat(1.0 * unit.bar, t_anneal * unit.kelvin, 25); omm_system.addForce(baro)
    ig = LangevinMiddleIntegrator(t_anneal * unit.kelvin, 1.0 / unit.picosecond, DT * unit.femtoseconds)
    ig.setRandomNumberSeed(seed)
    sim = Simulation(omm_top, omm_system, ig, plat)
    sim.context.setPeriodicBoxVectors(*bc); sim.context.setPositions(pc)
    sim.context.setVelocitiesToTemperature(t_anneal * unit.kelvin, seed)
    with P.block("07_anneal_melt"):
        sim.step(150 * spp)
        ig.setTemperature(t_high * unit.kelvin); sim.context.setParameter(baro.Temperature(), t_high)
        sim.step(150 * spp)
        # Fonte longue OPTIONNELLE : relaxe les dimensions de chaîne (Rg/Ree/C∞ sinon effondrés par
        # le conformère compact initial). Relaxation ∝ N² → coûteux. Défaut 0 (Tg/densité OK sans).
        extra_melt = float(os.environ.get("EXTRA_MELT_PS", "0"))
        if extra_melt > 0:
            sim.step(int(extra_melt * spp))
    # Snapshot de fonte (pour le mode indépendant/parallélisable).
    melt_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    melt_box = sim.context.getState().getPeriodicBoxVectors()
    # Dimensions de chaîne (gratuit, depuis la config de fonte équilibrée) : Rg, Ree, C∞.
    with P.block("07b_chain_dims"):
        mp = melt_pos.value_in_unit(unit.nanometer)
        boxL = np.diag(np.array(melt_box.value_in_unit(unit.nanometer)))
        rg_nm, ree_nm, cinf = compute_chain_dims(mp, boxL, omm_top, n_chains)

    # Refroidissement → ρ(T). COOL_INDEP=1 : chaque palier repart du SNAPSHOT DE FONTE (indépendant
    # → parallélisable sur N GPU, ~N× plus rapide). Sinon séquentiel (chaque T part du précédent).
    indep = os.environ.get("COOL_INDEP", "0") == "1"
    temps = np.arange(t_high, t_low - 1, -t_step)
    eff_rate = t_step / ((equil_ps + sample_ps) * 1e-12)
    R = np.full(len(temps), np.nan)
    U = np.full(len(temps), np.nan)   # énergie totale ⟨U⟩ par palier (pour Cp = dU/dT)
    for k, Tk in enumerate(temps):
        with P.block("08_cool_palier"):
            if indep:
                sim.context.setPeriodicBoxVectors(*melt_box); sim.context.setPositions(melt_pos)
                sim.context.setVelocitiesToTemperature(t_high * unit.kelvin, k * 13 + seed)
            ig.setTemperature(float(Tk) * unit.kelvin); sim.context.setParameter(baro.Temperature(), float(Tk))
            sim.step(int(equil_ps * spp))
            rs, es = [], []
            for _ in range(int(sample_ps / 10)):
                sim.step(10 * spp); rs.append(rho(sim))
                st = sim.context.getState(getEnergy=True)
                es.append((st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
            R[k] = float(np.mean(rs)); U[k] = float(np.mean(es))
        print(f"   palier {k+1}/{len(temps)} T={Tk:.0f}K ρ={R[k]:.3f}", flush=True)
    # Cp CLASSIQUE = dU/dT (pente de U(T)), ramené par gramme. ⚠ la MD classique SURESTIME Cp
    # (équipartition : modes quantiques haute-fréquence non gelés) — pas de correction quantique ici.
    cp_slope = float(np.polyfit(temps, U, 1)[0])                 # kJ/mol(box)/K
    cp_classical = cp_slope * 1000.0 / 6.02214076e23 / mass_g    # J/(g·K) CLASSIQUE (surestimé ~2.3×)
    cp_jgk = cp_classical / 2.27   # correction classique→quantique (1er ordre, calibré 8 polym., résiduel ~15%)

    with P.block("09_tg_fit"):
        tg_hyp, det = tg_kinetics.fit_tg_hyperbola(temps, R)
        popt, pcov = det["params"], det.get("pcov")
        quality = tg_kinetics.assess_fit(temps, tg_hyp, det, t_step)
        # Tg ROBUSTE : hyperbole quand le fit est fiable (précis, sub-pas) ; sinon REPLI sur le
        # coude-2-segments qui ne latche jamais (l'hyperbole latche → artefacts +31). Réduit la MAE
        # de ~17→~10 K sur 16 familles (validé hors-GPU sur ρ(T) existants).
        if quality["reliable"]:
            tg_sim, tg_method, tg_std = tg_hyp, "hyperbole", det.get("tg_std", float("nan"))
        else:
            tg_sim, bpdet = tg_kinetics.fit_tg_breakpoint(temps, R)
            tg_method, tg_std = "coude-2seg", bpdet["step"] / 2.0
        tg_pred = tg_sim / 1.50                       # facteur universel
        # densité @300K RÉEL : extrapolation de la branche VITREUSE du fit (T réelle, pas rescalée).
        T_ROOM = 300.0
        dens300, dens300_std = tg_kinetics.rho_at_T(T_ROOM, popt, pcov)
        dens_extrap = T_ROOM < float(np.min(temps))
        # FFV (gratuit) : fraction de volume libre = 1 − 1.3·V_vdw/V_sp (Bondi). V_vdw = grille RDKit
        # sur la chaîne (bouts négligeables à n=40). Physique (>0) ; convention 1.3 → comparer en RELATIF.
        ffv = 1.0 - 1.3 * dens300 * v_vdw_chain * 0.60221 / m_chain if dens300 else None
        # indice de réfraction (gratuit) : Lorentz-Lorenz, φ = R_M·ρ/M (R_M = réfraction molaire Crippen)
        phi_ll = mr_chain * dens300 / m_chain if dens300 else None
        n_refr = (((1 + 2 * phi_ll) / (1 - phi_ll)) ** 0.5) if (phi_ll and phi_ll < 1) else None
        rec = tg_kinetics.recommend_window(temps, R, tg_hyp, t_step) if not quality["reliable"] else None
    if not quality["reliable"]:
        print(f"  ⚠️  hyperbole peu fiable → Tg via COUDE-2-SEGMENTS (repli robuste) : "
              + " ; ".join(quality["reasons"]), flush=True)
        if rec:
            print(f"  → si besoin d'affiner ({rec['direction']}) : {rec['message']}", flush=True)

    props = {"seed": seed,
             "Tg_method": tg_method,
             "Tg_sim": round(tg_sim, 1),
             "Tg_sim_ci": round(tg_std, 1) if np.isfinite(tg_std) else None,
             "Tg_sim_hyperbola": round(tg_hyp, 1),
             "Tg_pred": round(tg_pred, 1),
             "Tg_pred_ci": round(tg_std / 1.50, 1) if np.isfinite(tg_std) else None,
             "eff_rate": eff_rate,
             "density_300K": round(dens300, 4),
             "density_300K_ci": round(dens300_std, 4) if dens300_std else None,
             "density_300K_extrapolated": bool(dens_extrap),
             "FFV": round(ffv, 4) if ffv is not None else None,        # fraction de volume libre (gratuit)
             "Rg_nm": round(rg_nm, 2),                                 # rayon de giration (SOUS-ESTIMÉ : chaînes
             "Ree_nm": round(ree_nm, 2),                               # effondrées par le conformère compact)
             # C_inf RETIRÉ : non fiable (chaînes effondrées + taille finie 40-mère ; ×0.3 vs exp).
             "Cp_JgK": round(cp_jgk, 2),                               # Cp corrigé (classique ÷2.27)
             "Cp_classical_JgK": round(cp_classical, 2),               # Cp classique brut (surestimé)
             "refractive_index": round(n_refr, 3) if n_refr else None, # indice de réfraction (gratuit)
             # CTE : RETIRÉ de la sortie (non fiable, r≈−0.46 vs exp). C'est un sous-produit gratuit du
             # fit (det["cte_glass"]) — pas un calcul MD séparé — donc rien à économiser, juste pas affiché.
             "fit_reliable": quality["reliable"], "fit_warnings": quality["reasons"],
             "fit_direction": rec["direction"] if rec else None,
             "fit_recommendation": rec["message"] if rec else None}

    # ─────────────── MÉCANIQUE (sur le verre à MECH_T, défaut 300 K) ───────────────
    # On évalue K/E/ν à 300 K (T ambiante = standard des modules reportés), BIEN sous Tg : à t_low
    # (souvent ~Tg−50K) le verre est mou et s'écoule tôt (E sous-estimé, courbe σ(ε) qui sature).
    # On trempe donc d'abord de t_low → MECH_T.
    if do_mech:
        mech_T = float(os.environ.get("MECH_T", "300"))
        kT = 1.380649e-23 * mech_T                  # J
        # K via fluctuations de volume (NPT court au verre, à MECH_T).
        with P.block("10_K_volfluct"):
            ig.setTemperature(mech_T * unit.kelvin); sim.context.setParameter(baro.Temperature(), mech_T)
            sim.step(60 * spp)                       # trempe t_low→MECH_T + ré-équilibre (densifie)
            Vs = []
            for _ in range(40):                      # NB: K par fluctuations de volume = IMPRÉCIS/dispersé
                sim.step(2 * spp)                    # (sensible à la longueur de sampling) → flaggé ⚠ dans la CLI
                Vs.append(sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) * 1e-27)  # m³
            Vs = np.array(Vs); K_fluct = kT * Vs.mean() / Vs.var() / 1e9   # GPa
            # ± : incertitude statistique d'une variance estimée sur N échantillons ≈ √(2/(N−1)).
            K_ci = float(K_fluct) * (2.0 / (len(Vs) - 1)) ** 0.5
        props["K_GPa"] = round(float(K_fluct), 2)         # K = SAIN (fluctuations de volume)
        props["K_GPa_ci"] = round(K_ci, 2)
        props["compressibility_1_GPa"] = round(1.0 / float(K_fluct), 4)   # κ = 1/K (gratuit)

        # CED / paramètre de solubilité δ (conformation gelée). À 300 K (T de référence pour δ ;
        # le calculer au verre t_low — souvent ≫300K — sous-estimait δ de ~20-30%).
        with P.block("10b_ced"):
            ig.setTemperature(300.0 * unit.kelvin); sim.context.setParameter(baro.Temperature(), 300.0)
            sim.step(40 * spp)                       # 40 ps d'équilibration à 300 K
            stg = sim.context.getState(getPositions=True, getEnergy=True)
            pos_ced = stg.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            boxL_ced = np.diag(np.array(stg.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)))
            e_bulk = stg.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            V_m3 = stg.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) * 1e-27
            sys1 = ff.create_interchange(Topology.from_molecules([offmol]),
                                         charge_from_molecules=[offmol]).to_openmm()
            for fr in sys1.getForces():
                if isinstance(fr, NonbondedForce):
                    fr.setNonbondedMethod(NonbondedForce.NoCutoff)
            ig1 = LangevinMiddleIntegrator(t_low * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds)
            sim1 = Simulation(Topology.from_molecules([offmol]).to_openmm(), sys1, ig1, plat)
            ced_mpa, delta = compute_ced(pos_ced, boxL_ced, omm_top, n_chains, sim1, e_bulk, V_m3)
        props["CED_MPa"] = round(ced_mpa, 0)
        # δ = √CED ×1.25 : correction du biais de méthode (PME périodique vs NoCutoff en vide),
        # systématique ~−20% ; calibré sur 8 polymères → MAE 20%→10%. (cf Cp÷2.27, Tg÷1.50.)
        props["solubility_delta"] = round(delta * 1.25, 1)

        # E / ν / G : ESSAI DE TRACTION UNIAXIALE (stress-strain). Remplace l'energy-strain cassé.
        # On étire la boîte selon z par PALIERS de déformation ε_zz, en laissant x,y RELAXER à 1 bar
        # (barostat ANISOTROPE, z non scalé) → essai de traction réel (contrainte uniaxiale). À chaque
        # palier on mesure :
        #   σ_zz = ⟨∂U/∂ε_zz⟩ / V   — la CONTRAINTE (réponse en O(ε), 1ʳᵉ dérivée) via différence finie
        #         d'énergie sur un rescale affine ±δ en z. C'est tout l'intérêt vs l'ancienne méthode :
        #         on lit une 1ʳᵉ dérivée (bon SNR), pas la COURBURE de ⟨E⟩(γ) (2ᵉ dérivée, noyée dans kT).
        #         Le terme cinétique (~ρkT/axe) est ~constant en ε → s'annule dans la PENTE.
        #   ε_xx = ⟨Lx⟩/Lx(ε=0) − 1   — contraction latérale (gardée en simple DIAGNOSTIC).
        # E = pente(σ_zz vs ε_zz). Puis ν et G DÉRIVÉS de E (traction) + K (fluctuations de volume),
        # les deux grandeurs robustes : ν=(3K−E)/(6K), G=3KE/(9K−E). On évite le ν par dimensions de
        # boîte (trop bruité sur un verre rigide : sa variation par 1% axial n'est que ~ν% de Lx).
        if os.environ.get("MECH_TENSILE", "1") == "1":
            print("  → E/ν par traction uniaxiale (stress-strain, latéral relaxé)…", flush=True)
            with P.block("11_tensile"):
                from openmm import MonteCarloAnisotropicBarostat, Vec3
                for idx in reversed(range(omm_system.getNumForces())):   # retire le barostat isotrope
                    if isinstance(omm_system.getForce(idx), MonteCarloBarostat):
                        omm_system.removeForce(idx)
                # barostat anisotrope : relaxe X,Y à 1 bar, NE scale PAS Z (piloté par nous).
                abaro = MonteCarloAnisotropicBarostat(Vec3(1.0, 1.0, 1.0) * unit.bar,
                                                      mech_T * unit.kelvin, True, True, False, 25)
                omm_system.addForce(abaro)
                igT = LangevinMiddleIntegrator(mech_T * unit.kelvin, 1.0 / unit.picosecond, DT * unit.femtoseconds)
                simT = Simulation(omm_top, omm_system, igT, plat)
                st0 = sim.context.getState(getPositions=True)
                box0 = st0.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                pos0 = st0.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                Lz0 = float(box0[2, 2]); Lx0 = float(box0[0, 0])
                simT.context.setPeriodicBoxVectors(*(box0 * unit.nanometer))
                simT.context.setPositions(pos0 * unit.nanometer)
                simT.context.setVelocitiesToTemperature(mech_T * unit.kelvin, 7)
                dlt = 1e-4                                               # δ de la différence finie σ
                eq_ps = float(os.environ.get("TENSILE_EQUIL_PS", "40"))   # équilibration latérale à ε=0
                mode = os.environ.get("TENSILE_MODE", "palier")          # 'palier' (quasi-statique) | 'ramp'

                def _sig_lx():
                    # σ_zz (FD énergie ±δ, rescale affine z) + Lx à l'état COURANT, SANS avancer le temps.
                    s = simT.context.getState(getPositions=True)
                    bb = s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                    pp = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    V = float(np.linalg.det(bb)) * 1e-27
                    u = []
                    for dd in (+dlt, -dlt):
                        bd = bb.copy(); bd[2, 2] *= (1.0 + dd); pd = pp.copy(); pd[:, 2] *= (1.0 + dd)
                        simT.context.setPeriodicBoxVectors(*(bd * unit.nanometer)); simT.context.setPositions(pd * unit.nanometer)
                        u.append(simT.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    simT.context.setPeriodicBoxVectors(*(bb * unit.nanometer)); simT.context.setPositions(pp * unit.nanometer)
                    return (u[0] - u[1]) / (2.0 * dlt) * 1000.0 / 6.02214076e23 / V / 1e9, float(bb[0, 0])  # GPa, nm

                def _sample(nfr, step_ps=2.0):                           # ⟨σ_zz⟩,⟨Lx⟩ en avançant entre frames
                    szs, lxs = [], []
                    for _ in range(nfr):
                        simT.step(int(step_ps * spp)); sg, lx = _sig_lx(); szs.append(sg); lxs.append(lx)
                    return float(np.mean(szs)), float(np.mean(lxs))

                if mode == "ramp":
                    # RAMPE CONTINUE à vitesse contrôlée ε̇ (le barostat latéral reste actif → traction
                    # uniaxiale vraie). C'est le protocole de réf (polyimides) : E réaliste aux vitesses MD,
                    # et la PENTE E vs log(ε̇) donne la correction de vitesse (analogue mécanique du ÷1.50 Tg).
                    rate = float(os.environ.get("TENSILE_RATE", "1e-6"))     # ε̇ par fs
                    eps_max = float(os.environ.get("TENSILE_EPS_MAX", "0.04"))
                    cap = float(os.environ.get("TENSILE_ELASTIC_CAP", "0.02"))   # fit du module sous ce seuil
                    nsamp = int(os.environ.get("TENSILE_RAMP_SAMPLES", "40"))
                    simT.step(int(eq_ps * spp))                              # équilibre à ε=0
                    sig0, Lx_ref = _sample(6)
                    Lz_ref = float(simT.context.getState().getPeriodicBoxVectors(asNumpy=True)
                                   .value_in_unit(unit.nanometer)[2, 2])
                    deps = eps_max / nsamp                                   # incrément de strain par échantillon
                    chunk = max(1, int(round(deps / (rate * DT))))          # nb de pas MD/incrément → FIXE la vitesse
                    eps_l, sig_l, exx_l = [0.0], [0.0], [0.0]
                    for _k in range(nsamp):
                        st = simT.context.getState(getPositions=True)
                        b = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                        p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        b[2, 2] *= (1.0 + deps); p[:, 2] *= (1.0 + deps)
                        simT.context.setPeriodicBoxVectors(*(b * unit.nanometer)); simT.context.setPositions(p * unit.nanometer)
                        simT.step(chunk)                                    # intègre à la vitesse voulue
                        sg, lx = _sig_lx()
                        eps_l.append(float(b[2, 2]) / Lz_ref - 1.0); sig_l.append(sg - sig0); exx_l.append(lx / Lx_ref - 1.0)
                    ez_a = np.array(eps_l); sig = np.array(sig_l); exx = np.array(exx_l)
                    m = ez_a <= cap
                    E = float(np.polyfit(ez_a[m], sig[m], 1)[0])
                    nu_box = float(-np.polyfit(ez_a[m], exx[m], 1)[0])
                    print(f"     [ramp ε̇={rate:.0e}/fs, {int(chunk*DT)}fs/incr] {int(m.sum())} pts ≤{cap:.0%} → E={E:.2f} GPa", flush=True)
                    if eps_max >= 0.1:    # C1 : grande déformation → SEUIL D'ÉCOULEMENT (pic de σ)
                        # NB : en bulk PÉRIODIQUE (pas de surface/défaut), le matériau N'A PAS de rupture
                        # fragile — il atteint un pic (limite élastique/écoulement) puis flue/cavite. On
                        # rapporte donc σ_y, ε_y (écoulement DUCTILE), PAS un strain-at-break fragile.
                        iy = int(np.argmax(sig)); ductile = bool(iy < len(sig) - 2)
                        props["yield_stress_GPa"] = round(float(sig[iy]), 3)
                        props["yield_strain"] = round(float(ez_a[iy]), 3)
                        props["mech_note"] = ("seuil d'écoulement (bulk périodique → ductile, "
                                              "PAS rupture fragile)")
                        print(f"     [C1] seuil : σ_y={sig[iy]:.3f} GPa à ε_y={ez_a[iy]:.1%}"
                              f" {'(pic franc)' if ductile else '(pas de pic net dans la plage)'}", flush=True)
                else:
                    # PALIERS quasi-statiques : on relaxe eq_ps à chaque ε puis on échantillonne (limite
                    # de vitesse → 0 ; donne le module RELAXÉ, borne basse). Plage ≤3% pour rester élastique.
                    strains = [float(x) for x in os.environ.get("TENSILE_STRAINS",
                                                                "0.0,0.005,0.01,0.015,0.02,0.025,0.03").split(",")]
                    nfr = int(os.environ.get("TENSILE_FRAMES", "30"))
                    sig_raw, lx_raw = [], []
                    for ez in strains:
                        st = simT.context.getState(getPositions=True)
                        b = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                        p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        sc = Lz0 * (1.0 + ez) / b[2, 2]; b[2, 2] = Lz0 * (1.0 + ez); p[:, 2] *= sc
                        simT.context.setPeriodicBoxVectors(*(b * unit.nanometer)); simT.context.setPositions(p * unit.nanometer)
                        simT.step(int(eq_ps * spp))
                        sg, lx = _sample(nfr); sig_raw.append(sg); lx_raw.append(lx)
                    Lx_ref = lx_raw[0]; ez_a = np.array(strains)
                    sig = np.array([s - sig_raw[0] for s in sig_raw]); exx = np.array([lx / Lx_ref - 1.0 for lx in lx_raw])
                    E = float(np.polyfit(ez_a, sig, 1)[0]); nu_box = float(-np.polyfit(ez_a, exx, 1)[0])
                    print(f"     [diag] σ_zz(GPa)={[round(float(x),3) for x in sig]}", flush=True)
                    print(f"     [diag] ε_xx={[round(float(x),5) for x in exx]}", flush=True)
                # ν, G DÉRIVÉS de E (traction) + K (fluctuations de volume) — les DEUX grandeurs robustes.
                # Évite le ν par dimensions de boîte, intrinsèquement bruité pour un verre rigide.
                #   ν = (3K−E)/(6K) ; G = 3KE/(9K−E).   (E=3K(1−2ν) ; E=2G(1+ν))
                K = float(K_fluct)
                nu = (3.0 * K - E) / (6.0 * K) if K else None
                G = 3.0 * K * E / (9.0 * K - E) if (9.0 * K - E) else None
            props.update({"E_GPa": round(E, 2), "poisson": round(nu, 3) if nu is not None else None,
                          "G_GPa": round(G, 2) if G else None,
                          "poisson_boxdiag": round(nu_box, 3)})
            print(f"     E={E:.2f} GPa (traction) | K={K:.2f} GPa (vol.fluct) → ν={nu:.3f} G={G:.2f} GPa"
                  f" | ν_boxdiag={nu_box:.3f}",
                  flush=True)

    print("\n=== PROPRIÉTÉS ===")
    print(json.dumps(props, indent=1))
    P.report()


if __name__ == "__main__":
    main()
