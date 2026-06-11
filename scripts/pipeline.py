"""Pipeline UNIFIÉ MD-FF : un SMILES de monomère → propriétés thermiques + mécaniques.
Un seul GPU, ~15-18 min (boîte 48k atomes — voir le plancher SNR ci-dessous).

SORTIES :
  Tg_pred   : Tg expérimentale prédite = Tg_sim / 1.50  (correction CINÉTIQUE universelle,
              MAE ~26 K sur 14 polymères, sans dépendance chimique ; cf project-md-pipeline).
  density_300K : densité (FIABLE). CTE_*_experimental : dilatation (gratuit depuis ρ(T), ⚠ non validée).
  K_GPa     : module de compression (fluctuations de volume — FIABLE).
  STANDARD (toujours) : Tg, densité, Cp, K, Rg/Ree, FFV, CED/δ, indice, CTE(flaggée).
  OPT-IN (coûteux, +dizaines de min, cases à cocher webapp) :
    E/ν/G  (MECH_TENSILE=1) : traction uniaxiale stress-strain ; ν,G dérivés de E+K.
    diélectrique statique + auto-diffusion (DIELECTRIC=1) : sampling long 300 K (~400 ps).

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


def xtb_electronic(unit_smiles):
    """Propriétés ÉLECTRONIQUES du motif (HOMO/LUMO/gap, dipôle, polarisabilité) par xtb GFN2-xTB
    (semi-empirique, ~secondes) sur le monomère cappé H. Info NOUVELLE (non dérivable de la MD), utile
    optique/électronique. Renvoie {} si xtb indisponible/échoue (gracieux). XTB_BIN surchargeable."""
    import subprocess, tempfile, shutil, re
    xtb = os.path.expanduser(os.environ.get("XTB_BIN", "~/conda_xtb/bin/xtb"))
    if not os.path.exists(xtb):
        return {}
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(str(unit_smiles).replace("*", "[H]"))
    if m is None:
        return {}
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
        return {}
    AllChem.MMFFOptimizeMolecule(m, maxIters=300)
    d = tempfile.mkdtemp(); xyz = os.path.join(d, "m.xyz")
    Chem.MolToXYZFile(m, xyz)
    try:
        out = subprocess.run([xtb, xyz, "--gfn", "2"], cwd=d, capture_output=True,
                             text=True, timeout=180).stdout
    except Exception:
        return {}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res = {}
    pats = {"homo_lumo_gap_eV": (r"HOMO-LUMO gap\s+([-\d.]+)\s+eV", 3),
            "HOMO_eV": (r"([-\d.]+)\s+\(HOMO\)", 3),
            "LUMO_eV": (r"([-\d.]+)\s+\(LUMO\)", 3),
            "dipole_Debye": (r"molecular dipole:.*?full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)", 3)}
    for key, (pat, nd) in pats.items():
        mm = re.search(pat, out, re.DOTALL)
        if mm:
            res[key] = round(float(mm.group(1)), nd)
    a = re.search(r"\(0\)\s*/au\s*:\s*([\d.]+)", out)               # Mol. α(0) /au (α unicode)
    if a:
        res["polarizability_A3"] = round(float(a.group(1)) * 0.1481847, 2)   # ua → Å³
    # IP/EA verticaux (2ᵉ appel --vipea : calcs cation/anion) — potentiel d'ionisation, affinité électr.
    d2 = tempfile.mkdtemp()
    try:
        Chem.MolToXYZFile(m, os.path.join(d2, "m.xyz"))
        out2 = subprocess.run([xtb, "m.xyz", "--gfn", "2", "--vipea"], cwd=d2,
                              capture_output=True, text=True, timeout=180).stdout
        ip = re.search(r"delta SCC IP \(eV\):\s+([-\d.]+)", out2)
        ea = re.search(r"delta SCC EA \(eV\):\s+([-\d.]+)", out2)
        if ip:
            res["ionization_potential_eV"] = round(float(ip.group(1)), 3)
        if ea:
            res["electron_affinity_eV"] = round(float(ea.group(1)), 3)
    except Exception:
        pass
    finally:
        shutil.rmtree(d2, ignore_errors=True)
    return res


def thermal_conductivity_nemd(omm_top, omm_system, plat, pos_nm, box_diag_nm, target_T=300.0):
    """Conductivité thermique κ (W/m/K) par reverse-NEMD à FLUX IMPOSÉ (algo HEX, Ikeshoji-Hafskjold).
    OpenMM n'a pas de `fix thermal/conductivity` natif (≠ LAMMPS/RadonPy, Müller-Plathe). On impose
    plutôt un flux de chaleur CONNU en ré-échelonnant périodiquement les vitesses AUTOUR du COM de
    chaque slab : +ΔE au slab CHAUD (milieu de la boîte), −ΔE au slab FROID (bord). Ce ré-échelonnage
    (i) conserve la quantité de mouvement du slab, (ii) injecte un ΔE EXACT, (iii) marche avec des masses
    mixtes (≠ échange de vitesses, qui suppose masses égales). En régime établi le profil T(z) est
    linéaire dans chaque demi-boîte ; κ = J / ∇T avec J = ΔE/(intervalle · 2A) (le 2 = 2 chemins par PBC).
    Dynamique NVE (Verlet) : aucun thermostat global ne doit lutter contre le gradient.
    OPT-IN (THERMAL=1) — coûteux (~1 ns) et SOUS-ESTIMÉ par effet de taille finie (z≈6 nm → libre
    parcours des phonons tronqué ; RadonPy allonge la boîte). À prendre comme borne basse / ordre de
    grandeur. Renvoie (kappa, info) ou (None, {}) si le gradient ne s'établit pas."""
    import openmm
    from openmm import unit, VerletIntegrator, LangevinMiddleIntegrator, MonteCarloBarostat
    from openmm.app import Simulation
    kB = 1.380649e-23; NA = 6.02214076e23
    C = 500.0 / NA                       # KE[J] = C · Σ m[g/mol] · v²[nm/ps]   (½·1e3/NA)
    m = np.array([omm_system.getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(omm_system.getNumParticles())])
    dt = 2.0                              # fs (HMR dans le système → 2 fs sûr en NVE)
    box = np.array([float(box_diag_nm[0]), float(box_diag_nm[1]), float(box_diag_nm[2])])

    def _purge_baro():
        for i in reversed(range(omm_system.getNumForces())):
            if isinstance(omm_system.getForce(i), MonteCarloBarostat):
                omm_system.removeForce(i)
    # 1) ré-équilibration NPT à target_T → fixe la BONNE densité (ρ@300K) AVANT de geler le volume.
    #    (sans ça, on hérite du volume du dernier palier de refroidissement → densité fausse → T moyenne off.)
    eq_ps = float(os.environ.get("THERMAL_EQ_PS", "40"))
    _purge_baro(); omm_system.addForce(MonteCarloBarostat(1.0 * unit.bar, target_T * unit.kelvin, 25))
    ige = LangevinMiddleIntegrator(target_T * unit.kelvin, 1.0 / unit.picosecond, dt * unit.femtoseconds)
    ige.setRandomNumberSeed(777)
    se = Simulation(omm_top, omm_system, ige, plat)
    se.context.setPeriodicBoxVectors(*(np.eye(3) * box)); se.context.setPositions(pos_nm * unit.nanometer)
    se.context.setVelocitiesToTemperature(target_T * unit.kelvin, 777)
    se.step(int(eq_ps * 1000 / dt))
    stt = se.context.getState(getPositions=True, getVelocities=True)
    pos = stt.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    vel = stt.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    box = np.diag(stt.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))   # densité corrigée
    del se
    _purge_baro()                        # NEMD = volume fixe → on gèle la boîte ré-équilibrée
    # 2) production NVE + échanges HEX
    sp = Simulation(omm_top, omm_system, VerletIntegrator(dt * unit.femtoseconds), plat)
    sp.context.setPeriodicBoxVectors(*(np.eye(3) * box)); sp.context.setPositions(pos * unit.nanometer)
    sp.context.setVelocities(vel * unit.nanometer / unit.picosecond)
    nslab = int(os.environ.get("THERMAL_NSLAB", "20")); nslab -= nslab % 2
    hot, cold = nslab // 2, 0
    W = int(os.environ.get("THERMAL_W", "250"))               # pas entre échanges
    dE = float(os.environ.get("THERMAL_DE", "30.0")) * 1e3 / NA   # J/échange (def 30 kJ/mol)
    n_ns = float(os.environ.get("THERMAL_NS", "1.2"))
    n_exch = max(40, int(n_ns * 1e6 / dt / W))
    Lz = box[2]; slabw = Lz / nslab; A = box[0] * box[1] * 1e-18   # m²
    # dof EFFECTIFS par atome : les liaisons X–H sont CONTRAINTES (rigides, cf HMR) → chaque contrainte
    # retire 1 dof. Sans cette correction T est sous-estimée d'un facteur g=(3N−N_c)/3N (≈0.83 pour 50% H)
    # → ∇T trop petit → κ SURESTIMÉ de 1/g. On répartit g uniformément sur les atomes.
    nat = len(m); g_dof = (3.0 * nat - omm_system.getNumConstraints()) / (3.0 * nat)
    Tsum = np.zeros(nslab); Tcnt = np.zeros(nslab); warm = n_exch // 2; n_ok = 0
    Tmean_hist = []

    def slab_kepec(idx):
        mi = m[idx]; vi = vel[idx]; vcom = (mi[:, None] * vi).sum(0) / mi.sum()
        dv = vi - vcom; return vcom, dv, C * (mi[:, None] * dv * dv).sum()

    def slab_T(idx, kep):
        return 2.0 * kep / (max(3.0 * len(idx) * g_dof - 3.0, 1.0) * kB)

    def global_T(v):
        vcomG = (m[:, None] * v).sum(0) / m.sum(); dv = v - vcomG
        return 2.0 * C * (m[:, None] * dv * dv).sum() / (max(3.0 * nat * g_dof - 3.0, 1.0) * kB), vcomG

    pin = float(os.environ.get("THERMAL_PIN", "1.0"))    # force du verrou de T moyenne (1=plein/échange)
    for e in range(n_exch):
        sp.step(W)
        st = sp.context.getState(getPositions=True, getVelocities=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        z = pos[:, 2] - Lz * np.floor(pos[:, 2] / Lz)
        sid = np.minimum((z / slabw).astype(int), nslab - 1)
        ok = True
        for slab, sign in ((hot, +1.0), (cold, -1.0)):
            idx = np.where(sid == slab)[0]
            if len(idx) < 10: ok = False; continue
            vcom, dv, kep = slab_kepec(idx)
            if sign < 0 and kep <= dE * 1.5: ok = False; continue
            vel[idx] = vcom + (1.0 + sign * dE / kep) ** 0.5 * dv
        # ANTI-DÉRIVE : la NVE fuit (projection des contraintes sur les atomes ré-échelonnés + relaxation
        # résiduelle du verre) → T moyenne dérive (ex. PMMA 302→208 K). On l'épingle à target_T par un
        # rescale UNIFORME autour du COM GLOBAL : multiplicatif identique sur tous les atomes → la FORME
        # du profil T(z) (donc ∇T et κ) est INCHANGÉE, seul le niveau moyen est corrigé. No-op si pas de fuite.
        Tg, vcomG = global_T(vel)
        if Tg > 0:
            lamG = 1.0 + pin * ((target_T / Tg) ** 0.5 - 1.0)
            vel = vcomG + lamG * (vel - vcomG)
        sp.context.setVelocities(vel * unit.nanometer / unit.picosecond)
        if ok: n_ok += 1
        if e >= warm:                                          # accumulation régime établi
            for s in range(nslab):
                idx = np.where(sid == s)[0]
                if len(idx) < 10: continue
                Tsum[s] += slab_T(idx, slab_kepec(idx)[2]); Tcnt[s] += 1
        if e % max(1, n_exch // 10) == 0:
            Tmean_hist.append(round(float(np.mean([slab_T(ix, slab_kepec(ix)[2])
                              for s in range(nslab) for ix in [np.where(sid == s)[0]]
                              if len(ix) >= 10])), 1))
    Tprof = np.where(Tcnt > 0, Tsum / np.maximum(Tcnt, 1), np.nan)
    J = (n_ok * dE) / (n_exch * W * dt * 1e-15) / (2 * A)      # W/m² (flux moyen réel)
    zc = (np.arange(nslab) + 0.5) * slabw * 1e-9              # centres de slab (m)

    def slope(a, b):
        ss = np.arange(a, b); good = ~np.isnan(Tprof[ss])
        return float(np.polyfit(zc[ss][good], Tprof[ss][good], 1)[0]) if good.sum() >= 3 else None
    grads = [abs(s) for s in (slope(cold + 1, hot), slope(hot + 1, nslab)) if s is not None]
    if not grads:
        return None, {"Tprofile": [round(float(x), 1) for x in Tprof], "Tmean_hist": Tmean_hist}
    gradT = float(np.mean(grads)); kappa = J / gradT
    return kappa, {"dT_K": round(float(np.nanmax(Tprof) - np.nanmin(Tprof)), 1),
                   "flux_Wm2": float(f"{J:.3e}"), "gradT_Kpm": float(f"{gradT:.3e}"),
                   "box_z_nm": round(float(Lz), 2), "n_exch": n_exch, "frac_ok": round(n_ok / n_exch, 3),
                   "Tmean_hist": Tmean_hist, "Tprofile": [round(float(x), 1) for x in Tprof]}


def cp_dos_factor(omm_top, omm_system, plat, pos_nm, box_diag_nm, orig_mass, target_T=300.0):
    """Facteur de correction QUANTIQUE de Cp (PHYSIQUE, remplace le ÷2.27 calibré).
    La MD classique compte k_B par mode (équipartition) ; or les modes de vibration HF sont GELÉS
    quantiquement à 300 K (ℏω≫k_BT) → Cp surestimé. On échantillonne le DOS vibrationnel g(ν) via le
    spectre de puissance des vitesses (Wiener-Khinchin : g(ν) ∝ Σ_i m_i |v̂_i(ν)|²), puis on pondère
    chaque mode par la capacité d'Einstein c_E(ν)=x²eˣ/(eˣ−1)², x=hν/k_BT. Facteur = ⟨c_E⟩_g = ∫g·c_E/∫g.
    Cp_quantique = Cp_classique × facteur, PAR POLYMÈRE (vs ÷2.27 universel). CRUCIAL : masses VRAIES
    (non-HMR) sinon ω décalées (~×√2). Les C–H contraintes sont absentes du DOS = cohérent (gelées, c_E≈0)."""
    import openmm
    from openmm import unit, VerletIntegrator, LangevinMiddleIntegrator
    from openmm.app import Simulation
    h = 6.62607015e-34; kB = 1.380649e-23
    N = omm_system.getNumParticles()
    hmr = [omm_system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(N)]
    for i in range(N):
        omm_system.setParticleMass(i, orig_mass[i] * unit.dalton)
    box = np.array([float(box_diag_nm[0]), float(box_diag_nm[1]), float(box_diag_nm[2])])
    dt = 1.0                                          # fs — masses vraies (H léger) → petit pas
    rec_fs = float(os.environ.get("CP_DOS_REC_FS", "4")); nfr = int(os.environ.get("CP_DOS_FRAMES", "2048"))
    rec_steps = max(1, int(round(rec_fs / dt)))
    try:
        # Langevin FAIBLE (γ=1/ps) tout du long : ne broie que les modes < ~5 cm⁻¹ (où c_E≈1 de toute
        # façon) → le facteur est inchangé, et on évite l'explosion NVE (config verre froid + dt=1fs).
        se = Simulation(omm_top, omm_system,
                        LangevinMiddleIntegrator(target_T * unit.kelvin, 1.0 / unit.picosecond, dt * unit.femtoseconds),
                        plat)
        se.context.setPeriodicBoxVectors(*(np.eye(3) * box)); se.context.setPositions(pos_nm * unit.nanometer)
        se.minimizeEnergy(maxIterations=500)          # enlève les mauvais contacts (sinon NaN)
        se.context.setVelocitiesToTemperature(target_T * unit.kelvin, 99)
        se.step(int(5000 / dt))                       # 5 ps d'équilibration
        Vv = np.empty((nfr, N, 3), dtype=np.float32)
        for f in range(nfr):
            se.step(rec_steps)
            Vv[f] = se.context.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond)
    finally:
        for i in range(N):                            # RESTAURE les masses HMR (blocs suivants en dépendent)
            omm_system.setParticleMass(i, hmr[i] * unit.dalton)
    Vv -= Vv.mean(axis=0, keepdims=True)              # retire la dérive (composante DC)
    m = np.array(orig_mass, dtype=np.float64)
    dos = np.zeros(nfr // 2 + 1)
    for c0 in range(0, N, 4000):                      # FFT par blocs d'atomes (borne la mémoire)
        c1 = min(c0 + 4000, N)
        Pw = (np.abs(np.fft.rfft(Vv[:, c0:c1, :], axis=0)) ** 2).sum(axis=2)   # [freq, atomes]
        dos += (Pw * m[None, c0:c1]).sum(axis=1)
    freq = np.fft.rfftfreq(nfr, d=rec_fs * 1e-15)     # Hz
    x = h * freq / (kB * target_T)
    with np.errstate(over="ignore", invalid="ignore"):
        cE = np.where(x > 1e-6, x ** 2 * np.exp(-x) / (-np.expm1(-x)) ** 2, 1.0)   # forme stable (e^-x)
    cE = np.nan_to_num(cE, nan=0.0)
    w = dos.copy(); w[0] = 0.0                         # exclut DC
    factor = float((w * cE).sum() / w.sum())
    return factor, {"cp_dos_factor": round(factor, 3), "cp_dos_equiv_divisor": round(1.0 / factor, 2),
                    "dos_mean_cm1": round(float((w * (freq / 2.99792458e10)).sum() / w.sum()), 0),
                    "nyquist_cm1": round(float(freq[-1] / 2.99792458e10), 0)}


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
    # masses VRAIES (avant HMR) — nécessaires au calcul du DOS pour Cp quantique : HMR déplace la masse
    # H→lourd, ce qui DÉCALE les fréquences de vibration (atomes lourds allégés → ω ~√2 plus haut) →
    # un DOS issu d'une dynamique HMR donnerait de mauvaises fréquences. On les conserve telles quelles.
    orig_mass = [omm_system.getParticleMass(i).value_in_unit(unit.dalton)
                 for i in range(omm_system.getNumParticles())]
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
    # Cp QUANTIQUE : par DÉFAUT correction PHYSIQUE (DOS vibrationnel × Einstein, par polymère) ;
    # repli sur le ÷2.27 calibré si le DOS échoue. Validé verres PS/PMMA/PC : MAE 12.3%→3.2% vs ÷2.27.
    # (≫Tg : Cp exp inclut l'excès configurationnel liquide que la branche vitreuse ne capte pas — limite
    # commune aux 2 méthodes, ex. PEO@298K>Tg213K.) CP_DOS=0 force le ÷2.27.
    cp_div227 = cp_classical / 2.27
    cp_jgk = cp_div227
    cp_dos_info = {}
    if os.environ.get("CP_DOS", "1") == "1":
        try:
            st_d = sim.context.getState(getPositions=True)
            pd = st_d.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            bd = np.diag(st_d.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            with P.block("09b_cp_dos"):
                fac, cp_dos_info = cp_dos_factor(omm_top, omm_system, plat, pd, bd, orig_mass,
                                                 target_T=float(os.environ.get("MECH_T", "300")))
            cp_jgk = cp_classical * fac                      # DOS = défaut (physique)
            cp_dos_info["Cp_JgK_div227"] = round(cp_div227, 3)
            print(f"   [Cp-DOS] facteur={fac:.3f} (÷{1.0/fac:.2f}) | Cp_DOS={cp_jgk:.3f} (défaut) vs "
                  f"÷2.27={cp_div227:.3f} J/g/K | DOS moy={cp_dos_info.get('dos_mean_cm1')}cm⁻¹", flush=True)
        except Exception as e:
            print(f"   [Cp-DOS] échec ({type(e).__name__}: {e}) → repli ÷2.27", flush=True)
            cp_dos_info = {}

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
        dens300_md, dens300_std = tg_kinetics.rho_at_T(T_ROOM, popt, pcov)   # densité MD brute (FF)
        # CALIBRATION FF : OpenFF Sage est SOUS-DENSE (sous-cohésion ; biais −7.5% systématique sur 12
        # polymères, physique épuisée — polarizable réfuté). Facteur universel → meilleure estimation de
        # ρ_exp, qui CASCADE sur n (Lorentz-Lorenz ∝ ρ : le −3% de n n'est QUE le −7% de densité propagé)
        # et FFV. ⚠ biais FF chimie-dépendant (−2 à −12%) → résiduel ~3% après correction (densité 7.5→2.7%,
        # n 2.8→1.3% sur le benchmark). Brut conservé (density_300K_md). DENS_FF_CORR=1.0 désactive.
        dens_corr = float(os.environ.get("DENS_FF_CORR", "1.078"))
        dens300 = dens300_md * dens_corr if dens300_md else dens300_md
        dens300_std = (dens300_std * dens_corr) if dens300_std else dens300_std
        dens_extrap = T_ROOM < float(np.min(temps))
        # FFV (gratuit) : fraction de volume libre = 1 − 1.3·V_vdw/V_sp (Bondi). V_vdw = grille RDKit
        # sur la chaîne (bouts négligeables à n=40). Physique (>0) ; convention 1.3 → comparer en RELATIF.
        ffv = 1.0 - 1.3 * dens300 * v_vdw_chain * 0.60221 / m_chain if dens300 else None
        # indice de réfraction (gratuit) : Lorentz-Lorenz, φ = R_M·ρ/M (R_M = réfraction molaire Crippen)
        phi_ll = mr_chain * dens300 / m_chain if dens300 else None
        n_refr = (((1 + 2 * phi_ll) / (1 - phi_ll)) ** 0.5) if (phi_ll and phi_ll < 1) else None
        # CTE (déjà calculée par le fit : asymptotes vitreuse/fondue de ρ(T), det["cte_glass/rubber"]).
        # ⚠ NON VALIDÉE (r≈−0.46 vs exp ; RadonPy aussi la sort bruitée → PMMA négatif). On l'EXPOSE
        # quand même, flaggée _experimental, pour la parité de largeur — c'est un sous-produit gratuit,
        # aucun MD en plus. Convention : coefficient VOLUMIQUE en ppm/K (det l'a déjà).
        cte_glass = round(det["cte_glass"], 1) if det.get("cte_glass") is not None else None
        cte_melt = round(det["cte_rubber"], 1) if det.get("cte_rubber") is not None else None
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
             "density_300K": round(dens300, 4),                          # estimation ρ_exp (FF corrigé ×1.078)
             "density_300K_md": round(dens300_md, 4) if dens300_md else None,  # densité MD brute (FF, sous-dense)
             "density_300K_ci": round(dens300_std, 4) if dens300_std else None,
             "density_300K_extrapolated": bool(dens_extrap),
             "FFV": round(ffv, 4) if ffv is not None else None,        # fraction de volume libre (gratuit)
             "Rg_nm": round(rg_nm, 2),                                 # rayon de giration (SOUS-ESTIMÉ : chaînes
             "Ree_nm": round(ree_nm, 2),                               # effondrées par le conformère compact)
             # C_inf RETIRÉ : non fiable (chaînes effondrées + taille finie 40-mère ; ×0.3 vs exp).
             "Cp_JgK": round(cp_jgk, 2),                               # Cp corrigé (classique ÷2.27)
             "Cp_classical_JgK": round(cp_classical, 2),               # Cp classique brut (surestimé)
             **cp_dos_info,                                            # Cp_JgK_dos + facteur (si CP_DOS=1)
             "refractive_index": round(n_refr, 3) if n_refr else None, # indice de réfraction (gratuit)
             "CTE_glass_ppmK_experimental": cte_glass,    # dilatation volumique vitreuse — ⚠ NON VALIDÉE
             "CTE_melt_ppmK_experimental": cte_melt,      #   (r≈−0.46 vs exp) ; exposée pour parité largeur
             "fit_reliable": quality["reliable"], "fit_warnings": quality["reasons"],
             "fit_direction": rec["direction"] if rec else None,
             "fit_recommendation": rec["message"] if rec else None}

    # ─────────────── MÉCANIQUE (sur le verre à MECH_T, défaut 300 K) ───────────────
    # On évalue K/E/ν à 300 K (T ambiante = standard des modules reportés), BIEN sous Tg : à t_low
    # (souvent ~Tg−50K) le verre est mou et s'écoule tôt (E sous-estimé, courbe σ(ε) qui sature).
    # On trempe donc d'abord de t_low → MECH_T.
    if do_mech:
        mech_T = float(os.environ.get("MECH_T", "300"))
        # DIAGNOSTIC (opt-in, défaut OFF) modules vs densité. Le FF est SOUS-DENSE → verre mou → modules
        # sous-estimés. On peut atteindre une densité cible par recherche de pression adaptative.
        # ⚠ NE PAS utiliser comme FIX par défaut : comprimer PAR PRESSION introduit un RAIDISSEMENT-PRESSION
        # qui s'AJOUTE au raidissement-densité → SURESTIME E quand l'écart de densité est grand (PS gap 5%,
        # 3669bar : E 1.3→3.39≈exp ✓ ; PMMA gap 13%, 8543bar : E→4.5 ≫ exp 2.9 ✗ ; PC E→3.9 ≫ 2.3 ✗).
        # Cause racine = SOUS-COHÉSION du FF (chimie-dépendante, pire pour polaires) = limite FF, pas
        # d'échantillonnage. MECH_RHO_TARGET = densité absolue g/cm³ (validation) ; MECH_RHO_FAC × densité
        # prédite ; MECH_P_BAR force une pression brute. Défaut 1.0/1.0 = aucune correction.
        mech_p = float(os.environ.get("MECH_P_BAR", "1.0"))
        rho_fac = float(os.environ.get("MECH_RHO_FAC", "1.0"))
        rho_target_env = os.environ.get("MECH_RHO_TARGET", "")
        kT = 1.380649e-23 * mech_T                  # J

        def _rho_now():
            V = sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) * 1e-21
            return mass_g / V                        # g/cm³

        # K via fluctuations de volume (NPT court au verre, à MECH_T).
        with P.block("10_K_volfluct"):
            ig.setTemperature(mech_T * unit.kelvin); sim.context.setParameter(baro.Temperature(), mech_T)
            sim.context.setParameter(baro.Pressure(), mech_p * unit.bar)
            sim.step(60 * spp)                       # trempe t_low→MECH_T + ré-équilibre (densifie)
            rho_tgt = float(rho_target_env) if rho_target_env else (_rho_now() * rho_fac if rho_fac != 1.0 else None)
            if rho_tgt:                              # RECHERCHE de pression pour atteindre la densité cible
                Pp, rp = mech_p, _rho_now(); Keff = 4.0    # K initial grossier (GPa) du verre comprimé
                for _ in range(5):
                    rho_c = _rho_now()
                    if abs(rho_c / rho_tgt - 1.0) < 0.008:
                        break
                    Pn = max(1.0, Pp + Keff * 1e4 * (rho_tgt / rho_c - 1.0))   # Newton (ΔP=K·1e4·Δρ/ρ ; 1GPa=1e4bar)
                    sim.context.setParameter(baro.Pressure(), Pn * unit.bar); sim.step(40 * spp)
                    rn = _rho_now()
                    if abs(rn - rp) > 1e-4 and abs(Pn - Pp) > 1.0:            # K_eff local mesuré → adaptatif
                        Keff = min(max(((Pn - Pp) / 1e4) / ((rn - rp) / rn), 1.0), 20.0)
                    Pp, rp = Pn, rn
                mech_p = Pp
                print(f"   [modules@ρ] cible ρ={rho_tgt:.3f} → P={mech_p:.0f}bar ρ_obt={_rho_now():.3f} g/cm³", flush=True)
            if mech_p > 1.0:                          # densité atteinte sous la pression imposée
                props["density_mechP"] = round(_rho_now(), 4); props["mech_P_bar"] = round(mech_p, 0)
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

        # CTE VITREUX DÉDIÉ (room-T) : NPT à 280/300/320 K → α_V = (1/V₃₀₀)·dV/dT. PLUS FIABLE que la pente
        # de la branche vitreuse du fit de refroidissement : ses paliers sont près de Tg_sim (verre "frais",
        # encore mou → CTE surestimée ~2×, PS ~460 vs exp ~210 ppm/K). Ici on mesure au VRAI verre froid.
        cte_roomT = None
        with P.block("10e_cte_roomT"):
            sim.context.setParameter(baro.Pressure(), 1.0 * unit.bar)   # CTE au repos (1 bar), pas comprimé
            # span LARGE (270-330K, signal ΔV/V ~ +50%) car dV/dT est un petit signal et le V du verre est
            # autocorrélé (N effectif faible) ; équilibration plus longue (le volume du verre relaxe lentement).
            Tcte = [270.0, 290.0, 310.0, 330.0]; Vcte = []
            for Tc in Tcte:
                ig.setTemperature(Tc * unit.kelvin); sim.context.setParameter(baro.Temperature(), Tc)
                sim.step(50 * spp)                       # ré-équilibre (verre lent)
                vv = [sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
                      for _ in range(30) if not sim.step(2 * spp)]
                Vcte.append(float(np.mean(vv)))
            V300 = float(np.interp(300.0, Tcte, Vcte))
            cte_roomT = float(np.polyfit(Tcte, Vcte, 1)[0]) / V300 * 1e6   # ppm/K volumétrique
        props["CTE_glass_ppmK"] = round(cte_roomT, 1)
        print(f"   [CTE] vitreux dédié 270-330K = {cte_roomT:.0f} ppm/K (vs fit-refroidiss. {cte_glass})", flush=True)
        cte_for_thermo = cte_roomT if cte_roomT else cte_glass    # le dédié alimente les dérivés thermo

        # Dérivés thermo (gratuits) : Cv + grandeurs ISENTROPIQUES via Cp−Cv = T·α_V²/(ρ·κ_T) puis γ=Cp/Cv.
        # ⚠ dépend de la CTE (α_V), peu fiable → flaggés _experimental. (RadonPy les sort aussi du même eq.)
        # NB: K et CTE sont mesurés à la densité MD (FF) → on utilise dens300_md (brute) ici pour la
        # COHÉRENCE thermodynamique (Cp−Cv et v_son liés à K à la même densité), pas la densité corrigée.
        dens_thermo = dens300_md if dens300_md else dens300
        if cte_for_thermo is not None and dens_thermo and K_fluct:
            aV = cte_for_thermo * 1e-6                                         # CTE volumique (1/K, dédié)
            dCpv = mech_T * aV ** 2 / (dens_thermo * 1000.0 * (1.0 / (float(K_fluct) * 1e9))) / 1000.0  # Cp−Cv
            cv = cp_jgk - dCpv
            props["Cv_JgK_experimental"] = round(cv, 2)
            if cv:
                gamma = cp_jgk / cv
                Ks = float(K_fluct) * gamma                                   # module isentropique (GPa)
                props["isentropic_compressibility_1_GPa_experimental"] = round((1.0 / float(K_fluct)) / gamma, 4)
                props["isentropic_K_GPa_experimental"] = round(Ks, 2)
                # vitesse du son (bulk) v = √(K_S/ρ) — ρ MD brute (cohérent avec K mesuré à cette densité)
                props["sound_velocity_ms"] = round((Ks * 1e9 / (dens_thermo * 1000.0)) ** 0.5, 0)

        # Ordre nématique (gratuit) : S = max valeur propre de Q = ⟨(3 u⊗u − I)/2⟩ sur les vecteurs liaison
        # u. 0 = isotrope (amorphe attendu), →1 = aligné. Vérif qualité (pas d'orientation parasite) +
        # pertinence cristal-liquide. RadonPy le sort aussi.
        with P.block("10d_nematic"):
            st_n = sim.context.getState(getPositions=True)
            pn = st_n.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            boxn = np.diag(st_n.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            uu = []
            for b in omm_top.bonds():
                d = pn[b.atom2.index] - pn[b.atom1.index]; d -= boxn * np.round(d / boxn)   # image minimale
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-6:
                    uu.append(d / nrm)
            uu = np.array(uu)
            Q = (3.0 * np.einsum("ia,ib->ab", uu, uu) / len(uu) - np.eye(3)) / 2.0
            S_nem = float(np.linalg.eigvalsh(Q).max())
        props["nematic_order"] = round(S_nem, 3)

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

        # ── Diélectrique statique + auto-diffusion — OPT-IN (DIELECTRIC=1) ──
        # CONVERGENT LENTEMENT (le diélectrique demande ~ns pour converger ⟨M²⟩) → hors run standard ;
        # coché par l'utilisateur (coût +~10-30 min). On réutilise l'état 300 K + V_m3 du bloc CED.
        if os.environ.get("DIELECTRIC", "0") == "1":
            with P.block("10c_dielec_diff"):
                nbf = next(f for f in omm_system.getForces() if isinstance(f, NonbondedForce))
                q = np.array([nbf.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                              for i in range(omm_system.getNumParticles())])
                mols = [np.array([a.index for a in ch.atoms()]) for ch in omm_top.chains()]
                nfr = int(os.environ.get("DIEL_FRAMES", "200"))      # ~400 ps (opt-in → long)
                Ms, coms, box = [], [], None
                for _ in range(nfr):
                    sim.step(2 * spp)                                # 2 ps entre frames
                    s = sim.context.getState(getPositions=True)
                    p = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    box = np.diag(s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
                    Mtot, cm = np.zeros(3), []
                    for idx in mols:
                        r = p[idx]; r = r - box * np.round((r - r[0]) / box)   # molécule rendue entière
                        Mtot += (q[idx][:, None] * r).sum(0)                   # dipôle (e·nm), neutre→invariant
                        cm.append(r.mean(0))
                    Ms.append(Mtot); coms.append(np.array(cm))
                Ms = np.array(Ms); coms = np.array(coms)
                e_nm = 1.602176634e-19 * 1e-9
                var_M = float((Ms ** 2).sum(1).mean() - (Ms.mean(0) ** 2).sum())
                eps_st = 1.0 + var_M * e_nm ** 2 / (3 * 8.8541878128e-12 * V_m3 * 1.380649e-23 * 300.0)
                disp = np.diff(coms, axis=0); disp -= box * np.round(disp / box)
                unwr = np.cumsum(np.vstack([coms[:1], disp]), axis=0)
                msd = ((unwr - unwr[0]) ** 2).sum(2).mean(1)
                t_ps = np.arange(nfr) * 2.0
                D = max(0.0, float(np.polyfit(t_ps[nfr // 2:], msd[nfr // 2:], 1)[0]) / 6.0 * 1e-6) if nfr > 6 else None
            props["static_dielectric"] = round(float(eps_st), 2)             # ⚠ encore sous-convergé
            props["self_diffusion_m2s"] = float(f"{D:.2e}") if D is not None else None  # ~0 au verre

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
        if os.environ.get("MECH_TENSILE", "0") == "1":
            print("  → E/ν par traction uniaxiale (stress-strain, latéral relaxé)…", flush=True)
            with P.block("11_tensile"):
                from openmm import MonteCarloAnisotropicBarostat, Vec3
                for idx in reversed(range(omm_system.getNumForces())):   # retire le barostat isotrope
                    if isinstance(omm_system.getForce(idx), MonteCarloBarostat):
                        omm_system.removeForce(idx)
                # barostat anisotrope : relaxe X,Y à mech_p bar (=1 défaut ; >1 = diag densité exp),
                # NE scale PAS Z (piloté par nous).
                abaro = MonteCarloAnisotropicBarostat(Vec3(mech_p, mech_p, mech_p) * unit.bar,
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

    # ── Conductivité thermique κ par reverse-NEMD (flux imposé HEX) — OPT-IN (THERMAL=1) ──
    # Grosse propriété RadonPy restante. Coûteux (~1 ns) + sous-estimé (taille finie) → coché par
    # l'utilisateur. On part de l'état verre 300 K courant de `sim` (ré-équilibré dans la fonction).
    if os.environ.get("THERMAL", "0") == "1":
        with P.block("13_thermal_nemd"):
            st_t = sim.context.getState(getPositions=True)
            tp = st_t.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            tb = np.diag(st_t.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            kappa, tinfo = thermal_conductivity_nemd(omm_top, omm_system, plat, tp, tb,
                                                     target_T=float(os.environ.get("MECH_T", "300")))
        if kappa is not None:
            props["thermal_conductivity_WmK"] = round(float(kappa), 3)
            props["thermal_nemd_dT_K"] = tinfo["dT_K"]
        print(f"     κ={kappa} W/m/K | ΔT={tinfo.get('dT_K')}K | ∇T={tinfo.get('gradT_Kpm')} K/m"
              f" | box_z={tinfo.get('box_z_nm')}nm | ok={tinfo.get('frac_ok')}"
              f"\n     T(z)={tinfo.get('Tprofile')}\n     T_moy(t)={tinfo.get('Tmean_hist')}", flush=True)

    # Propriétés électroniques QM (xtb sur le motif) : HOMO/LUMO/gap, dipôle, polarisabilité. ~secondes,
    # gracieux si xtb absent. (Info NOUVELLE non dérivable de la MD ; pertinence optique/électronique.)
    with P.block("12_xtb_electronic"):
        qm_elec = xtb_electronic(smiles)
    props.update(qm_elec)
    # Indice de réfraction QM (Lorentz-Lorenz avec la polarisabilité xtb) en plus de l'estimation Crippen.
    if qm_elec.get("polarizability_A3") and props.get("density_300K"):
        # φ = (4π/3)·N_A·α·ρ/M_unit ; α en cm³, ρ en g/cm³, M_unit en g/mol. (α du motif cappé H ≈ par-unité.)
        alpha_cm3 = qm_elec["polarizability_A3"] * 1e-24
        phi = 4.0 / 3.0 * 3.14159265 * 6.02214076e23 * alpha_cm3 * props["density_300K"] / m_chain * n_units
        n_qm = ((1 + 2 * phi) / (1 - phi)) ** 0.5 if 0 < phi < 1 else None
        if n_qm:
            props["refractive_index_QM"] = round(float(n_qm), 3)

    print("\n=== PROPRIÉTÉS ===")
    print(json.dumps(props, indent=1))
    P.report()


if __name__ == "__main__":
    main()
