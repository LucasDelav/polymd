"""tgcli — interface conviviale pour le pipeline MD (SMILES → propriétés), calcul sur CRIANN.

Le calcul (dynamique moléculaire, ~48k atomes, GPU A100) ne tourne PAS sur un laptop :
il est soumis sur le cluster CRIANN via SLURM. Cette CLI = le « front-end » local :
elle valide les entrées, soumet le job, **streame les logs en direct**, puis affiche
et sauvegarde le tableau des propriétés prédites.

Architecture : on NE touche pas à ~/tg_ml/run_pipeline.slurm (workflow manuel). La CLI
génère son propre job script isolé dans ~/tg_ml/.tgcli/ et le soumet.

Commandes :
  tgcli run       — soumettre un calcul, suivre, afficher les propriétés
  tgcli attach     — se rebrancher sur un job déjà lancé (par JOBID)
  tgcli status     — voir la file SLURM
  tgcli check      — diagnostiquer la connexion CRIANN

Lancement : `uv run tgcli run` (ou `.venv/bin/tgcli run`).
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ───────────────────────── Config CRIANN ─────────────────────────
HOST = "criann"                       # alias ~/.ssh/config (austral.criann.fr)
REMOTE_ROOT = "~/tg_ml"               # racine du projet sur le cluster
REMOTE_PY = "conda_openff_gpu/bin/python"   # interpréteur avec OpenMM-CUDA + OpenFF
REMOTE_JOBDIR = ".tgcli"              # sous-dossier isolé pour nos jobs (relatif à REMOTE_ROOT)
# Multiplexage SSH : la 1ʳᵉ connexion ouvre un master, les suivantes le réutilisent
# (pas de ré-auth → streaming rapide même avec un tick toutes les 6 s).
SSH_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/cm-tgcli-%r@%h:%p",
    "-o", "ControlPersist=120",
    "-o", "ConnectTimeout=15",
]
SEP = "<<<TGCLI_SEP>>>"               # séparateur log/état dans les ticks de streaming
POLL_S = 6                            # intervalle de polling pendant le streaming

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_OUT = PROJECT_ROOT / "outputs" / "tgcli"

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Pipeline MD polymères (SMILES → Tg, densité, CTE, modules) — calcul sur CRIANN.")


class SSHError(RuntimeError):
    pass


# ───────────────────────── Plomberie SSH ─────────────────────────
def _ssh(remote_cmd: str, *, stdin: Optional[str] = None, timeout: int = 60,
         check: bool = True) -> str:
    """Exécute une commande sur CRIANN, renvoie stdout. Lève SSHError si échec."""
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, HOST, remote_cmd],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or f"ssh code {proc.returncode}"
        raise SSHError(msg)
    return proc.stdout


def _rsync(local: str, remote_rel: str) -> None:
    """Pousse un fichier/dossier local vers REMOTE_ROOT/remote_rel via rsync (réutilise le master SSH).
    NB : pour un DOSSIER (local finissant par '/'), on ré-ajoute le slash que Path() supprime —
    sinon rsync copie le dossier DANS la cible (→ src/tg_ml/tg_ml/) au lieu d'en synchroniser le contenu."""
    src = str(PROJECT_ROOT / local)
    if local.endswith("/"):
        src += "/"
    subprocess.run(
        ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
         src, f"{HOST}:tg_ml/{remote_rel}"],
        check=True, capture_output=True, text=True,
    )


def preflight() -> None:
    """Vérifie connexion + présence de la stack distante. Lève SSHError sinon."""
    out = _ssh(
        f"cd {REMOTE_ROOT} 2>/dev/null && "
        f"([ -x {REMOTE_PY} ] && echo PY_OK || echo PY_NO); "
        f"([ -f scripts/pipeline.py ] && echo PIPE_OK || echo PIPE_NO)",
        timeout=25,
    )
    if "PY_OK" not in out:
        raise SSHError(f"{REMOTE_PY} introuvable sur CRIANN (env conda OpenMM-GPU).")
    if "PIPE_OK" not in out:
        raise SSHError("scripts/pipeline.py introuvable sur CRIANN (sync le projet d'abord).")


# ───────────────────────── Validation SMILES ─────────────────────────
def normalize_psmiles(raw: str) -> str:
    """Normalise une sortie d'éditeur (Ketcher) en PSMILES canonique `*…*`.

    Gère les formes que produit Ketcher quand on pose les points d'attache avec
    l'outil « attachment point » ou « R-group » plutôt que l'atome générique `*` :
      - extension CXSMILES séparée par un espace  (`*CC* |$;;;$|`)
      - atom-map + fermetures de cycle inter-fragments
        (`C%91C%92.[*:1]%91.[*:1]%92`  →  `*CC*`)
    Renvoie '' si non parsable (laisse l'appelant gérer l'échec)."""
    s = (raw or "").strip()
    if not s:
        return ""
    core = s.split()[0]                       # retire l'extension CXSMILES (après l'espace)
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(core)
        if mol is None:
            return ""
        for a in mol.GetAtoms():              # efface les atom-map (_AP1/_R1 → `*` nu)
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)
    except ImportError:
        return core


def ket_to_psmiles(ket: str) -> str:
    """Convertit le format natif KET de Ketcher (JSON pur, PAS de WebAssembly) en PSMILES `*…*`.

    Indispensable pour les navigateurs où WASM est désactivé (LibreWolf, Firefox durci) :
    `ketcher.getSmiles()` y échoue (Indigo = WASM), mais `ketcher.getKet()` marche (sérialiseur JS).
    Gère les deux façons de poser un point d'attache : atome générique `label:"*"`, ET la propriété
    `attachmentPoints` posée sur un atome (→ on ajoute autant d'atomes `*` factices liés). '' si échec."""
    import json
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        data = json.loads(ket)
    except Exception:
        return ""
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
          3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    # molécules référencées par root.nodes (ordre du dessin), sinon tout molN
    mol_keys = [n["$ref"] for n in data.get("root", {}).get("nodes", [])
                if isinstance(n, dict) and isinstance(data.get(n.get("$ref")), dict)
                and data[n["$ref"]].get("type") == "molecule"]
    if not mol_keys:
        mol_keys = [k for k, v in data.items()
                    if isinstance(v, dict) and v.get("type") == "molecule"]
    rw = Chem.RWMol()
    for mk in mol_keys:
        mol = data[mk]
        atoms = mol.get("atoms", [])
        local = {}
        for i, at in enumerate(atoms):
            label = at.get("label") or "*"
            if at.get("type") == "rg-label" or label.startswith("R") or label in ("*", "A", "Q"):
                a = Chem.Atom(0)                       # point d'attache / R-group / atome générique
            else:
                try:
                    a = Chem.Atom(label)
                except Exception:
                    a = Chem.Atom(0)
            if at.get("charge"):
                a.SetFormalCharge(int(at["charge"]))
            local[i] = rw.AddAtom(a)
        for b in mol.get("bonds", []):
            ij = b.get("atoms", [])
            if len(ij) >= 2 and ij[0] in local and ij[1] in local:
                rw.AddBond(local[ij[0]], local[ij[1]], bt.get(b.get("type", 1), Chem.BondType.SINGLE))
        for i, at in enumerate(atoms):                 # propriété attachmentPoints (bitmask) → atomes `*`
            n_ap = bin(int(at.get("attachmentPoints") or 0)).count("1")
            for _ in range(n_ap):
                d = rw.AddAtom(Chem.Atom(0))
                rw.AddBond(local[i], d, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            return ""
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def validate_psmiles(smiles: str) -> tuple[bool, str]:
    """Vérifie que le SMILES est un PSMILES valide (2 points d'attache `*`).
    Utilise RDKit s'il est dispo ; sinon contrôle minimal sur le compte de `*`."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "SMILES non parsable par RDKit."
        n_star = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
        if n_star != 2:
            return False, f"PSMILES doit avoir 2 points d'attache `*` (trouvé : {n_star})."
        return True, f"OK ({mol.GetNumHeavyAtoms()} atomes lourds, 2 points d'attache)."
    except ImportError:
        n_star = smiles.count("*")
        if n_star != 2:
            return False, f"PSMILES doit contenir 2 `*` (trouvé : {n_star}). [RDKit absent : contrôle minimal]"
        return True, "OK (contrôle minimal, RDKit absent)."


def psmiles_info(smiles: str) -> dict:
    """Validation STRUCTURÉE (pour la webapp i18n) : renvoie des codes + nombres, aucun texte en dur.
    code ∈ {ok, empty, unparsable, attach}. n_heavy = atomes lourds, n_attach = points d'attache `*`."""
    if not (smiles or "").strip():
        return {"ok": False, "code": "empty", "n_heavy": 0, "n_attach": 0}
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        n = smiles.count("*")
        return {"ok": n == 2, "code": "ok" if n == 2 else "attach", "n_heavy": None, "n_attach": n}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "code": "unparsable", "n_heavy": 0, "n_attach": 0}
    n_star = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_star != 2:
        return {"ok": False, "code": "attach", "n_heavy": n_heavy, "n_attach": n_star}
    return {"ok": True, "code": "ok", "n_heavy": n_heavy, "n_attach": 2}


# Motifs à liaisons H inter-chaînes FORTES/DIRECTIONNELLES : montent fortement la vraie Tg et
# sont sous-rendus par un FF à charges fixes + non capturés par la correction cinétique ÷1.50.
# → quand présents, la Tg prédite est probablement SOUS-ESTIMÉE (heuristique calée sur Nylon-6,
# ΔTg −86, + raisonnement physique + consensus littérature sur polyamides/polyuréthanes).
RISK_SMARTS = [("amide", "[NX3][CX3]=[OX1]"), ("urethane", "[NX3][CX3](=[OX1])[OX2]"),
               ("urea", "[NX3][CX3](=[OX1])[NX3]"), ("carboxylic_acid", "[CX3](=[OX1])[OX2H1]")]
# Libellés français des clés de risque (la clé canonique sert d'identifiant pour l'i18n de la webapp).
RISK_LABELS_FR = {"amide": "amide", "urethane": "uréthane", "urea": "urée",
                  "carboxylic_acid": "acide carboxylique"}


def chemistry_risk(smiles: str) -> list:
    """Détecte les motifs à risque de SOUS-ESTIMATION de Tg, sur le DIMÈRE (les liaisons
    inter-unités comme l'amide de Nylon n'existent pas dans le monomère seul). Renvoie la liste
    des motifs trouvés ([] si aucun ; [] aussi si RDKit absent)."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return []
    unit = Chem.MolFromSmiles(smiles)
    if unit is None:
        return []
    dummies = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    mol = unit
    if len(dummies) == 2:                       # construit un dimère tête-à-queue (topologie)
        head_d, tail_d = dummies
        nat = unit.GetNumAtoms()
        nbr = lambda d: unit.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx()
        rw = Chem.RWMol(Chem.CombineMols(unit, unit))
        rw.AddBond(nbr(tail_d), nbr(head_d) + nat, Chem.BondType.SINGLE)
        for idx in sorted([tail_d, head_d + nat], reverse=True):
            rw.RemoveAtom(idx)
        cand = rw.GetMol()
        try:
            Chem.SanitizeMol(cand); mol = cand
        except Exception:
            mol = unit
    return [name for name, sma in RISK_SMARTS if mol.HasSubstructMatch(Chem.MolFromSmarts(sma))]


# ───────────────────────── Job script + soumission ─────────────────────────
def build_job_script(env: dict, out_rel: str, job_name: str,
                     time_limit: str, partition: str) -> str:
    """Génère le script SLURM (valeurs bakées en single-quote → robuste au `*`/parenthèses du SMILES)."""
    exports = "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time={time_limit}
#SBATCH --output={out_rel}
cd {REMOTE_ROOT} || exit 1
export FF_CUDA=1
{exports}
echo "=== tgcli | début $(date '+%F %T') | $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
{REMOTE_PY} -u scripts/pipeline.py
echo "=== tgcli | fin $(date '+%F %T') ==="
"""


def submit(env: dict, job_name: str, time_limit: str, partition: str) -> tuple[str, str]:
    """Écrit le job script sur CRIANN et le soumet. Renvoie (jobid, chemin_sortie_relatif)."""
    token = time.strftime("%Y%m%d-%H%M%S")
    script_rel = f"{REMOTE_JOBDIR}/{job_name}_{token}.slurm"
    out_rel = f"{REMOTE_JOBDIR}/{job_name}_%j.out"
    script = build_job_script(env, out_rel, job_name, time_limit, partition)

    _ssh(f"mkdir -p {REMOTE_ROOT}/{REMOTE_JOBDIR}")
    _ssh(f"cat > {REMOTE_ROOT}/{script_rel}", stdin=script)
    out = _ssh(f"cd {REMOTE_ROOT} && sbatch {script_rel}")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise SSHError(f"sbatch n'a pas renvoyé de job id :\n{out}")
    jobid = m.group(1)
    return jobid, f"{REMOTE_JOBDIR}/{job_name}_{jobid}.out"


# ───────────────────────── Streaming + état ─────────────────────────
def job_state(jobid: str) -> str:
    """État SLURM du job (PENDING/RUNNING/...) ou "" s'il a quitté la file."""
    return _ssh(f"squeue -j {jobid} -h -o %T 2>/dev/null", check=False).strip()


def stream(jobid: str, out_rel: str) -> str:
    """Streame le fichier de sortie en direct jusqu'à fin du job. Renvoie le log complet."""
    out_path = f"{REMOTE_ROOT}/{out_rel}"
    printed = 0
    log = ""
    running_announced = False

    with console.status("[bold cyan]En file d'attente CRIANN…", spinner="dots") as status:
        while True:
            tick = _ssh(
                f'cat {out_path} 2>/dev/null; echo "{SEP}"; squeue -j {jobid} -h -o %T 2>/dev/null',
                check=False, timeout=40,
            )
            log, _, state = tick.rpartition(SEP)
            state = state.strip()

            if log and not running_announced:
                status.stop()
                console.rule(f"[bold green]Job {jobid} — logs en direct")
                running_announced = True
            elif state and not running_announced:
                status.update(f"[bold cyan]Job {jobid} : {state}…")

            new = log[printed:]
            if new:
                sys.stdout.write(new)
                sys.stdout.flush()
                printed = len(log)

            if not state:                       # job sorti de la file → terminé
                break
            time.sleep(POLL_S)

    # Capture finale (épilogue de comptabilité CRIANN écrit après la sortie squeue).
    time.sleep(2)
    final = _ssh(f"cat {out_path} 2>/dev/null", check=False)
    if len(final) > printed:
        sys.stdout.write(final[printed:])
        sys.stdout.flush()
        log = final
    if running_announced:
        console.rule()
    return log


# ───────────────────────── Parsing + rendu des propriétés ─────────────────────────
def parse_props(log: str) -> Optional[dict]:
    """Extrait le bloc JSON des propriétés (après le marqueur, avant l'épilogue SLURM)."""
    m = re.search(r"=== PROPRIÉTÉS ===", log)
    if not m:
        return None
    rest = log[m.end():]
    start = rest.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(rest)):
        if rest[i] == "{":
            depth += 1
        elif rest[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(rest[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# label, unité, fiabilité (cf PIPELINE.md)
PROP_META = {
    "Tg_pred":               ("Tg prédite",                 "K",        "[green]✅ fiable (MAE ~13 K, 30 polym.)"),
    "Tg_sim":                ("Tg simulée (avant ÷1.50)",   "K",        "[dim]info"),
    "density_300K":          ("Densité @300 K",             "g/cm³",    "[green]✅ fiable (~5 %)"),
    "FFV":                   ("Volume libre (FFV)",         "—",        "[green]✅ physique (convention 1.3)"),
    "Rg_nm":                 ("Rayon de giration",          "nm",       "[yellow]🟡 borne basse (chaînes effondrées)"),
    "Ree_nm":                ("Distance bout-à-bout",       "nm",       "[yellow]🟡 borne basse (chaînes effondrées)"),
    "refractive_index":      ("Indice de réfraction",       "—",        "[green]✅ (MAE ~0.03 vs exp)"),
    "Cp_JgK":                ("Cp",                         "J/g/K",    "[green]✅ corrigé ÷2.27 (~15%)"),
    "K_GPa":                 ("Module de compression K",    "GPa",      "[yellow]🟡 dispersé (fluct. de volume)"),
    "compressibility_1_GPa": ("Compressibilité isotherme",  "GPa⁻¹",    "[yellow]🟡 (= 1/K, dispersé)"),
    "solubility_delta":      ("Paramètre de solubilité δ",  "MPa^0.5",  "[green]✅ corrigé ×1.25 (~10%)"),
    "CED_MPa":               ("Énergie cohésive (CED)",     "MPa",      "[dim]info (brut, avant ×1.25²)"),
    "G_GPa_EXPERIMENTAL":    ("Cisaillement G",             "GPa",      "[yellow]⚠️ EXPÉRIMENTAL non validé"),
    "E_GPa_EXPERIMENTAL":    ("Module de Young E",          "GPa",      "[yellow]⚠️ EXPÉRIMENTAL non validé"),
    "poisson_EXPERIMENTAL":  ("Coefficient de Poisson ν",   "—",        "[yellow]⚠️ EXPÉRIMENTAL non validé"),
}
ORDER = ["Tg_pred", "Tg_sim", "density_300K", "FFV", "Rg_nm", "Ree_nm",
         "refractive_index", "Cp_JgK", "K_GPa", "compressibility_1_GPa",
         "solubility_delta", "CED_MPa",
         "G_GPa_EXPERIMENTAL", "E_GPa_EXPERIMENTAL", "poisson_EXPERIMENTAL"]


def render_props(props: dict, jobid: str) -> None:
    table = Table(title=f"Propriétés prédites — job {jobid}", title_style="bold",
                  header_style="bold", show_lines=False)
    table.add_column("Propriété")
    table.add_column("Valeur", justify="right")
    table.add_column("Unité")
    table.add_column("Fiabilité")
    for key in ORDER:
        if key not in props:
            continue
        label, unit, rel = PROP_META[key]
        val = props[key]
        if val is None:
            sval = "n/a"
        else:
            sval = f"{val:g}" if isinstance(val, (int, float)) else str(val)
            ci = props.get(key + "_ci")
            if isinstance(ci, (int, float)):
                sval += f" ± {ci:g}"
        table.add_row(label, sval, unit, rel)
    console.print(table)
    console.print("[dim]± = intervalle de confiance 1σ (Tg/densité : covariance du fit ; "
                  "K : statistique des fluctuations de volume).[/dim]")
    if props.get("Tg_method") == "coude-2seg":
        console.print(f"[dim]• Tg via repli ROBUSTE (coude-2-segments) car l'hyperbole était instable "
                      f"— Tg_sim hyperbole aurait donné {props.get('Tg_sim_hyperbola','?')} K.[/dim]")
    if props.get("density_300K_extrapolated"):
        console.print("[dim]• Densité @300 K : extrapolée sous la fenêtre de refroidissement "
                      "(branche vitreuse du fit) — l'incertitude croît avec la distance.[/dim]")


def save_results(name: str, jobid: str, smiles: str, env: dict,
                 props: Optional[dict], log: str) -> tuple[Path, Path]:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    log_path = LOCAL_OUT / f"{name}_{jobid}.log"
    json_path = LOCAL_OUT / f"{name}_{jobid}.json"
    log_path.write_text(log)
    json_path.write_text(json.dumps(
        {"jobid": jobid, "name": name, "smiles": smiles, "inputs": env,
         "properties": props, "log_file": log_path.name,
         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2, ensure_ascii=False,
    ))
    return json_path, log_path


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)[:12] or "poly"


# Marge (K, espace SIMULÉ) ajoutée de part et d'autre de 1.5×plage : juste assez pour que le
# coude ρ(T) ne tombe pas pile sur une borne. Petite par défaut (l'utilisateur est responsable
# de la fiabilité ; le pipeline signale après coup si le fit est douteux). Réglable via --margin.
DEFAULT_MARGIN = 30.0
T_LOW_FLOOR = 120.0          # plancher dur du pipeline
DEFAULT_RANGE = "250-500"    # « aucune idée » → scan large (Tg expérimentale, K)


def parse_range(s: str) -> tuple[float, float]:
    """Parse 'LO-HI' (séparateur -, :, , ou espace) → (lo, hi) triés. Lève ValueError sinon."""
    nums = re.findall(r"\d+\.?\d*", s)
    if len(nums) < 2:
        raise ValueError(f"attendu 'LO-HI' (ex. 350-420), reçu '{s}'.")
    lo, hi = float(nums[0]), float(nums[1])
    if lo == hi:
        raise ValueError("les deux bornes sont identiques.")
    return (lo, hi) if lo < hi else (hi, lo)


def resolve_temperature(tg_exp: Optional[float], tg_range: Optional[str],
                        margin: float = DEFAULT_MARGIN) -> tuple[dict, str, str]:
    """Construit les variables de fenêtre. Renvoie (env_fenêtre, mode, description_expérimentale).

    Mode 'plage' : la plage est en Tg EXPÉRIMENTALE ; on la mappe sur la fenêtre SIMULÉE
    [1.5×lo − margin, 1.5×hi + margin] via TG_SIM_PRIOR/WIN_HI/WIN_LO (sans toucher au pipeline).
    Mode 'point' : on passe TG_EXP, le pipeline fait 1.5× + marges par défaut (80/140).
    """
    if tg_range is not None:
        lo, hi = parse_range(tg_range)
        t_high = 1.5 * hi + margin
        t_low = max(1.5 * lo - margin, T_LOW_FLOOR)
        env = {"TG_EXP": round((lo + hi) / 2, 1),        # pour l'affichage/header seulement
               "TG_SIM_PRIOR": round(t_low, 1),
               "WIN_HI": round(t_high - t_low, 1),
               "WIN_LO": 0}
        return env, "plage", f"plage Tg_exp {lo:g}–{hi:g} K (marge ±{margin:g} K)"
    return {"TG_EXP": tg_exp}, "point", f"Tg_exp {tg_exp:g} K (fenêtre ×1.5)"


def effective_window(env: dict, t_step: float) -> tuple[float, float, int]:
    """Reproduit le calcul de fenêtre du pipeline depuis l'env final → (t_high, t_low, n_paliers)."""
    prior = env["TG_SIM_PRIOR"] if "TG_SIM_PRIOR" in env else 1.5 * env.get("TG_EXP", 373)
    win_hi = env.get("WIN_HI", 80.0)
    win_lo = env.get("WIN_LO", 140.0)
    t_high = prior + win_hi
    t_low = max(prior - win_lo, T_LOW_FLOOR)
    n_paliers = int(round((t_high - t_low) / t_step)) + 1
    return t_high, t_low, n_paliers


def _collect_env(smiles, box_a, n_units, t_step, equil_ps, sample_ps,
                 mech, shear, cool_indep) -> dict:
    """Paramètres NON liés à la fenêtre de température. N'inclut que ceux explicitement fixés."""
    env = {"SMILES": smiles}
    opt = {"BOX_A": box_a, "N_UNITS": n_units, "T_STEP": t_step,
           "EQUIL_PS": equil_ps, "SAMPLE_PS": sample_ps}
    for k, v in opt.items():
        if v is not None:
            env[k] = v
    if mech is not None:
        env["MECH"] = 1 if mech else 0
    if shear:
        env["SHEAR"] = 1
    if cool_indep:
        env["COOL_INDEP"] = 1
    return env


# ───────────────────────── Multi-seed (réduire l'incertitude) ─────────────────────────
AGG_KEYS = ["Tg_sim", "Tg_pred", "density_300K", "CTE_glass_1e6", "K_GPa",
            "FFV", "Rg_nm", "Ree_nm", "refractive_index", "Cp_JgK", "solubility_delta"]


def wait_for_jobs(jobids: list, label: str = "seeds") -> None:
    """Attend que tous les jobids quittent la file SLURM (spinner avec compteur)."""
    ids = list(jobids)
    with console.status(f"[cyan]Calcul des {len(ids)} {label}…", spinner="dots") as st:
        while True:
            out = _ssh("squeue --me -h -o %i 2>/dev/null", check=False, timeout=40)
            running = [j for j in ids if j in out]
            if not running:
                break
            st.update(f"[cyan]{len(ids) - len(running)}/{len(ids)} {label} terminés "
                      f"(en cours : {', '.join(running)})…")
            time.sleep(POLL_S * 2)


def aggregate_seeds(per_seed: list) -> dict:
    """Agrège N jeux de propriétés (1 par seed) → moyenne + erreur-type σ/√N (et σ brut).
    L'erreur-type rétrécit en √N → c'est elle qu'on affiche comme ± quand N>1."""
    import statistics
    n = len(per_seed)
    agg = {"n_seeds": n, "seeds": [p.get("seed") for p in per_seed],
           "n_reliable": sum(1 for p in per_seed if p.get("fit_reliable")),
           "density_300K_extrapolated": any(p.get("density_300K_extrapolated") for p in per_seed)}
    for key in AGG_KEYS:
        vals = [p[key] for p in per_seed if isinstance(p.get(key), (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            agg[key + "_ci"] = round(sd / len(vals) ** 0.5, 4)   # erreur-type = σ/√N
            agg[key + "_std"] = round(sd, 4)
        else:                                                     # 1 seul → CI du fit unique
            agg[key + "_ci"] = per_seed[0].get(key + "_ci")
        agg[key] = round(mean, 4)
    agg["fit_reliable"] = agg["n_reliable"] >= (n + 1) // 2       # majorité de fits fiables
    return agg


def _finish_seeds(name: str, smiles: str, jobs: list) -> None:
    """jobs = list de (seed, jobid, out_rel, env). Récupère, parse, agrège, affiche, sauve."""
    per_seed = []
    for seed, jid, out_rel, env in jobs:
        log = _ssh(f"cat {REMOTE_ROOT}/{out_rel} 2>/dev/null", check=False)
        p = parse_props(log)
        save_results(f"{name}_s{seed}", jid, smiles, env, p, log)
        if p:
            per_seed.append(p)
    console.print()
    if not per_seed:
        console.print("[yellow]⚠ Aucune seed exploitable.[/yellow]")
        return
    agg = aggregate_seeds(per_seed)
    render_props(agg, f"{name} — {agg['n_seeds']} seeds")
    console.print(f"[dim]Moyenne sur {agg['n_seeds']} seeds (graines {agg['seeds']}). "
                  f"± = erreur-type σ/√N. Fits fiables : {agg['n_reliable']}/{agg['n_seeds']}.[/dim]")
    # détail par seed pour Tg_pred
    detail = ", ".join(f"s{p.get('seed')}={p.get('Tg_pred')}" for p in per_seed)
    console.print(f"[dim]Tg_pred par seed : {detail}[/dim]")
    if not agg["fit_reliable"]:
        console.print("[yellow]⚠ Majorité de fits non fiables — les moyennes sont à valider.[/yellow]")


# ───────────────────────── Commandes ─────────────────────────
@app.command()
def run(
    smiles: str = typer.Option(..., "--smiles", "-s", prompt="PSMILES du monomère (2 `*`)",
                               help="PSMILES, ex. polystyrène : *CC(*)c1ccccc1"),
    tg_exp: Optional[float] = typer.Option(None, "--tg-exp", "-t",
                                 help="Tg connue/estimée (K) → fenêtre étroite auto-centrée sur 1.5× (le + rapide)."),
    tg_range: Optional[str] = typer.Option(None, "--tg-range", "-r",
                                 help="Plage de Tg EXPÉRIMENTALE estimée 'LO-HI' (K), ex. 350-420. Alternative à --tg-exp."),
    margin: float = typer.Option(DEFAULT_MARGIN, "--margin",
                                 help="Marge K de part et d'autre de 1.5×plage (mode plage). Plus petit = plus rapide, "
                                      "fit plus risqué (un avertissement signalera un fit douteux)."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Étiquette du run (défaut : dérivée du SMILES)."),
    partition: str = typer.Option("gpu", "--partition", help="Partition GPU : gpu | gpu_debug | gpu_h200."),
    time_limit: str = typer.Option("02:00:00", "--time", help="Limite de temps SLURM (HH:MM:SS)."),
    seeds: int = typer.Option(1, "--seeds", min=1, help="Nb de graines aléatoires (jobs indépendants). "
                              ">1 → moyenne ± erreur-type (σ/√N) : réduit l'incertitude affichée."),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Pousser pipeline.py + src/tg_ml avant soumission."),
    detach: bool = typer.Option(False, "--detach", help="Soumettre puis rendre la main (pas de streaming)."),
    force: bool = typer.Option(False, "--force", help="Ignorer l'échec de validation du SMILES."),
    # paramètres avancés (None = défaut du pipeline)
    box_a: Optional[float] = typer.Option(None, help="Arête de boîte Å (défaut 80 = ~48k atomes ; NE PAS réduire)."),
    n_units: Optional[int] = typer.Option(None, help="Degré de polymérisation (défaut 40)."),
    t_step: Optional[float] = typer.Option(None, help="Pas de température K (défaut 20)."),
    equil_ps: Optional[float] = typer.Option(None, help="Équilibration par palier ps (défaut 100)."),
    sample_ps: Optional[float] = typer.Option(None, help="Échantillonnage par palier ps (défaut 50)."),
    win_hi: Optional[float] = typer.Option(None, help="Fenêtre au-dessus du prior K (défaut 80)."),
    win_lo: Optional[float] = typer.Option(None, help="Fenêtre en-dessous du prior K (défaut 140)."),
    tg_prior: Optional[float] = typer.Option(None, help="Centre de fenêtre K (défaut 1.5×Tg_exp)."),
    mech: Optional[bool] = typer.Option(None, "--mech/--no-mech", help="Calculer K (défaut : oui)."),
    shear: bool = typer.Option(False, "--shear", help="Activer G/E/ν (EXPÉRIMENTAL, non validé)."),
    cool_indep: bool = typer.Option(False, "--cool-indep", help="Paliers indépendants depuis le snapshot de fonte."),
):
    """Soumettre un calcul MD sur CRIANN, suivre en direct, afficher les propriétés.

    Température : --tg-exp T (estimation ponctuelle → fenêtre étroite, le + rapide) OU
    --tg-range LO-HI (plage de Tg expérimentale → fenêtre élargie). Les deux sont optionnels :
    sans rien, mode interactif (ou scan large 250-500 K en non-interactif).
    """
    name = name or _slug(smiles)

    if tg_exp is not None and tg_range is not None:
        console.print("[red]✗ Donne soit --tg-exp soit --tg-range, pas les deux.[/red]")
        raise typer.Exit(1)

    # Aucune température fournie → demander (interactif) ou scan large (non-interactif).
    if tg_exp is None and tg_range is None:
        if sys.stdin.isatty():
            ans = typer.prompt("Tg estimée en K, ou plage 'LO-HI' (Entrée = scan large 250-500)",
                               default="", show_default=False).strip()
            nums = re.findall(r"\d+\.?\d*", ans)
            if len(nums) >= 2:
                tg_range = ans
            elif len(nums) == 1:
                tg_exp = float(nums[0])
            else:
                tg_range = DEFAULT_RANGE
        else:
            tg_range = DEFAULT_RANGE
            console.print(f"[yellow]Aucune Tg fournie → scan large {DEFAULT_RANGE} K (lent).[/yellow]")

    try:
        window_env, mode, exp_desc = resolve_temperature(tg_exp, tg_range, margin)
    except ValueError as e:
        console.print(f"[red]✗ Plage invalide :[/red] {e}")
        raise typer.Exit(1)

    # Surcharges manuelles (espace SIMULÉ) — hors mode plage qui les calcule lui-même.
    manual = {k: v for k, v in
              {"TG_SIM_PRIOR": tg_prior, "WIN_HI": win_hi, "WIN_LO": win_lo}.items() if v is not None}
    if manual and mode == "plage":
        console.print("[yellow]⚠ --tg-prior/--win-hi/--win-lo ignorés en mode plage.[/yellow]")
    elif manual:
        window_env.update(manual)

    # Libellé lisible pour le header du job distant (évite un "Tg_exp=<midpoint>" trompeur en mode plage).
    window_env["TG_DESC"] = exp_desc

    step = t_step if t_step is not None else 20.0
    t_high, t_low, n_paliers = effective_window(window_env, step)

    console.print(Panel.fit(
        f"[bold]SMILES[/bold]  {smiles}\n"
        f"[bold]Température[/bold]  {exp_desc}\n"
        f"[bold]Sweep simulé[/bold]  {t_low:.0f} → {t_high:.0f} K  "
        f"({n_paliers} paliers, pas {step:g} K)\n"
        f"[bold]run[/bold]  {name}   [bold]partition[/bold] {partition}",
        title="tgcli — pipeline MD → propriétés", border_style="cyan"))
    if n_paliers >= 22:
        console.print(f"[yellow]⚠ {n_paliers} paliers = run long. Resserre la plage "
                      f"ou augmente --t-step pour accélérer.[/yellow]")

    ok, msg = validate_psmiles(smiles)
    if ok:
        console.print(f"[green]✓[/green] SMILES : {msg}")
    else:
        console.print(f"[red]✗ SMILES invalide :[/red] {msg}")
        if not force:
            raise typer.Exit(1)
        console.print("[yellow]--force : on continue malgré tout.[/yellow]")

    risk = chemistry_risk(smiles)
    if risk:
        console.print(Panel(
            f"Motif(s) détecté(s) : [bold]{', '.join(RISK_LABELS_FR.get(r, r) for r in risk)}[/bold]\n"
            "→ liaisons H inter-chaînes fortes : la Tg prédite est probablement [bold]SOUS-ESTIMÉE[/bold] "
            "(le FF à charges fixes + la correction ÷1,50 sous-rendent ces réseaux ; cf. Nylon-6, ΔTg −86). "
            "Densité/CTE restent fiables.",
            title="⚠ Risque de sous-estimation de Tg", border_style="yellow"))

    if partition == "gpu_debug":
        console.print("[yellow]⚠ gpu_debug = 30 min max ; le pipeline peut dépasser (~15-28 min). "
                      "Utilise 'gpu' si risque de dépassement.[/yellow]")

    jobs: list = []
    jobid = out_rel = None
    try:
        with console.status("[cyan]Vérification de CRIANN…", spinner="dots"):
            preflight()
        if sync:
            with console.status("[cyan]Synchronisation du code (rsync)…", spinner="dots"):
                _rsync("scripts/pipeline.py", "scripts/pipeline.py")
                _rsync("src/tg_ml/", "src/tg_ml/")
            console.print("[green]✓[/green] Code synchronisé sur CRIANN.")

        env = _collect_env(smiles, box_a, n_units, t_step, equil_ps, sample_ps,
                           mech, shear, cool_indep)
        env.update(window_env)
        if seeds > 1:
            jobs = []
            for s in range(1, seeds + 1):
                senv = dict(env); senv["SEED"] = s
                jid, orel = submit(senv, f"{name}_s{s}", time_limit, partition)
                jobs.append((s, jid, orel, senv))
            console.print(f"[green]✓[/green] {seeds} seeds soumis : "
                          + ", ".join(j[1] for j in jobs))
        else:
            jobid, out_rel = submit(env, name, time_limit, partition)
    except SSHError as e:
        console.print(f"[red]✗ Erreur CRIANN :[/red] {e}")
        console.print("[dim]Connexion VPN/SSH active ? Teste : ssh criann hostname[/dim]")
        raise typer.Exit(1)

    if seeds > 1:
        if detach:
            console.print(f"[cyan]Détaché.[/cyan] Jobs : {', '.join(j[1] for j in jobs)}")
            raise typer.Exit(0)
        wait_for_jobs([j[1] for j in jobs])
        _finish_seeds(name, smiles, jobs)
        raise typer.Exit(0)

    console.print(f"[green]✓[/green] Job soumis : [bold]{jobid}[/bold]  (sortie : {REMOTE_ROOT}/{out_rel})")
    if detach:
        console.print(f"[cyan]Détaché.[/cyan] Suivre plus tard : [bold]tgcli attach {jobid}[/bold]")
        raise typer.Exit(0)

    log = stream(jobid, out_rel)
    _finish(name, jobid, smiles, env, log)


@app.command()
def attach(jobid: str = typer.Argument(..., help="JOBID SLURM à suivre."),
           name: Optional[str] = typer.Option(None, "--name", "-n", help="Étiquette pour la sauvegarde.")):
    """Se rebrancher sur un job tgcli déjà lancé et récupérer ses résultats."""
    try:
        listing = _ssh(f"ls {REMOTE_ROOT}/{REMOTE_JOBDIR}/*_{jobid}.out 2>/dev/null", check=False).strip()
        if not listing:
            console.print(f"[red]✗[/red] Aucun fichier de sortie pour le job {jobid} dans {REMOTE_JOBDIR}/.")
            raise typer.Exit(1)
        out_rel = listing.splitlines()[0].split("tg_ml/", 1)[-1]
        name = name or Path(out_rel).stem.rsplit("_", 1)[0]
        log = stream(jobid, out_rel)
    except SSHError as e:
        console.print(f"[red]✗ Erreur CRIANN :[/red] {e}")
        raise typer.Exit(1)
    _finish(name, jobid, "(attach)", {}, log)


@app.command()
def status():
    """Afficher la file SLURM de l'utilisateur sur CRIANN."""
    try:
        out = _ssh("squeue --me 2>/dev/null || squeue -u $USER", timeout=25)
    except SSHError as e:
        console.print(f"[red]✗ Erreur CRIANN :[/red] {e}")
        raise typer.Exit(1)
    console.print(out.rstrip() or "[dim]File vide.[/dim]")


@app.command()
def check():
    """Diagnostiquer la connexion et la stack distante CRIANN."""
    try:
        host = _ssh("hostname", timeout=20).strip()
        console.print(f"[green]✓[/green] SSH OK → {host}")
        preflight()
        console.print(f"[green]✓[/green] {REMOTE_PY} et scripts/pipeline.py présents.")
        console.print("[bold green]CRIANN prêt.[/bold green]")
    except SSHError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


def _finish(name: str, jobid: str, smiles: str, env: dict, log: str) -> None:
    props = parse_props(log)
    console.print()
    if props:
        render_props(props, jobid)
        if props.get("fit_reliable") is False:
            warns = props.get("fit_warnings") or ["raison non précisée"]
            body = "\n".join(f"• {w}" for w in warns)
            rec = props.get("fit_recommendation")
            if rec:
                arrow = {"plus haut": "↑", "plus bas": "↓", "plus large": "↔"}.get(
                    props.get("fit_direction") or "", "→")
                body += f"\n\n[bold]{arrow} Recommandation :[/bold] {rec}"
            console.print(Panel(body, title="⚠ Fit Tg peu fiable — résultat à valider",
                                border_style="yellow"))
    else:
        console.print("[yellow]⚠ Aucun bloc PROPRIÉTÉS trouvé (job échoué/interrompu ?). "
                      "Log brut sauvegardé.[/yellow]")
    json_path, log_path = save_results(name, jobid, smiles, env, props, log)
    console.print(f"\n[dim]Résultats : {json_path}\nLog : {log_path}[/dim]")


if __name__ == "__main__":
    app()
