"""Webapp LOCALE pour le pipeline MD : dessine le motif (éditeur Ketcher) + paramètres → soumet sur
CRIANN → logs en direct → tableau des propriétés. RÉUTILISE tout le backend de cli.py (validation,
drapeau de risque, fenêtre, soumission, parsing). Le calcul reste sur CRIANN (le laptop ne calcule pas).

Lancer :  uv run polyweb   → ouvre http://127.0.0.1:8000
L'éditeur de molécule est Ketcher (Apache-2.0), servi LOCALEMENT depuis static/ketcher/standalone/.
"""
from __future__ import annotations

import ipaddress
import json
import re
import shlex
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import cli

app = FastAPI(title="polyMD — webapp")
STATIC = Path(__file__).parent / "static"
# sert le build Ketcher (static/ketcher/standalone/) et les autres fichiers statiques
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ───────────────────────── Sécurité : validation des entrées qui touchent CRIANN ─────────────────────────
# Tout ce qui peut finir dans une commande SSH est validé par liste blanche AVANT interpolation.
# Un jobid SLURM est un entier ; un out_rel ne peut être QUE le format que `submit()` produit
# (`.tgcli/<slug>_<jobid>.out`, slug = alphanum + '_') → bloque injection shell ET traversée de chemin.
JOBID_RE = re.compile(r"^[0-9]{1,12}$")
OUTREL_RE = re.compile(r"^\.tgcli/[A-Za-z0-9_]+\.out$")
TIME_RE = re.compile(r"^[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$")
PARTITIONS = {"gpu", "gpu_debug", "gpu_h200"}
MAX_SMILES = 2000          # garde-fou DoS : un PSMILES réel fait quelques dizaines de caractères
MAX_KET = 2_000_000        # un dessin Ketcher volumineux reste bien en-deçà


def _check_jobid(j: str) -> str:
    if not JOBID_RE.match(str(j or "")):
        raise ValueError("jobid invalide")
    return str(j)


def _check_outrel(o: str) -> str:
    if not OUTREL_RE.match(o or ""):
        raise ValueError("out_rel invalide")
    return o


@app.middleware("http")
async def _guard(request: Request, call_next):
    """Anti DNS-rebinding / CSRF : on n'accepte que les requêtes dont le Host est une IP littérale
    ou localhost (un site malveillant qui « rebind » un nom de domaine vers 127.0.0.1 enverrait un
    Host = nom de domaine → rejeté). N'entrave pas l'accès LAN par http://<ip>:8000."""
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    ok = host in ("localhost", "127.0.0.1", "::1", "")
    if not ok:
        try:
            ipaddress.ip_address(host)
            ok = True
        except ValueError:
            ok = False
    if not ok:
        return JSONResponse({"error": "host_not_allowed"}, status_code=403)
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def index():
    # no-store : Firefox/Chrome ne doivent JAMAIS servir une version cachée de la page
    # (sinon un ancien index.html sans le retour visible du bouton reste collé en cache).
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


def _num(x):
    """Cast tolérant : '' / None / 'null' → None, sinon float (ou None si non numérique)."""
    if x in (None, "", "null"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ───────────────────────── Persistance des runs (rechargement / reattach par N° de job) ─────────────────────────
# Un run aveugle = 3-5 fenêtres. On écrit un MANIFESTE serveur (clé = jobid de la fenêtre centrale) qui
# fait AUTORITÉ : le stage-2 ajoute des fenêtres pendant le streaming, donc un client rechargé doit relire
# l'état serveur. Le client ne garde que le run_id en localStorage. /api/run/{N} retrouve le manifeste par
# run_id OU par n'importe quel jobid contenu → on peut recharger la page ou saisir un N° de job vu à l'écran.
RUNS_DIR = cli.LOCAL_OUT / "runs"
MARK = "<<<TGCLI_WIN:"            # marqueur de début de log par fenêtre dans un tick multi-fenêtres


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{_check_jobid(run_id)}.json"


def _save_manifest(m: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _run_path(m["run_id"]).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(run_id: str):
    try:
        p = _run_path(run_id)
    except ValueError:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_manifest(ident: str):
    """Par run_id direct, sinon par n'importe quel jobid de fenêtre contenu dans un manifeste."""
    ident = str(ident or "")
    if not JOBID_RE.match(ident):
        return None
    m = _load_manifest(ident)
    if m:
        return m
    if RUNS_DIR.exists():
        for p in RUNS_DIR.glob("*.json"):
            try:
                mm = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if any(str(w.get("jobid")) == ident for w in mm.get("windows", [])):
                return mm
    return None


@app.post("/api/validate")
async def validate(req: Request):
    """Validation live STRUCTURÉE du PSMILES (codes + nombres, le texte est localisé côté client) :
    points d'attache, atomes lourds, clés de risque + graine van-Krevelen → centre simulé auto."""
    d = await req.json()
    raw = (d.get("smiles") or "").strip()[:MAX_SMILES]
    ket = d.get("ket")
    if isinstance(ket, str) and len(ket) > MAX_KET:
        ket = None
    smiles = cli.normalize_psmiles(raw)
    if not smiles and ket:                          # repli sans WASM (Firefox durci) : KET natif → RDKit
        smiles = cli.ket_to_psmiles(ket)
    smiles = smiles or raw                           # tolère la sortie brute de Ketcher
    info = cli.psmiles_info(smiles)
    out = {**info, "norm": smiles,
           "risk": cli.chemistry_risk(smiles) if info["ok"] else []}
    if info["ok"]:
        hint = _num(d.get("tg_hint"))               # override optionnel de la graine
        if hint is not None:
            out.update({"seed_K": round(hint), "center_sim": round(1.5 * hint), "seed_src": "hint"})
        else:
            seed = cli.vk_seed_K(smiles)
            if seed is not None:
                out.update({"seed_K": round(seed), "center_sim": round(1.5 * seed), "seed_src": "vk"})
    return out


def _base_env_from(d: dict, smiles: str) -> dict:
    nu = _num(d.get("n_units"))
    n_units = int(nu) if nu else None
    return cli._collect_env(smiles, _num(d.get("box_a")), n_units, _num(d.get("t_step")),
                            _num(d.get("equil_ps")), _num(d.get("sample_ps")),
                            mech=bool(d.get("mech", True)),
                            tensile=bool(d.get("shear", d.get("tensile", False))),  # case "Module d'Young"
                            dielectric=bool(d.get("dielectric", False)),            # case "Diélectrique"
                            thermal=bool(d.get("thermal", False)))                  # case "Conductivité κ"


@app.post("/api/submit")
async def submit(req: Request):
    """Submit AVEUGLE : graine VK (ou tg_hint) → 3 fenêtres ∓150 K × graines. Écrit le manifeste serveur,
    renvoie {run_id, manifest}. La recette (poolage + densité/Prigogine-Defay) se fait au streaming."""
    d = await req.json()
    raw = (d.get("smiles") or "").strip()
    smiles = cli.normalize_psmiles(raw) or raw      # jamais de %91/CXSMILES jusqu'à CRIANN
    info = cli.psmiles_info(smiles)
    if not info["ok"]:
        return JSONResponse({"error": "invalid_psmiles", "code": info["code"],
                             "n_attach": info["n_attach"]}, status_code=400)
    # graine : Tg expérimentale estimée → centre SIMULÉ = 1.5×graine
    hint = _num(d.get("tg_hint"))
    if hint is not None:
        seed_K, seed_src = float(hint), "hint"
    else:
        seed_K = cli.vk_seed_K(smiles)
        seed_src = "vk"
        if seed_K is None:
            seed_K, seed_src = 373.0, "default"
    center_sim = 1.5 * seed_K
    base = _base_env_from(d, smiles)
    name = cli._slug(smiles)
    seeds = max(1, min(8, int(_num(d.get("seeds")) or 1)))       # borne le nb de jobs (anti-abus)
    # partition + temps SLURM sont écrits TELS QUELS dans des directives #SBATCH → liste blanche + format strict
    partition = d.get("partition", "gpu")
    if partition not in PARTITIONS:
        return JSONResponse({"error": "bad_partition"}, status_code=400)
    time_limit = d.get("time", "02:00:00")
    if not TIME_RE.match(str(time_limit)):
        return JSONResponse({"error": "bad_time"}, status_code=400)
    converge = not bool(d.get("no_converge", False))
    try:
        cli.preflight()
        if d.get("sync", True):
            cli._rsync("src/polymd/", "src/polymd/")
        windows, central_jobid = [], None
        for s in range(1, seeds + 1):
            for off in cli.BLIND_STAGE1:
                full = (off == 0.0)
                env = cli._blind_window_env(base, center_sim, off, full_props=full)
                if seeds > 1:
                    env["SEED"] = s
                tag = f"{name}_w1_s{s}_{off:+.0f}".replace("+", "p").replace("-", "m")
                jid, orel = cli.submit(env, tag, time_limit, partition)
                windows.append({"jobid": jid, "out_rel": orel, "offset": off,
                                "central": full, "seed": s})
                if full and s == 1:
                    central_jobid = jid
    except cli.SSHError as e:
        return JSONResponse({"error": "ssh", "detail": str(e)}, status_code=502)
    run_id = central_jobid or windows[0]["jobid"]
    manifest = {
        "run_id": run_id, "smiles": smiles, "name": name,
        "seed_K": round(seed_K, 1), "seed_src": seed_src, "center_sim": round(center_sim, 1),
        "seeds": seeds, "base_env": base, "partition": partition, "time": time_limit,
        "converge": converge, "stage": 1, "windows": windows, "done": False,
        "risk": cli.chemistry_risk(smiles), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_manifest(manifest)
    return {"run_id": run_id, "manifest": manifest}


@app.get("/api/run/{ident}")
def get_run(ident: str):
    """Retrouve un run par run_id ou par n'importe quel N° de job contenu (rechargement / reattach)."""
    m = _find_manifest(ident)
    if not m:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return {"run_id": m["run_id"], "manifest": m}


def _submit_stage2(m: dict) -> list:
    """Soumet les 2 fenêtres élargies ∓300 K (allégées) et renvoie leurs entrées de fenêtre."""
    new_w = []
    for s in range(1, m["seeds"] + 1):
        for off in cli.BLIND_STAGE2:
            env = cli._blind_window_env(m["base_env"], m["center_sim"], off, full_props=False)
            if m["seeds"] > 1:
                env["SEED"] = s
            tag = f"{m['name']}_w2_s{s}_{off:+.0f}".replace("+", "p").replace("-", "m")
            jid, orel = cli.submit(env, tag, m["time"], m["partition"])
            new_w.append({"jobid": jid, "out_rel": orel, "offset": off, "central": False, "seed": s})
    return new_w


@app.get("/api/blind_stream/{run_id}")
def blind_stream(run_id: str):
    """SSE de l'orchestrateur aveugle : streame toutes les fenêtres (un aller-retour SSH par tick), poole
    quand tout est sorti de la file, lance la recette ; si l'angle reste < 48° AJOUTE le stage-2 ∓300 K
    (côté serveur, manifeste mis à jour) puis continue ; sinon → propriétés + Tg poolée → done."""
    try:
        run_id = _check_jobid(run_id)
    except ValueError:
        return JSONResponse({"error": "bad_params"}, status_code=400)
    if _load_manifest(run_id) is None:
        return JSONResponse({"error": "unknown_run"}, status_code=404)

    def gen():
        from .tg_blind import blind_tg_recipe, converged
        m = _load_manifest(run_id)
        if m is None:                                       # supprimé entre-temps
            yield _sse({"done": True, "props": None}); return
        printed: dict = {}
        while True:
            try:                                            # défense en profondeur (manifeste = serveur)
                wins = [{"jobid": _check_jobid(w["jobid"]), "out_rel": _check_outrel(w["out_rel"]),
                         "offset": w["offset"], "central": w["central"], "seed": w["seed"]}
                        for w in m["windows"]]
            except (ValueError, KeyError):
                yield _sse({"done": True, "props": None}); return
            ids = ",".join(w["jobid"] for w in wins)
            parts = [f'echo "{MARK}{w["jobid"]}>>>"; '
                     f'cat {cli.REMOTE_ROOT}/{shlex.quote(w["out_rel"])} 2>/dev/null' for w in wins]
            parts.append(f'echo "{cli.SEP}"; squeue -j {ids} -h -o %i 2>/dev/null')
            try:
                tick = cli._ssh("; ".join(parts), check=False, timeout=60)
            except cli.SSHError as e:
                yield _sse({"log": f"[erreur SSH: {e}]"}); time.sleep(cli.POLL_S); continue
            body, _, state = tick.rpartition(cli.SEP)
            state = state.strip()
            logs = {}                                       # jobid → log complet
            for seg in body.split(MARK)[1:]:
                head, _, content = seg.partition(">>>")
                logs[head.strip()] = content
            for w in wins:                                  # nouvelles lignes par fenêtre
                full = logs.get(w["jobid"], "")
                pr = printed.get(w["jobid"], 0)
                if len(full) > pr:
                    printed[w["jobid"]] = len(full)
                    tag = f"{w['offset']:+.0f}K" + (f" s{w['seed']}" if m["seeds"] > 1 else "")
                    for line in full[pr:].splitlines():
                        yield _sse({"log": f"[{tag}] {line}"})
            running = [x for x in state.split() if x]
            yield _sse({"windowstate": {"done": len(wins) - len(running), "total": len(wins),
                                        "stage": m["stage"]}})
            if running:
                time.sleep(cli.POLL_S); continue
            # ── toutes les fenêtres terminées : poolage + recette ──
            curves = cli._pool_windows([w["out_rel"] for w in wins])
            tg, info = blind_tg_recipe(curves)
            angle = info.get("angle_deg")
            if (m.get("converge", True) and m["stage"] == 1 and tg is not None
                    and (angle is None or angle < cli.BLIND_ANGLE_THRESH)):
                try:
                    new_w = _submit_stage2(m)
                except cli.SSHError as e:
                    new_w = []
                    yield _sse({"log": f"[erreur SSH stage 2: {e}]"})
                if new_w:
                    m["windows"] += new_w; m["stage"] = 2
                    _save_manifest(m)
                    yield _sse({"stage2": {"added": len(new_w),
                                           "angle": round(angle, 1) if angle is not None else None,
                                           "jobids": [w["jobid"] for w in new_w]}})
                    time.sleep(cli.POLL_S); continue
            # ── finalisation : propriétés (fenêtres centrales) + Tg poolée aveugle ──
            per_seed = []
            for w in wins:
                if w["central"]:
                    p = cli.parse_props(logs.get(w["jobid"], ""))
                    if p:
                        p["seed"] = w["seed"]; per_seed.append(p)
            props = cli.aggregate_seeds(per_seed) if len(per_seed) > 1 else (per_seed[0] if per_seed else {})
            blind = None
            if tg is not None:
                tg_K = round(tg + 273.15, 1)
                props["Tg_pred"] = tg_K
                props.pop("Tg_pred_ci", None)               # incertitude = angle, pas le fit mono-fenêtre
                if info.get("Tg_sim") is not None:
                    props["Tg_sim"] = info["Tg_sim"]
                props["confidence"] = info.get("tier", "moyenne")
                _, conv_ok, _ = converged(angle if angle is not None else 0.0, m["stage"],
                                          angle_thresh=cli.BLIND_ANGLE_THRESH, max_calc=2)
                blind = {"tg_K": tg_K, "tg_C": round(tg, 1), "tier": info.get("tier"),
                         "angle": round(angle, 1) if angle is not None else None,
                         "value_obs": info.get("value_obs"), "pdefay": info.get("pdefay"),
                         "converged": conv_ok, "n_windows": len(wins)}
            m["done"] = True; m["result"] = blind; _save_manifest(m)
            yield _sse({"done": True, "props": props or None, "blind": blind, "risk": m.get("risk", [])})
            return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    import argparse
    import socket
    import uvicorn
    ap = argparse.ArgumentParser(prog="polyweb", description="Webapp locale polycli (calcul sur CRIANN).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface d'écoute. 127.0.0.1 = ta machine seulement (défaut). "
                         "0.0.0.0 = accessible depuis le réseau local (⚠ aucune authentification : "
                         "quiconque atteint le port soumet des jobs sur TON compte CRIANN).")
    ap.add_argument("--port", type=int, default=8000, help="Port d'écoute (défaut 8000).")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = args.host
        print(f"⚠ polyweb exposé sur le RÉSEAU ({args.host}) — AUCUNE authentification.")
        print(f"   Tes collègues : http://{lan_ip}:{args.port}/  (même réseau + pare-feu ouvert)")
        print(f"   Tout visiteur peut soumettre des jobs sur TON compte CRIANN.")
    else:
        print(f"polycli webapp → http://127.0.0.1:{args.port}  (local uniquement ; --host 0.0.0.0 pour le LAN)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
