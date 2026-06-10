"""Webapp LOCALE pour le pipeline MD : dessine le motif (éditeur JSME) + paramètres → soumet sur
CRIANN → logs en direct → tableau des propriétés. RÉUTILISE tout le backend de cli.py (validation,
drapeau de risque, fenêtre, soumission, parsing). Le calcul reste sur CRIANN (le laptop ne calcule pas).

Lancer :  uv run tgweb   → ouvre http://127.0.0.1:8000
L'éditeur de molécule est JSME (open-source, chargé depuis CDN) ; swappable par Ketcher (Apache-2.0).
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

app = FastAPI(title="tgcli — webapp")
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


@app.post("/api/validate")
async def validate(req: Request):
    """Validation live STRUCTURÉE du PSMILES (codes + nombres, le texte est localisé côté client) :
    points d'attache, atomes lourds, clés de risque, fenêtre de température + nb de paliers."""
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
        try:
            tg = float(d.get("tg_exp") or 373)
            step = _num(d.get("t_step")) or 20.0
            win, _, _ = cli.resolve_temperature(tg, None)
            t_hi, t_lo, n_pal = cli.effective_window(win, step)
            out.update({"win_lo": round(t_lo), "win_hi": round(t_hi), "paliers": n_pal})
        except Exception:
            pass
    return out


@app.post("/api/submit")
async def submit(req: Request):
    """Valide, (re)synchronise le code, soumet le job sur CRIANN. Renvoie le jobid + le fichier de sortie."""
    d = await req.json()
    raw = (d.get("smiles") or "").strip()
    smiles = cli.normalize_psmiles(raw) or raw      # jamais de %91/CXSMILES jusqu'à CRIANN
    info = cli.psmiles_info(smiles)
    if not info["ok"]:
        return JSONResponse({"error": "invalid_psmiles", "code": info["code"],
                             "n_attach": info["n_attach"]}, status_code=400)
    tg_exp = _num(d.get("tg_exp")) or 373.0
    win, _, desc = cli.resolve_temperature(tg_exp, None)
    win["TG_DESC"] = desc
    nu = _num(d.get("n_units"))
    n_units = int(nu) if nu else None
    env = cli._collect_env(smiles, _num(d.get("box_a")), n_units, _num(d.get("t_step")),
                           _num(d.get("equil_ps")), _num(d.get("sample_ps")),
                           mech=bool(d.get("mech", True)),
                           tensile=bool(d.get("tensile", False)),        # case "Module d'Young" (coûteux)
                           dielectric=bool(d.get("dielectric", False)))  # case "Diélectrique+diffusion" (coûteux)
    env.update(win)
    name = cli._slug(smiles)
    seeds = max(1, min(20, int(_num(d.get("seeds")) or 1)))      # borne aussi le nb de jobs (anti-abus)
    # partition + temps SLURM sont écrits TELS QUELS dans des directives #SBATCH du script → liste blanche
    # + format strict (sinon un retour-ligne injecterait des lignes de script exécutées sur le nœud).
    partition = d.get("partition", "gpu")
    if partition not in PARTITIONS:
        return JSONResponse({"error": "bad_partition"}, status_code=400)
    time_limit = d.get("time", "02:00:00")
    if not TIME_RE.match(str(time_limit)):
        return JSONResponse({"error": "bad_time"}, status_code=400)
    try:
        cli.preflight()
        if d.get("sync", True):
            cli._rsync("scripts/pipeline.py", "scripts/pipeline.py")
            cli._rsync("src/tg_ml/", "src/tg_ml/")
        if seeds > 1:
            # N jobs indépendants (1 graine chacun) → moyenne ± erreur-type σ/√N (cf. cli.aggregate_seeds)
            jobs = []
            for s in range(1, seeds + 1):
                senv = dict(env); senv["SEED"] = s
                jid, orel = cli.submit(senv, f"{name}_s{s}", time_limit, partition)
                jobs.append({"seed": s, "jobid": jid, "out_rel": orel})
            return {"multi": True, "jobs": jobs, "name": name, "risk": cli.chemistry_risk(smiles)}
        jobid, out_rel = cli.submit(env, name, time_limit, partition)
    except cli.SSHError as e:
        return JSONResponse({"error": "ssh", "detail": str(e)}, status_code=502)
    return {"jobid": jobid, "out_rel": out_rel, "name": name,
            "risk": cli.chemistry_risk(smiles)}


@app.get("/api/stream/{jobid}")
def stream(jobid: str, out_rel: str):
    """SSE : streame le fichier de sortie distant (cat incrémental + état squeue) jusqu'à fin du job."""
    try:                                            # jobid/out_rel viennent du client → validés strictement
        jobid = _check_jobid(jobid)
        out_rel = _check_outrel(out_rel)
    except ValueError:
        return JSONResponse({"error": "bad_params"}, status_code=400)
    # on ne quote QUE out_rel (déjà validé) — pas REMOTE_ROOT, dont le `~` doit rester expansé par le shell
    out_path = f"{cli.REMOTE_ROOT}/{shlex.quote(out_rel)}"     # quoting en défense de profondeur
    qjob = shlex.quote(jobid)

    def gen():
        printed = 0
        while True:
            try:
                tick = cli._ssh(f'cat {out_path} 2>/dev/null; echo "{cli.SEP}"; '
                                f'squeue -j {qjob} -h -o %T 2>/dev/null', check=False, timeout=40)
            except cli.SSHError as e:
                yield f"data: {json.dumps({'log': f'[erreur SSH: {e}]'})}\n\n"
                time.sleep(cli.POLL_S); continue
            logtxt, _, state = tick.rpartition(cli.SEP)
            state = state.strip()
            new = logtxt[printed:]
            if new:
                printed = len(logtxt)
                for line in new.splitlines():
                    yield f"data: {json.dumps({'log': line})}\n\n"
            if not state:                                  # job sorti de la file → terminé
                props = cli.parse_props(logtxt)
                yield f"data: {json.dumps({'done': True, 'props': props})}\n\n"
                break
            yield f"data: {json.dumps({'state': state})}\n\n"
            time.sleep(cli.POLL_S)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


SEED_MARK = "<<<TGCLI_SEED:"      # marqueur de début de log par graine dans un tick multi-seed


@app.get("/api/stream_seeds")
def stream_seeds(jobs: str):
    """SSE multi-seed : streame les N jobs (logs préfixés [sX]) en un seul aller-retour SSH par tick,
    puis agrège (moyenne ± erreur-type σ/√N) quand TOUS sont sortis de la file."""
    try:                                                         # jobs vient du client → tout validé
        raw_list = json.loads(jobs)
        if not isinstance(raw_list, list) or not (1 <= len(raw_list) <= 20):
            raise ValueError("jobs")
        job_list = [{"seed": int(j["seed"]),
                     "jobid": _check_jobid(j["jobid"]),
                     "out_rel": _check_outrel(j["out_rel"])} for j in raw_list]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return JSONResponse({"error": "bad_jobs"}, status_code=400)
    ids = ",".join(j["jobid"] for j in job_list)                 # jobids = entiers validés

    def gen():
        printed = {j["seed"]: 0 for j in job_list}
        total = len(job_list)
        while True:
            parts = [f'echo "{SEED_MARK}{j["seed"]}>>>"; '
                     f'cat {cli.REMOTE_ROOT}/{shlex.quote(j["out_rel"])} 2>/dev/null'
                     for j in job_list]
            parts.append(f'echo "{cli.SEP}"; squeue -j {ids} -h -o %i 2>/dev/null')
            try:
                tick = cli._ssh("; ".join(parts), check=False, timeout=60)
            except cli.SSHError as e:
                yield f"data: {json.dumps({'log': f'[erreur SSH: {e}]'})}\n\n"
                time.sleep(cli.POLL_S); continue
            body, _, state = tick.rpartition(cli.SEP)
            state = state.strip()
            logs = {}                                            # seed → log complet
            for seg in body.split(SEED_MARK)[1:]:
                head, _, content = seg.partition(">>>")
                try:
                    logs[int(head)] = content
                except ValueError:
                    continue
            for j in job_list:                                   # nouvelles lignes par graine
                seed = j["seed"]
                full = logs.get(seed, "")
                if len(full) > printed[seed]:
                    new = full[printed[seed]:]
                    printed[seed] = len(full)
                    for line in new.splitlines():
                        yield f"data: {json.dumps({'log': f'[s{seed}] {line}'})}\n\n"
            if not state:                                        # toutes les graines terminées
                per_seed = []
                for j in job_list:
                    p = cli.parse_props(logs.get(j["seed"], ""))
                    if p:
                        p["seed"] = j["seed"]
                        per_seed.append(p)
                if per_seed:
                    agg = cli.aggregate_seeds(per_seed)
                    detail = {p["seed"]: p.get("Tg_pred") for p in per_seed}
                    yield f"data: {json.dumps({'done': True, 'props': agg, 'per_seed_tg': detail})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True, 'props': None})}\n\n"
                break
            running = [x for x in state.split() if x]
            yield f"data: {json.dumps({'seedstate': {'done': total - len(running), 'total': total}})}\n\n"
            time.sleep(cli.POLL_S)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    import argparse
    import socket
    import uvicorn
    ap = argparse.ArgumentParser(prog="tgweb", description="Webapp locale tgcli (calcul sur CRIANN).")
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
        print(f"⚠ tgweb exposé sur le RÉSEAU ({args.host}) — AUCUNE authentification.")
        print(f"   Tes collègues : http://{lan_ip}:{args.port}/  (même réseau + pare-feu ouvert)")
        print(f"   Tout visiteur peut soumettre des jobs sur TON compte CRIANN.")
    else:
        print(f"tgcli webapp → http://127.0.0.1:{args.port}  (local uniquement ; --host 0.0.0.0 pour le LAN)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
