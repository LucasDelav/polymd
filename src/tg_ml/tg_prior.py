"""Prior Tg rapide (SMILES → Tg estimé) pour centrer la fenêtre de refroidissement du pipeline
MD quand l'utilisateur ne fournit PAS de Tg_exp (prédiction d'un polymère inconnu).

Usage prévu CÔTÉ CLIENT (cli/webapp en local) : on calcule le prior puis on le passe à CRIANN
comme TG_EXP. polyBERT (sentence-transformers) n'est donc PAS requis sur le cluster.

    from tg_ml.tg_prior import tg_prior
    tg_c = tg_prior("*CC(*)c1ccccc1")     # → ~Tg en °C

MAE CV ≈ 58 K sur 511 polymères (NeurIPS 2025) — grossier mais suffisant : la fenêtre de
refroidissement fait ~±100-140 K, et le pipeline recommande de la décaler si le coude touche un bord.
Le modèle est entraîné par tg_ml_academic_archive/tg_prior_train.py.
"""
from __future__ import annotations
import os
from pathlib import Path

_MODEL_PATH = Path(os.environ.get("TG_PRIOR_MODEL",
                                  Path(__file__).resolve().parents[2] / "tg_prior_model.joblib"))
_MODEL_NAME = "xushijie/polyBERT"      # même miroir que l'archive
_bundle = None
_embedder = None


def _to_psmiles(s: str) -> str:
    s = str(s)
    return s if "[*]" in s else s.replace("*", "[*]")


def _load():
    global _bundle, _embedder
    if _bundle is None:
        from joblib import load
        _bundle = load(_MODEL_PATH)
    if _embedder is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _bundle, _embedder


def tg_prior(smiles: str) -> float:
    """Tg estimé en °C depuis le PSMILES. Lève si le modèle/polyBERT sont indisponibles."""
    bundle, emb = _load()
    v = emb.encode([_to_psmiles(smiles)], show_progress_bar=False)
    return float(bundle["model"].predict(v)[0])


def tg_prior_kelvin(smiles: str) -> float:
    return tg_prior(smiles) + 273.15


if __name__ == "__main__":
    import sys
    # polymères connus (Tg exp °C) pour sanity-check : PS~100, PMMA~105, PE~-120, PDMS~-125, PC~150
    tests = sys.argv[1:] or {
        "*CC(*)c1ccccc1": ("PS", 100), "*CC(*)(C)C(=O)OC": ("PMMA", 105),
        "*CC*": ("PE", -120), "*O[Si](C)(C)*": ("PDMS", -125),
    }
    if isinstance(tests, dict):
        for smi, (name, exp) in tests.items():
            print(f"{name:6s} {smi:22s} → Tg_prior {tg_prior(smi):6.0f} °C  (exp ~{exp})")
    else:
        for smi in tests:
            print(f"{smi} → {tg_prior(smi):.0f} °C")
