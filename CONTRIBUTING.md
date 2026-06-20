# Contributing

Thanks for your interest! This is a small research tool, kept deliberately simple.

## Project layout

```
src/tg_ml/
  cli.py          # `tgcli` — submit to the cluster, stream logs, parse results
  webapp.py       # `tgweb` — FastAPI front-end over the same backend
  static/index.html   # single-file web UI (Ketcher editor, served locally + i18n FR/EN)
  tg_kinetics.py  # legacy single-window ρ(T) hyperbola / breakpoint fit (the default in-pipeline Tg)
  tg_blind.py     # blind Tg: VK seed → pooled-window density coude + Prigogine–Defay + diffusion-angle confidence
  md_build.py     # build the oligomer + pack the simulation box
scripts/
  pipeline.py     # the MD pipeline itself (one cooling run; runs on the cluster)
  blind_tg.py     # blind-Tg driver — orchestrates multiple windowed runs and pools them via tg_blind.py
  vk_centerer.py  # van-Krevelen group-contribution Tg estimate, used to seed the window (predict via vk_centerer_model.json)
  # compare_exp.py / exp_reference.py — reliability benchmark helpers (research, not the Tg tool)
```

## Dev setup

```bash
uv sync                       # install the local front-end
uv run tgcli --help
uv run tgweb                  # http://127.0.0.1:8000
```

The Ketcher editor bundle is **not** committed (~97 MB). Download it once:

```bash
curl -sL https://github.com/epam/ketcher/releases/download/v3.12.0/ketcher-standalone-3.12.0.zip -o /tmp/k.zip
mkdir -p src/tg_ml/static/ketcher && (cd src/tg_ml/static/ketcher && unzip -q /tmp/k.zip)
```

Running an actual computation needs SSH access to an HPC cluster with an
OpenMM-CUDA + OpenFF conda environment (see the README). You can work on the
front-end (validation, UI, i18n) without a cluster.

## Guidelines

- **Security first.** Any value that can reach an SSH command on the cluster must
  be allow-listed before interpolation (see `webapp.py` and `SECURITY.md`). Do not
  loosen these checks.
- Keep the UI a single self-contained `index.html`; new user-facing strings go in
  the `I18N` object (both `fr` and `en`).
- Match the surrounding style; keep comments meaningful.
- Open an issue before large changes so we can agree on the approach.

## Pull requests

Small, focused PRs with a clear description are easiest to review. Mention how you
tested the change (front-end behaviour, and a real run if you have cluster access).
