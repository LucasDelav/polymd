# tg-md — polymer properties from a monomer SMILES, by Molecular Dynamics

Draw a polymer repeat unit (or type its PSMILES), and get a physics-based
prediction of its **glass-transition temperature (Tg)**, **density**, refractive
index, heat capacity and more — computed by **all-atom Molecular Dynamics**, not a
fitted ML model.

The heavy compute (~48k-atom MD on a GPU, ~15–18 min per polymer) runs on an **HPC
cluster**; your laptop only runs a small front-end that submits the job, streams
the logs live, and shows the results. There are two front-ends over the same
backend:

- **`tgcli`** — a polished command-line interface.
- **`tgweb`** — a local web app with a molecule editor (draw the unit), live logs,
  tooltips, English/French UI, and multi-seed averaging.

> Why MD and not ML? An ML model only knows the polymers it was trained on. MD
> computes the property from physics, so it extrapolates to monomers no model has
> seen — at the cost of GPU time.

## What it predicts

Reliability is calibrated against experiment. **Tg is validated blind** on 100
polymers (49 sugar-derived + 51 classic PS/PMMA/PET-family): **median error ≈ 16 K**,
and ≈ 13 K on the polymers the force field handles well. Each value comes with a
confidence tier.

| Property | Method | Reliability (rel. MAE) |
|---|---|---|
| **n** (refractive index) | Lorentz–Lorenz (Crippen molar refraction + density) | 🟢 **~2 %** |
| **density @300 K** | glassy branch of the ρ(T) fit | 🟢 **~6 %** |
| **Tg** | blind: density coude + Prigogine–Defay, confidence from the diffusion-coude angle → **÷ 1.50** | 🟢 **median ≈ 16 K** (≈13 K FF-tractable) |
| **δ** (solubility parameter) | √(cohesive energy density) **× 1.25** | 🟢 **~10 %** |
| **Cp** (heat capacity) | dU/dT on cooling **÷ 2.27** (classical→quantum) | 🟢 **~13 %** |
| **K** (bulk modulus) | NPT volume fluctuations | 🔴 ~48 % (noisy) |
| **FFV** (free volume) | 1 − 1.3·V_vdw/V_sp | 🔴 ~45 % |
| **Rg, Ree** (chain dimensions) | melt configuration | 🟡 lower bounds |
| **G, E, ν** | shear deformations | ⚠️ experimental, not validated |

### Blind Tg prediction

The project's most accurate Tg path uses **no experimental Tg at all** — it finds the
transition on its own. It runs as a separate **multi-window orchestrator**,
`scripts/blind_tg.py` (the recipe lives in `src/tg_ml/tg_blind.py`), which submits several
windowed cooling runs and pools them. (A plain `tgcli run`/`tgweb` submits a *single*
cooling run and reports a quicker in-run Tg estimate from the one-window ρ(T) hyperbola fit;
you give it a rough Tg with `-t`/`-r` to centre the window.) The blind driver:

1. **Seed** the window with a van-Krevelen group-contribution estimate (`scripts/vk_centerer.py`).
2. **Bracket** the transition with 3 cooling windows (centre ± 150 K) and pool all per-step points.
3. **Value** from the **density coude**: fit the glassy and rubbery asymptotes of ρ(T) and
   intersect them — a window-independent *two-tangent* construction, robust to noise because it
   averages over many points rather than hunting one onset. Then nudge toward the caloric (enthalpy)
   Tg by a single universal **Prigogine–Defay** term, `ρ + 0.20·(U − ρ)`, which corrects part of the
   per-polymer kinetic spread (the volume and enthalpy responses decouple with fragility).
4. **Confidence** from the **diffusion-coude angle**: the more orthogonal the glassy (D ≈ 0) and melt
   branches of D(T), the sharper the real transition and the more trustworthy the value.
5. **Converge by adding coverage** (not seeds — the angle is seed-stable): if the angle is below 48°,
   add two windows at ± 350 K and re-test; if it still doesn't sharpen, flag the result *to verify*.
6. **Kinetic correction**: `Tg_exp ≈ Tg_sim / 1.50`.

MD cools ~10¹¹ K/s vs ~0.1 K/s in the lab, so it over-predicts Tg by a roughly constant factor.
`÷ 1.50` is a **universal kinetic correction** (tested on 14 diverse polymers; no chemical descriptor
explains the residual), kept as-is rather than re-fit per family. The residual error is dominated by
force-field under-cohesion of strong H-bonds (nylons, polyols), which systematically lowers their
predicted Tg. Cp (÷2.27) and δ (×1.25) are similarly physically motivated.

## How it works

```
SMILES ──► validate (RDKit) ──► submit SLURM job over SSH ──► [ HPC GPU ]
                                                                   │
   results table ◄── parse properties ◄── stream logs live ◄──────┘
```

Each MD run (`scripts/pipeline.py`) builds an oligomer (RDKit ETKDG + MMFF), applies the
OpenFF Sage force field with NAGL charges, compresses & melts the box, cools it in steps
recording **ρ(T), enthalpy U(T), cage mobility ⟨u²⟩(T) and diffusion D(T) per step**, and
derives all the properties plus a quick in-run Tg estimate. The blind Tg driver above
(`blind_tg.py`) runs several such windows and pools their curves (`tg_blind.py`) for the
authoritative Tg; the other properties come from a single cooling run.

## Install (front-end)

Requires Python ≥ 3.12. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run tgcli --help
```

The web app embeds the [Ketcher](https://github.com/epam/ketcher) editor (Apache-2.0),
whose ~97 MB build is **not** committed. Download it once:

```bash
curl -sL https://github.com/epam/ketcher/releases/download/v3.12.0/ketcher-standalone-3.12.0.zip -o /tmp/k.zip
mkdir -p src/tg_ml/static/ketcher && (cd src/tg_ml/static/ketcher && unzip -q /tmp/k.zip)
```

### Cluster side (one-time)

You need SSH access to an HPC cluster with a GPU and a conda environment providing
**OpenMM (CUDA build), OpenFF Toolkit, ASE, RDKit, NumPy, SciPy**. The front-end
expects an SSH host alias (default `criann`) and pushes `scripts/pipeline.py` +
`src/tg_ml/` to `~/tg_ml` on the cluster. Adjust `HOST`, `REMOTE_ROOT` and
`REMOTE_PY` near the top of `src/tg_ml/cli.py` for your own cluster.

## Usage

### CLI

```bash
uv run tgcli check                                  # diagnose the cluster connection
uv run tgcli run -s '*CC(*)c1ccccc1' -t 373         # polystyrene, Tg estimate 373 K
uv run tgcli run -s '*CC*' -r 350-420 --seeds 3     # PE, Tg range, 3 seeds (mean ± SE)
uv run tgcli status                                 # SLURM queue
uv run tgcli attach <JOBID>                         # re-attach to a detached run
```

### Web app

```bash
uv run tgweb                    # → http://127.0.0.1:8000  (localhost only)
```

Draw the repeat unit (mark the two attachment points `*`), tweak parameters in the
sidebar (each has a hover tooltip), hit **Launch**, and the main panel switches to
live logs then a results table. The UI follows your browser language (FR/EN, with a
manual switch). A monomer is just a SMILES with **exactly two `*`** — e.g.
polyethylene is `*CC*`, polystyrene `*CC(*)c1ccccc1`.

## Sharing on a LAN — read SECURITY.md first

`tgweb` binds to `127.0.0.1` by default. You can expose it to a **trusted lab LAN**:

```bash
uv run tgweb --host 0.0.0.0     # colleagues reach http://<your-ip>:8000
```

This intentionally lets colleagues **without a cluster account** compute through
yours. There is **no authentication**: anyone reaching the port submits jobs on
your account. Input that reaches the cluster is strictly allow-listed (no shell
injection / path traversal), and a Host-header guard blocks DNS-rebinding — but you
must still **never expose it to the public Internet**. See [SECURITY.md](SECURITY.md).

## Limitations

- `G, E, ν` (shear/Young moduli) are experimental and noisy (`--shear`).
- Atoms limited to C/H/O/N (OpenFF Sage); no Si or metals.
- The ÷1.50 correction assumes ~atactic tacticity.
- ~48k atoms is a statistical floor: smaller boxes drown the Tg signal in noise.

## License

[MIT](LICENSE).

## Acknowledgements

Built on [RDKit](https://www.rdkit.org/), [OpenMM](https://openmm.org/),
[OpenFF](https://openforcefield.org/), and the [Ketcher](https://github.com/epam/ketcher)
molecule editor. Method inspired by ensemble-MD Tg work (Patrone et al.; Afzal 2021;
Soldera 2006).
