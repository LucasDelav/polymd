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

Reliability is calibrated against experiment (Tg validated on ~72 polymers,
MAE ≈ 13–15 K). Each value comes with a ±1σ confidence interval.

| Property | Method | Reliability (rel. MAE) |
|---|---|---|
| **n** (refractive index) | Lorentz–Lorenz (Crippen molar refraction + density) | 🟢 **~2 %** |
| **density @300 K** | glassy branch of the ρ(T) fit | 🟢 **~6 %** |
| **Tg** | ρ(T) knee → hyperbola/breakpoint fit → **÷ 1.50** | 🟢 **~6 %** (≈13–18 K) |
| **δ** (solubility parameter) | √(cohesive energy density) **× 1.25** | 🟢 **~10 %** |
| **Cp** (heat capacity) | dU/dT on cooling **÷ 2.27** (classical→quantum) | 🟢 **~13 %** |
| **K** (bulk modulus) | NPT volume fluctuations | 🔴 ~48 % (noisy) |
| **FFV** (free volume) | 1 − 1.3·V_vdw/V_sp | 🔴 ~45 % |
| **Rg, Ree** (chain dimensions) | melt configuration | 🟡 lower bounds |
| **G, E, ν** | shear deformations | ⚠️ experimental, not validated |

### The ÷1.50 correction

MD cools ~10¹¹ K/s vs ~0.1 K/s in the lab, so it over-predicts Tg by a roughly
constant factor. `Tg_exp ≈ Tg_sim / 1.50` is a **universal kinetic correction**
(tested on 14 diverse polymers; no chemical descriptor explains the residual),
not a per-polymer fit. Cp (÷2.27) and δ (×1.25) are similarly physically motivated.

## How it works

```
SMILES ──► validate (RDKit) ──► submit SLURM job over SSH ──► [ HPC GPU ]
                                                                   │
   results table ◄── parse properties ◄── stream logs live ◄──────┘
```

The MD pipeline (`scripts/pipeline.py`) builds an oligomer (RDKit ETKDG + MMFF),
applies the OpenFF Sage force field with NAGL charges, compresses & melts the box,
cools it in steps, fits the density–temperature knee to get Tg, and derives the
other properties.

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
