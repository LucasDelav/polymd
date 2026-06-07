# Security Policy

## Threat model — read this before exposing the web app

`tgweb` is a **thin front-end** that submits Molecular-Dynamics jobs to an HPC
cluster **over your own SSH connection**. It has **no authentication**: anyone who
can reach the port can submit jobs that run **on your cluster account** and consume
your allocation.

By design it binds to **`127.0.0.1` (localhost only)**. You can expose it on a
**trusted LAN** with `tgweb --host 0.0.0.0` — this is intended for a lab setting
where colleagues without a cluster account compute through yours.

**Hard rules:**

- ✅ OK: localhost, or a trusted lab LAN reached by IP (`http://<your-ip>:8000`).
- ❌ Never expose it to the public Internet, never port-forward it, and avoid
  shared/open networks (guest Wi-Fi, campus-wide Wi-Fi) where "same network" can
  mean thousands of machines.

## Hardening already in place

The remote side is reached only through a tightly constrained surface:

- **No shell injection into cluster commands.** Every value that reaches an SSH
  command is allow-listed *before* interpolation: job ids must be integers, output
  paths must match the exact `.tgcli/<name>.out` shape the app itself produces
  (this also blocks path traversal), the SLURM partition is checked against a
  fixed list, and the time limit must match `HH:MM:SS`. `shlex.quote` is applied
  as defense in depth. The monomer SMILES is canonicalised by RDKit and quoted.
- **Anti DNS-rebinding / CSRF.** A middleware rejects any request whose `Host`
  header is not an IP literal or `localhost` (so a malicious site cannot rebind a
  domain to `127.0.0.1` and drive your instance). Reach a LAN instance by **IP**,
  not by hostname.
- **Abuse limits.** Seed count is capped, and SMILES / drawing payloads are size-
  limited to avoid trivial denial of service.

## Residual risk (accepted by design)

After the above, the worst a LAN user can do is submit **legitimate** pipeline
jobs on your account (waste GPU hours) — not run arbitrary commands. If you need
real access control (per-user separation, a shared secret), that is not provided
yet; open an issue if you want to discuss it.

## Reporting a vulnerability

Please open a GitHub issue, or — for anything sensitive — use GitHub's private
"Report a vulnerability" advisory feature on this repository. Include steps to
reproduce. There is no formal SLA; this is a small research tool maintained on a
best-effort basis.
