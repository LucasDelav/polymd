"""Compare les sorties du pipeline aux valeurs expérimentales (exp_reference.EXP).

Usage :
  python scripts/compare_exp.py PS=run_PS.out PMMA=run_PMMA.out ...   # paires nom=fichier
  python scripts/compare_exp.py --dir .tgcli --glob 'val_*'           # auto-détecte
Sort, par propriété : valeur sim, valeur exp, erreur signée %, et un MAE agrégé par propriété
sur tous les polymères fournis. Sert à mesurer la fiabilité RÉELLE et cibler physique/calibration.
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_reference import EXP, PIPELINE_KEY, cte_vol_glass_ppmK

# noms courts → clés EXP (md_build.REFERENCE_POLYMERS)
ALIAS = {"PS": "polystyrene", "PMMA": "PMMA", "PE": "polyethylene", "PP": "polypropylene",
         "PEO": "PEO", "PC": "polycarbonate", "PVAc": "PVAc", "PLA": "PLA", "PIB": "polyisobutylene",
         "PB": "polybutadiene", "PaMS": "PaMS", "PMA": "PMA", "PnBMA": "PnBMA", "PEMA": "PEMA",
         "PI": "polyisoprene"}


def parse_props(path):
    """Extrait le dernier bloc JSON `=== PROPRIÉTÉS ===` d'un .out."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    i = txt.rfind("=== PROPRIÉTÉS ===")
    if i < 0:
        return None
    s = txt.find("{", i); depth = 0
    for j in range(s, len(txt)):
        if txt[j] == "{": depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(txt[s:j + 1])
                except Exception: return None
    return None


def err_pct(sim, exp):
    return None if (exp in (None, 0) or sim is None) else round(100.0 * (sim - exp) / exp, 1)


def main():
    args = sys.argv[1:]
    runs = {}
    if "--dir" in args:
        import glob
        d = args[args.index("--dir") + 1]
        pat = args[args.index("--glob") + 1] if "--glob" in args else "*"
        for f in glob.glob(os.path.join(d, pat + ".out")):
            m = re.search(r"(PS|PMMA|PE|PP|PEO|PC|PVAc|PLA|PIB|PB|PaMS|PMA|PnBMA|PEMA|PI)\b", os.path.basename(f))
            if m: runs[m.group(1)] = f
    else:
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1); runs[k] = v

    props_order = list(PIPELINE_KEY.keys())
    agg = {p: [] for p in props_order}
    for short, path in sorted(runs.items()):
        name = ALIAS.get(short, short)
        ref = EXP.get(name)
        pr = parse_props(path)
        if not ref or not pr:
            print(f"⚠ {short}: ref={'ok' if ref else 'MANQUE'} props={'ok' if pr else 'MANQUE'}"); continue
        print(f"\n=== {short} ({name}) ===")
        print(f"{'prop':<10}{'sim':>10}{'exp':>10}{'err%':>8}  conf")
        for ekey in props_order:
            pkey = PIPELINE_KEY[ekey]; sim = pr.get(pkey); exp = ref.get(ekey)
            if exp is None and sim is None: continue
            e = err_pct(sim, exp)
            if e is not None: agg[ekey].append(abs(e))
            ss = f"{sim:.3g}" if isinstance(sim, (int, float)) else "—"
            es = f"{exp:.3g}" if isinstance(exp, (int, float)) else "—"
            print(f"{ekey:<10}{ss:>10}{es:>10}{(str(e) if e is not None else '—'):>8}  {ref.get('conf','')}")

    print("\n================ MAE % PAR PROPRIÉTÉ (sur tous les polymères) ================")
    print(f"{'prop':<12}{'MAE%':>8}{'n':>5}   biais (signe dominant)")
    for p in props_order:
        v = agg[p]
        if v: print(f"{p:<12}{round(sum(v) / len(v), 1):>8}{len(v):>5}")


if __name__ == "__main__":
    main()
