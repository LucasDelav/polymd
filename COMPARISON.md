# polymd vs l'état de l'art (2000 → juin 2026)

Positionnement de **polymd** (pipeline MD mono-GPU : SMILES → propriétés du polymère amorphe en
masse, OpenMM + OpenFF Sage, ~15 min / ~0.3 GPU-h par polymère) face à *tout* le paysage de la
prédiction computationnelle des propriétés de polymères : ML (descripteurs/GNN/transformers),
MD classique (RadonPy/SPACIER, ensemble-MD), DFT-QSPR, et champs de force ML ab-initio (MLIP).

Synthèse appuyée sur une revue multi-agent adversariale (25 affirmations testées, 20 confirmées,
5 réfutées) + recherches ciblées MLIP. Toutes les sources sont primaires (peer-reviewed / dépôts officiels).
Nos chiffres polymd : voir [BENCHMARK.md](BENCHMARK.md) (12 polymères).

---

## 1. Verdict par propriété (qui nous bat, avec le chiffre)

| Propriété | **polymd** | Meilleur concurrent honnête (n, protocole) | Verdict |
|---|---|---|---|
| **Tg** | ~13 K (multi-seed) | Ensemble-MD JCTC 2025 **11–17.7 K** (6 époxydes) ; Polymer Genome GPR **18.8 K** (n=5076, CV 5-fold) ; Lieconv-Tg 24.4 K (n=7166, split aléatoire = fuite légère) | 🟢 **au plancher du réaliste** |
| **Densité** | 2.7 % (~0.03 g/cm³) | claim "ML GPR 0.03 g/cc" **RÉFUTÉE (0-3)** ; RadonPy R²~0.89 | 🟢 **personne ne nous bat clairement** |
| **Indice n** | ~1.3 % (~0.02) | SPACIER/RadonPy MAE **0.02** / R²0.92 (calibré n=26) ; ML Ramprasad RMSE 0.05 | 🟢 **= MD-SOTA, bat le ML** |
| **Cp (cap. calorifique)** | ~5 % (verres) | SPACIER/RadonPy **R²0.61 seulement** (n=72) ; Polymer Genome n=80 (affamé) | 🟢 **on attaque LE point faible MD** |
| **κ (cond. thermique)** | −11 à −21 % (borne basse) | RadonPy R²~0.49–0.61 ; ML MAE 0.02–0.03 W/m/K | 🟡 compétitif, mieux corrélé que RadonPy |
| **Module E/G/K** | élastique seulement (FF-limité) | Polymer Genome GPR **120 MPa** (n=629) ; GCN **R²~0.5** (pire prop.), dégrade >10 GPa | 🟡 **dur pour TOUT le monde** |
| **Diélectrique ε** | faible | Polymer Genome RMSE **0.16** (n=1193 exp) | 🔴 concurrent ML crédible |
| **IP/EA/gap** | xtb 1.5 % | TransPolymer/MMPolymer (mais labels **DFT**, pas exp) | 🟡 semi-empirique correct |
| **CTE** | dédié (PS 209 vs 210 exp), bruité | RadonPy R²~0.18–0.22 (faible partout) | 🟡 = personne n'est fiable |
| **δ / cohésion** | ~11 % | groupe-contribution (van Krevelen) ~5–10 % | 🟡 correct |

**Légende** : 🟢 au/au-dessus du SOTA · 🟡 compétitif ou difficile pour tous · 🔴 un concurrent fait mieux.

---

## 2. Paysage des méthodes — qui joue sur quel terrain

**Découverte structurante** : les grands modèles deep-learning **ne concurrencent presque pas notre
cœur thermophysique**.
- **polyBERT, TransPolymer, MMPolymer** ciblent les propriétés **électroniques/DFT** (gap, dérivés) ;
  ils ne prédisent **pas** Tg/densité/κ/Cp/modules. polyBERT ne fait qu'**égaler** Polymer Genome
  (R²0.80 vs 0.81) — avantage = vitesse (×215), pas précision.
- **RF + Morgan fingerprint bat les GNN sur Tg** (JCIM 2021, R²~0.71) → les descripteurs simples
  restent compétitifs.

**Nos vrais concurrents sur NOS propriétés :**
| Famille | Représentant | Force | Faiblesse |
|---|---|---|---|
| MD classique haut-débit | **RadonPy / SPACIER** (GAFF2/LAMMPS) | n, densité | **×100–300 plus cher** ; Cp R²0.61 |
| Descripteurs + GPR | **Polymer Genome** | diélectrique, module | CV-optimisme ; ne généralise pas hors domaine |
| Ensemble-MD | **JCTC 2025** (Patrone) | Tg 11–17.7 K | = **notre méthode** (valide notre multi-seed) |
| Transformers/GNN | polyBERT, TransPolymer | électronique, vitesse | pas le bulk thermo-méca |

---

## 3. Question MLIP (champs de force ML ab-initio) — peut-on gratter des perfs ?

**Non, pas gratuitement aujourd'hui pour le bulk polymère :**
1. **Coût rédhibitoire** : MACE-OFF = 3×10⁵ pas/jour/A100 = **~100–1000× plus lent** que notre FF →
   notre run 15 min deviendrait **des jours/polymère**. Même avec les accélérations 2025 (cuEquiv ×3 +
   BF16 ×4 ≈ ×12), des heures-jours. **Casse la niche.**
2. **Pas meilleur sur notre faiblesse** : MACE-OFF (petit modèle) **sur-prédit la densité, MAE
   0.23 g/cm³** — pire que nos 0.03 — et rate la cohésion longue-portée (chantier ouvert 2025).
3. **Non validé sur polymère amorphe en masse** : validé sur petites molécules / cristaux / liquides ;
   transférabilité-polymères seulement émergente (arXiv 2509.25022 ; SimPoly 2510.13696).
4. **Où ça pourrait aider** : κ et anharmonique. Le **module reste dur pour tous** (R²0.5).
5. **État mi-2026** : **recherche, pas production** pour le bulk polymère.

→ Coup malin : un chemin MLIP **opt-in ciblé κ sur petite boîte**, jamais le pipeline par défaut.

---

## 4. Vulnérabilités connues & comment les défendre

1. **« Un ML fait Tg sub-10 K »** → **fuite de données** (augmentation SMILES, motifs dupliqués
   train/test ; PI1M n'apporte aucune donnée exp nouvelle). Les Tg honnêtes plafonnent à **13–25 K**.
   Notre 13 K est au plancher.
2. **« Polymer Genome bat le diélectrique/module »** → vrai, mais sur **CV 5-fold** (optimisme), petits
   jeux (module n=629), et **ne généralise pas** aux chimies nouvelles. Nous : zéro entraînement.
3. **« SPACIER plus précis sur n »** → égalité (0.02), mais **calibré in-sample sur 26 polymères** et
   **×100–300** notre temps ; et leur **Cp R²0.61** là où le nôtre est physique à 5 %.
4. **« Et la mécanique de rupture (déchirement) ? »** → **hors de portée de toute MD bulk** (boîte
   périodique = pas de surface libre = pas de fracture fragile). On fait l'**élastique** (E/G/K/ν) ; la
   rupture se prédit expérimentalement ou par modèles dédiés, pas par MD bulk. Limite physique assumée.

---

## 5. La niche imprenable

Aucun outil du paysage ne fait ceci : **un SMILES → ~20 propriétés (thermo + mécanique élastique +
électronique + cohésion/transport/structure) AVEC la mécanique physique, sans aucune donnée
d'entraînement, sur n'importe quelle chimie inédite, sur 1 GPU en ~15 min.**

- Le **ML** ne sort pas de son domaine d'entraînement.
- **RadonPy** fait pareil mais **×100–300 plus lent**.
- La **DFT** ne touche pas le bulk amorphe.
- Les **MLIP** ne sont ni assez rapides ni validés sur bulk polymère.

Notre **Cp physique (DOS quantique)** attaque précisément la propriété où le leader MD est le plus
faible (R²0.61) — différenciateur publiable le plus net.

---

## Sources (toutes primaires)

- Polymer Genome, *J. Appl. Phys.* 2020 — table par propriété : https://pubs.aip.org/aip/jap/article/128/17/171104
- Ensemble-MD Tg (Patrone), *JCTC* 2025 (11–17.7 K, ≥10 répliques, N^−0.5) : https://pubs.acs.org/doi/10.1021/acs.jctc.4c01364
- RadonPy, *npj Comput. Mater.* 2022 (15 props, >1000 polymères) : https://www.nature.com/articles/s41524-022-00906-4
- SPACIER, *npj Comput. Mater.* 2025 (n 0.02/R²0.92, Cp R²0.61) : https://www.nature.com/articles/s41524-024-01492-3
- Lieconv-Tg (24.4 K, n=7166) : https://pmc.ncbi.nlm.nih.gov/articles/PMC10851255/
- polyBERT, *Nat. Commun.* 2023 (R²0.80 ≈ PG 0.81) : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10336012/
- TransPolymer, *npj* 2023 : https://www.nature.com/articles/s41524-023-01016-5
- MMPolymer, arXiv 2024 : https://arxiv.org/abs/2406.04727
- RF+Morgan > GNN sur Tg, *JCIM* 2021 : https://pubs.acs.org/doi/10.1021/acs.jcim.1c01031
- MACE-OFF, *JACS* 2024 (densité 0.23 g/cm³, cohésion 2 kcal/mol) : https://pubs.acs.org/doi/10.1021/jacs.4c07099
- SimPoly (MLIP polymères), arXiv 2025 : https://arxiv.org/abs/2510.13696
- Speeding up MACE (3×10⁵ pas/jour/A100), arXiv 2025 : https://arxiv.org/abs/2510.23621
- PI1M (génératif, 12k PolyInfo) : https://github.com/RUIMINMA1996/PI1M
- NeurIPS Open Polymer 2025 — 1ʳᵉ place : https://github.com/jday96314/NeurIPS-polymer-prediction
- Indice de réfraction ML (Ramprasad), *JAP* 2020 : https://ramprasad.mse.gatech.edu/wp-content/uploads/2020/06/RefractiveIndex-JAP2020.pdf

*Caveats : la plupart des chiffres ML sont en CV 5-fold (optimisme CV), pas test externe ; le comparateur
MD-Tg JCTC est sur époxydes thermodurcis (polymd = 12 thermoplastiques linéaires) ; SPACIER calibré
in-sample sur n=26/72. Le champ MLIP bouge vite (2025-2026) — re-vérifier périodiquement.*
