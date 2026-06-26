# TODO — 3D Point-Cloud Classifier (projet portfolio STEP Consulting)

> Objectif : faire **le meilleur projet du genre** pour l'offre STEP (Data Scientist /
> Ingénieur IA Software, Grenoble). Chaque item est mappé à un mot-clé de l'offre.
> Légende : `[x]` fait · `[ ]` à faire · ⭐ = différenciateur (ce qui te distingue
> des autres candidats).

---

## 0. C'est quoi le projet (à savoir réciter en entretien)

**En une phrase :** classer un objet 3D (meuble, pièce CAO, scan) à partir de sa
**seule géométrie** — pas de métadonnée, juste la forme — via le réseau **PointNet**,
servi par une **API FastAPI** comme en production.

**Le pipeline :**
```
mesh (.off/.ply/.stl/.obj)
  → échantillonnage surfacique (area-weighted) de 1024 points
  → normalisation (centre + sphère unité)  → invariance d'échelle
  → PointNet  → descripteur global 1024-d   → invariance à l'ordre des points
  → tête FC (512→256→10) + softmax
  → catégorie : "chair" 0.99
```

**Pourquoi PointNet (les 2 idées clés à défendre) :**
1. **Agrégation symétrique** : MLP par point puis `max-pool` global. `max` est
   indépendant de l'ordre → invariance aux N! permutations des N points.
   *(prouvé par `tests/test_model.py::test_permutation_invariance`)*
2. **Alignement spatial appris (T-Net)** : 2 sous-réseaux prédisent une transfo
   3×3 (entrée) et 64×64 (features) pour reposer l'objet dans un repère canonique.
   Régularisation d'orthogonalité sur la 64×64 pour pas perdre d'info.

**Pourquoi ce projet pour STEP :** l'offre cite littéralement "données complexes
(3D, documents techniques)" et "interfacer des modèles avec des applications
logicielles". Aucun autre projet (Syro RAG, forecasting, SDWPF) ne touche au 3D.
Celui-là coche LEUR mot-clé.

**Dataset :** ModelNet10 — 10 catégories de meubles CAO, ~3991 train / ~908 test.
Domaine-agnostique : réentraîner sur un catalogue de pièces industrielles = même code.

**Chiffres actuels (à citer) :**
- Accuracy test globale : **0.872** (PointNet from scratch, CPU, 15 epochs).
- Meilleures classes : chair 0.99, sofa 0.97, bed 0.96.
- Pires : desk 0.66, night_stand 0.66 — confusion géométrique attendue
  (desk↔table, night_stand↔dresser : ce sont des boîtes plates/cubiques).

---

## 1. SOCLE — déjà fait (vérifier que ça tourne)

- [x] PointNet fidèle au papier (TNet, encoder, classifier, régulariseur) — `model.py`
- [x] Pipeline data : mesh→cloud, normalize, augment, cache `.npy` — `data.py`
- [x] Boucle d'entraînement + MLflow + eval par classe — `train.py`
- [x] Inférence load-once (`lru_cache`) + top-k — `predict.py`
- [x] API FastAPI `/predict` `/health` `/classes` + validation d'entrée — `api.py`
- [x] Tests (model shapes, invariance permutation, data, API contract) — `tests/`
- [x] Docker + docker-compose + CI GitHub Actions
- [x] **Train complet → modèle sauvegardé, best acc 0.872** ✅
- [x] **Eval → matrice de confusion + acc par classe** ✅ `reports/confusion_matrix.png`
- [x] ⭐ **Benchmark robustesse (bruit / occlusion / rotation SO(3))** — `scripts/robustness.py`

---

## 2. MUST — finir ce week-end (sans ça, projet incomplet)

- [ ] **README — remplir `<!-- RESULTS -->`** avec les vrais chiffres :
      acc globale, tableau acc par classe, lien vers les 2 figures.
- [ ] **README — section "Robustness"** : interpréter les courbes bruit/occlusion +
      le résultat rotation SO(3). → mot-clé STEP "**robustes ... conditions réelles**".
- [ ] **README — "Failure analysis"** : pourquoi desk/night_stand ratent (confusion
      géométrique), lecture de la matrice de confusion. Montre la maturité d'analyse.
- [ ] **Vérifier la démo prédiction de bout en bout** :
      `uvicorn pointcloud_clf.api:app` puis `curl -F "file=@<mesh>.off" .../predict`.
      Screenshot de la réponse JSON pour le README.
- [ ] **`pytest` vert** sur la machine (sans dataset, primitives synthétiques).
- [ ] **`ruff check` + `ruff format`** propres (zéro warning).
- [x] **Smoke de l'API dans les tests** : `tests/test_api.py` POST un mesh box
      et check le schéma (`prediction`, `top_k`). ✅

---

## 3. DIFFÉRENCIATEURS — ce qui te met au-dessus des autres candidats

- [x] ⭐ **Robustesse caractérisée** (fait) — la plupart des repos PointNet
      s'arrêtent à "89% sur test propre". Toi tu montres où ça casse en conditions
      réelles. **C'est l'argument d'entretien n°1.**
- [x] ⭐ **Ablation study** (`scripts/ablation.py`) : full 0.865 · −feat-TNet 0.891
      · −aug 0.889. Finding honnête : sur test propre canonique, retirer ces
      composants *augmente* l'acc — leur valeur est la robustesse, pas le chiffre
      headline. Table + framing dans README. ✅ → "choix méthodologiques".
- [x] ⭐ **Fix de la limite trouvée** : réentraîné avec augmentation **SO(3)
      complète** (`scripts/train_so3.py`). Rotation acc **0.19 → 0.59 (≈3×)**,
      coût = canonical 0.87 → 0.64. Trade-off quantifié, table dans le README.
      → narratif "j'ai trouvé une faiblesse ET je l'ai corrigée". Très senior. ✅
- [x] ⭐ **Benchmark latence/débit** (`scripts/benchmark.py`) : **3.46M params,
      13.9 MB, ~4 ms/sample CPU, ~236–308 samples/s**. Table dans le README.
      → mots-clés STEP "**performance, passage à l'échelle**". ✅
- [x] **Calibration / OOD** (`scripts/ood_demo.py`) : sphere/torus/cone/cylinder
      → softmax sur-confiant (torus→bathtub 0.76, conf moy ~0.55). Note README. ✅
      Montre que tu penses "production / cas limites".

---

## 4. INDUSTRIALISATION — mots-clés "MLOps, API, intégration backend"

- [~] **Tester le build Docker** : BLOQUÉ — Docker Desktop pas lancé sur la machine.
      Dockerfile relu (OK). À vérifier toi-même : démarrer Docker Desktop puis
      `docker build -t pointcloud-clf:test .` → `docker run --rm -p 8000:8000 ...`
      → `curl localhost:8000/health`.
- [ ] **`docker compose up`** : API + serveur MLflow montent ensemble. Documenter.
- [ ] **CI verte** sur GitHub (ruff + pytest). Badge dans le README.
- [x] **Model card** : `models/MODEL_CARD.md` — données, métriques, limites
      (rotation, classes confondues), versions, seed, repro. ✅ → "reproductibilité".
- [ ] **Screenshot MLflow** (courbes loss/acc, params) pour le README. → preuve MLOps.
- [ ] **Démo Streamlit** : upload mesh + viewer 3D + prédiction. Screenshot/gif README.

---

## 5. STRETCH — si tu as encore du temps (bonus, pas obligatoire)

- [x] ⭐ **Export ONNX + inférence onnxruntime** : `scripts/export_onnx.py`,
      parité torch↔ONNX vérifiée (max diff 3.7e-4, argmax identique). ✅
      → "**interfacer des modèles avec des applications logicielles**" —
      exactement leurs mots, rare en portfolio.
- [x] **Optimisation inférence** : ONNX Runtime vs torch CPU mesuré PROPREMENT
      (machine idle) — comparable, ~3 ms torch vs ~4 ms ORT batch=1. Modèle trop
      petit pour que ORT batte l'eager sur CPU. Valeur ONNX = portabilité, pas
      vitesse. Note honnête README. ✅ (1ère mesure faussée par contention CPU.)
- [ ] **Note "pièces industrielles"** : court paragraphe + 1 figure montrant que le
      pipeline marche sur un mesh CAO non-meuble (STL d'une pièce méca). Relie au métier STEP.
- [ ] **Versioning données** : mention DVC ou hash du dataset pour la repro.

---

## 6. Ordre d'exécution recommandé (week-end)

1. **Vendredi soir** : §2 README complet (chiffres + 2 figures + failure analysis) +
   §2 pytest/ruff verts. → projet déjà présentable.
2. **Samedi** : §3 ablation + benchmark latence + le fix rotation SO(3). → différenciateurs.
3. **Dimanche** : §4 industrialisation (Docker test, model card, screenshots MLflow/Streamlit).
4. **Si rab** : §5 ONNX (le plus payant des stretch).

> Priorité absolue si peu de temps : **§2 + le robustness déjà fait + 1 ablation**.
> Ça suffit à défendre un projet sérieux. Le reste = profondeur supplémentaire.

---

## 7. Pitch entretien (3 phrases à mémoriser)

1. *"J'ai implémenté PointNet from scratch pour classer des objets 3D depuis leur
   seule géométrie, avec invariance à l'ordre des points et à l'orientation."*
2. *"Je ne me suis pas arrêté à 87% sur le test propre : j'ai caractérisé la
   robustesse sous bruit capteur, occlusion partielle et rotation arbitraire —
   les conditions d'un vrai scan."*
3. *"Le tout est industrialisé : API FastAPI, tests qui tournent sans le dataset,
   MLflow, Docker, CI — le chemin complet géométrie brute → modèle robuste → API."*
