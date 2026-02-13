# Projet fil rouge — Système de décision GBP/USD (M1 → M15)

Ce dépôt implémente un pipeline complet pour construire, évaluer et expérimenter
des stratégies de trading sur la paire GBP/USD : import M1, agrégation M15,
feature engineering, baselines (règles/aléatoire/buy&hold), ML simple et outils
de backtest. Le travail suit strictement le split temporel demandé :
2022 = entraînement, 2023 = validation, 2024 = test final.

**Livrables importants créés**
- `data/baseline.txt` : sortie texte des évaluations baseline (T06).
- `results/transaction_sweep.csv` : balayage des transaction_costs (résumé).
- `data/ml_baseline.csv` : métriques résumé des modèles ML entraînés (T07).
- `training/train_ml.py` : script d'entraînement ML (Logistic + RandomForest).
- `evaluation/evaluate_baseline.py` : évalue les 3 baselines et écrit `data/baseline.txt`.
- `featurePack/generate_feature_pack.py` : génération du pack de features V2 (obligatoire).

**Structure principale**
- `processing/` : import, agrégation M1→M15, nettoyage et rapports qualité.
- `data/` : jeux de données (raw, processed, features) et outputs.
- `featurePack/` : code pour construire le jeu de features V2 (`build_feature_pack_v2`).
- `evaluation/` : backtest, stratégies baseline, métriques et évaluation par année.
- `training/` : scripts d'entraînement ML (T07).
- `results/` : sorties d'expérimentations (CSV, figures).

**Fonctionnalités et conformité au TP**
- Split temporel strict : entraînement 2022, validation 2023, test 2024 (pas de fuite).
- Agrégation M1→M15 : règles appliquées (open = 1ère minute, high = max, low = min, close = dernière minute).
- Features V2 calculées uniquement à partir du passé (pas de leakage). Voir [featurePack/generate_feature_pack.py](featurePack/generate_feature_pack.py).
- Baselines obligatoires implémentées : `random_strategy`, `rule_strategy`, `buy_and_hold` (voir [evaluation/strategies.py](evaluation/strategies.py)).
- Backtest anti‑leakage : `position = signal.shift(1)` dans [evaluation/backtest.py](evaluation/backtest.py).
- Metrices financières : profit cumulé, max drawdown, Sharpe simplifié, profit factor (voir [evaluation/metrics.py](evaluation/metrics.py)).

**Comment exécuter les étapes principales**

- Générer les features M15 depuis les fichiers processed (exécute `build_feature_pack_v2` pour chaque année) :

```powershell
python featurePack/run_feature_pack.py
```

- Évaluer les baselines (affiche et écrit `data/baseline.txt`) :

```powershell
python -m evaluation.evaluate_baseline
```

- Lancer l'entraînement ML (T07) — entraîne et écrit `data/ml_baseline.csv` :

```powershell
python -m training.train_ml
```

Remarques :
- Les scripts sont conçus pour être lancés depuis la racine du projet.
- Pour capturer la sortie texte :

```powershell
python -m evaluation.evaluate_baseline > results/eval_output.txt
```

**Détails techniques / décisions prises**
- `featurePack/build_feature_pack_v2` : calcule returns, EMA, RSI, ATR, MACD, rolling std, wick/body, régime de volatilité, target (`future_return`) et `target_direction`.
- `evaluation/backtest.py` : application de coûts de transaction et calcul de la courbe d'équity.
- `training/train_ml.py` :
  - préparation : sélection features numériques, suppression des warm‑up NaN, `StandardScaler` entraîné sur 2022 et appliqué à val/test;
  - modèles baseline : `LogisticRegression`, `RandomForestClassifier` ; sauvegarde des modèles et du scaler dans `models_registry/v1/` (si présent) et écriture d'un CSV résumé (`data/ml_baseline.csv`).

**État actuel & prochaines étapes recommandées**
- T06 (évaluation baseline) : terminé.
- T07.1 (préparation des données) : terminé.
- T07.2 (hyperparam tuning temporel) : à implémenter — recommandation : `RandomizedSearchCV` avec `TimeSeriesSplit` et scorer financier (Sharpe ou profit cumulé).
- T07.3 (évaluation financière + plots) : à faire — produire PNG dans `results/figures/` (equity + drawdown par stratégie/modèle).
- T07.4 (versioning) : ajouter métadonnées complètes (features list, seed, hyperparams) au moment de sauvegarder un modèle.
- T07.5 (reproductibilité + tests) : ajouter CLI/Makefile et un smoke test end‑to‑end.

**Conventions Git (obligatoire pour le cours)**
- Branche par tâche : `<prenomnom>__<Txx>__<mot-cle>` (ex. `aya__T01__import_m1`).
- Commits réguliers avec préfixe `[Txx]` (ex. `[T02] add: aggregation M1->M15`).

**Dépendances**
- Voir [requirements.txt](requirements.txt) pour la liste complète. Installez avec :

```powershell
pip install -r requirements.txt
```

**Où regarder les résultats**
- Fichiers texte : `data/baseline.txt` (baselines), `data/ml_baseline.csv` (résumé ML).
- Expérimentations : `results/transaction_sweep.csv`.
- Graphiques (à générer) : `results/figures/`.

Si vous voulez, je peux maintenant :
- lancer le tuning temporel automatique (RandomizedSearchCV) pour T07.2, ou
- générer et sauvegarder les equity curves PNG pour T07.3, ou
- ajouter CLI et tests pour T07.5.

---
_Fichier mis à jour automatiquement pour refléter l'état actuel du projet (février 2026)._ 
# Projet M2 - Trading M1/M15

Ce projet traite des donnees de marche GBP/USD en M1 et genere des bougies M15 avec des controles de qualite. Il fournit des scripts pour importer, pretraiter, resampler, nettoyer et produire un rapport texte.

## Structure
- `processing/import_control_data.py` : import et verifications de base (timestamps, valeurs manquantes, OHLC).
- `processing/processed_data_M15.py` : resampling M15 et nettoyage.
- `processing/quality_report.py` : generation d un rapport qualite et export CSV.
- `processing/report_viz_processing.py` : visualisation (fonction de graphiques).

## Utilisation rapide
1) Generer le M15 et le rapport qualite :
```
python processing/quality_report.py
```

2) Visualisations (depuis un script ou un notebook) :
```python
from processing.report_viz_processing import plot_data_overview
# df doit etre un DataFrame indexe par timestamp
plot_data_overview(df)
```

## Donnees
- Entree principale : `data/processed_data/DAT_MT_GBPUSD_M1_2022_processed.csv`
- Sorties :
  - `data/processed_data/DAT_MT_GBPUSD_M15_2022_clean.csv`
  - `data/processed_data/DAT_MT_GBPUSD_M15_2022_quality_report.txt`

## Notes
- Les fichiers `.csv` peuvent etre ignores par Git selon le `.gitignore`.
- Le rapport qualite est un texte simple, facile a partager.
