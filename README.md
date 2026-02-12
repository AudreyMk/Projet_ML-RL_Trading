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
