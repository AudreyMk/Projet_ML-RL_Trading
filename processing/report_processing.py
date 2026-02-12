import os
from import_control_data import (
	load_data,
	preprocess_data,
	check_time_irregularities,
	check_missing_values,
	check_missing_timestamps,
	check_invalid_ohlc
)
from report_viz_processing import viz_data_report


def generate_import_report(file_path):
	"""
	Charge le fichier, effectue les vérifications et retourne un rapport sous forme de dictionnaire.
	"""
	report = {}
	# Chargement et prétraitement
	df_raw = load_data(file_path)
	df = preprocess_data(df_raw)

	# Vérifications
	report['nb_lignes'] = len(df)
	report['nb_time_irregularities'] = len(check_time_irregularities(df))
	report['missing_values'] = check_missing_values(df).to_dict()
	report['missing_timestamps'] = len(check_missing_timestamps(df))
	n_invalid_ohlc, invalid_ohlc = check_invalid_ohlc(df)
	report['nb_invalid_ohlc'] = n_invalid_ohlc

	return report, df 


# Boucle pour traiter les 3 années
for year in [2022, 2023, 2024]:
	print(f"\n{'='*60}")
	print(f"TRAITEMENT ANNÉE {year}")
	print(f"{'='*60}")
	
	# Chemin du fichier
	file = os.path.join("data", "raw", f"DAT_MT_GBPUSD_M1_{year}.csv")
	
	# Génération du rapport
	rapport, df = generate_import_report(file)
	
	# Sauvegarde du DataFrame traité
	output = os.path.join("data", "processed", f"DAT_MT_GBPUSD_M1_{year}.csv")
	df.reset_index().to_csv(output, index=False)
	print(f" Fichier sauvegardé : {output}")
	
	# Affichage du rapport
	print(f"\nRAPPORT {year}:")
	print(rapport)
	
	# Génération du rapport de visualisation
	visualization_report = viz_data_report(df)
	print(f"\nVISUALISATION {year}:")
	print(visualization_report)

print(f"\n{'='*60}")
print(" TRAITEMENT TERMINÉ POUR LES 3 ANNÉES")
print(f"{'='*60}")