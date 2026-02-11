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


# Remplace le chemin par celui de ton fichier à tester
file = os.path.join("data", "DAT_MT_GBPUSD_M1_2022.csv")
rapport, df = generate_import_report(file)
df.to_csv(os.path.join("data/processed_data", "DAT_MT_GBPUSD_M1_2022_processed.csv"))  # Sauvegarde du DataFrame traité
print(rapport)

# Génération du rapport de visualisation
visualization_report = viz_data_report(df)
print(visualization_report)
