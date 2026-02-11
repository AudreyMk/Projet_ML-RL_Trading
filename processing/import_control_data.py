import pandas as pd

## Chargement des données
def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV brut et retourne un DataFrame.
    """
    df = pd.read_csv(
        file_path,
        header=None,
        names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
        sep=','
    )
    return df

## Etape 1: preprocessing des données
## combinaison date et itme en timestamp

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%Y.%m.%d %H:%M'
    )

    df = df.sort_values('timestamp')
    df = df.drop(columns=['date', 'time', 'volume'])
    df = df.set_index('timestamp')

    return df


## vérification des irrégularités de pas de temps

def check_time_irregularities(df: pd.DataFrame, freq: int = 1) -> pd.DataFrame:

    time_diff = df.index.to_series().diff()
    expected_delta = pd.Timedelta(minutes=freq)
    irregular_steps = df[time_diff != expected_delta]

    return irregular_steps

## vérification des timestamps manquants et valeurs manquantes
def check_missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()

def check_missing_timestamps(df: pd.DataFrame, freq: str = "1min") -> pd.DatetimeIndex:
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )

    missing_timestamps = full_index.difference(df.index)
    return missing_timestamps


# Vérification des incohérences dans les données OHLC
def check_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie la cohérence des bougies OHLC.
    """
    invalid_ohlc = df[
        (df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
        (df['low'] > df[['open', 'close', 'high']].min(axis=1))
    ]

    number_invalid_ohlc = len(invalid_ohlc)

    if len(invalid_ohlc) == 0:
        print(
            "Lignes OHLC invalides = 0\n"
            "- Aucune bougie incohérente\n"
            "- Aucune corruption de prix\n"
            "- Données exploitables pour l’analyse"
        )
    return number_invalid_ohlc, invalid_ohlc
