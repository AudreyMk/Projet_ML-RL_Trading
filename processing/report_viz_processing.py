
import pandas as pd
import matplotlib.pyplot as plt

def viz_data_report(df: pd.DataFrame, year: int = None):
    """
    Génère les visualisations de l'activité de trading à partir d'un DataFrame indexé par timestamp.
    """
    # Nombre de données par jour
    data_per_day = df.groupby(df.index.date).size()

    # Ajouter jour de la semaine
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek  # 0=Lundi, 6=Dimanche
    df['is_weekend'] = df.index.dayofweek >= 5  # 5=Samedi, 6=Dimanche

    # Compter par jour de semaine
    count_by_day = df.groupby('day_of_week').size()

    # Nombre de données par heure
    df['hour'] = df.index.hour
    count_by_hour = df.groupby('hour').size()

    # Identifier les jours fériés (UK et US pour GBP/USD)
    if year is None:
        year = df.index[0].year
    jours_feries = [
        f'{year}-01-01',  # Nouvel An
        f'{year}-03-29',  # Vendredi Saint
        f'{year}-04-01',  # Lundi de Pâques
        f'{year}-05-06',  # May Day
        f'{year}-05-27',  # Spring Bank Holiday
        f'{year}-07-04',  # Independence Day (US)
        f'{year}-08-26',  # Summer Bank Holiday
        f'{year}-12-25',  # Noël
        f'{year}-12-26',  # Boxing Day
    ]
    jours_feries = pd.to_datetime(jours_feries).date

    # Marquer les jours fériés
    df['is_holiday'] = [date in jours_feries for date in df.index.date]

    # Activité par jour de semaine HORS jours fériés
    df_no_holiday = df[~df['is_holiday']]
    count_by_day_no_holiday = df_no_holiday.groupby('day_of_week').size()

    # Activité jours fériés
    data_per_day_holidays = data_per_day[data_per_day.index.isin(jours_feries)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Graphique 1 : Évolution temporelle avec jours fériés
    axes[0,0].plot(data_per_day.index, data_per_day.values/60, linewidth=0.8, label='Tous les jours')
    axes[0,0].scatter(data_per_day_holidays.index, data_per_day_holidays.values/60, color='red', s=30, label='Jours fériés', zorder=5)
    axes[0,0].set_title('Évolution : Nombre d\'heures par jour')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Nombre d\'heures')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Graphique 2 : Activité par jour de semaine GLOBAL
    day_labels = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    axes[0,1].bar(count_by_day.index, count_by_day.values/60, color=['green', 'green', 'green', 'green', 'green', 'red', 'red'])
    axes[0,1].set_title('Activité par jour (GLOBAL)')
    axes[0,1].set_xlabel('Jour')
    axes[0,1].set_ylabel('Nombre d\'heures')
    axes[0,1].set_xticks(range(7))
    axes[0,1].set_xticklabels(day_labels)
    axes[0,1].grid(True, axis='y', alpha=0.3)

    # Graphique 3 : Activité par jour de semaine HORS jours fériés
    axes[1,0].bar(count_by_day_no_holiday.index, count_by_day_no_holiday.values/60, color=['green', 'green', 'green', 'green', 'green', 'red', 'red'])
    axes[1,0].set_title('Activité par jour (HORS jours fériés)')
    axes[1,0].set_xlabel('Jour')
    axes[1,0].set_ylabel('Nombre d\'heures')
    axes[1,0].set_xticks(range(7))
    axes[1,0].set_xticklabels(day_labels)
    axes[1,0].grid(True, axis='y', alpha=0.3)

    # Graphique 4 : Activité par heure
    axes[1,1].bar(count_by_hour.index, count_by_hour.values/60, color='steelblue')
    axes[1,1].set_title('Activité par heure (UTC)')
    axes[1,1].set_xlabel('Heure')
    axes[1,1].set_ylabel('Nombre d\'heures')
    axes[1,1].set_xticks(range(0, 24))
    axes[1,1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'count_by_day_of_week': count_by_day,
        'count_by_hour': count_by_hour,
        'jours_feries': jours_feries
    }