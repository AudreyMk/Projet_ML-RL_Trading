# Protocole d'evaluation et stress tests RL

## Split temporel obligatoire
- 2022: entrainement
- 2023: validation
- 2024: test final (jamais utilise pour entrainer)

## Walk-forward (optionnel)
- Entrainement: 2022-2023
- Test: 2024

## Metriques principales
- cumulative_profit
- max_drawdown
- sharpe
- profit_factor

## Stress tests
1) Heures volatiles vs calmes
- 8h (heure volatile)
- 16h (heure calme)

2) Regimes de volatilite
- Low: quantile 20% de rolling_std_20
- High: quantile 80% de rolling_std_20

## Regles d'evaluation
- Pas de split aleatoire.
- Normalisation basee sur 2022 uniquement.
- Horizon reward: 1 bougie M15.
- Couts integres: transaction_cost + slippage.
