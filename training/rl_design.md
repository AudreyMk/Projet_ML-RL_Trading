# Conception Reinforcement Learning (obligatoire)

## 1) Probleme metier
- Objectif: apprendre une politique de trading M15 (GBP/USD) pour maximiser le PnL.
- Contraintes: couts de transaction et slippage, penalite de drawdown, pas de fuite de donnees.
- Horizon: M15, reward base sur future_return a horizon 1.

## 2) Donnees
- Source: fichiers features M15 (2022/2023/2024).
- Qualite: series reguliere M15, verifiee en amont (nettoyage, verifs OHLC).
- Alignement: index temporel, variables derives a partir du passe.
- Couts: transaction_cost + slippage integres au reward.

## 3) State
- Features: toutes les colonnes numeriques sauf future_return et target_direction.
- Normalisation: standardisation basee uniquement sur 2022.
- Warm-up: suppression des lignes NaN issues des rolling features.

## 4) Action
- Discret: 0=flat, 1=long, 2=short.
- Position appliquee a t (reward sur future_return a t+1).

## 5) Reward
- Reward = position * future_return - couts (transaction + slippage) - penalite drawdown.
- Penalite drawdown appliquee uniquement si drawdown negatif.

## 6) Environnement
- Simulateur simple base sur future_return.
- Cout transaction et slippage inclus.
- Anti-leakage: future_return est shift(-1).

## 7) Choix algorithme
- Q-learning discretise (etat continu -> bins par quantiles).
- Justification: simple, interpretable, evite l instabilite sur etat continu.

---

# Parametres cles

## Definition
- State: features normalisees + position.
- Action: discret (flat/long/short).
- Reward: PnL ajuste risque (drawdown, couts).
- Horizon: 1 bougie M15.
- Couts: transaction_cost=0.0001, slippage=0.00005.

## Entrainement
- gamma: 0.99
- learning_rate: 0.01
- epsilon: 0.1
- episodes: 50
- n_bins: 10
- seed: 42 (via numpy si besoin)

## Evaluation
- Split temporel obligatoire:
  - 2022: entrainement
  - 2023: validation
  - 2024: test final
- Walk-forward: entrainement 2022-2023, test 2024.
- Metriques: Sharpe, drawdown, cumulative_profit, profit_factor.
- Stress tests: heures volatiles (8h vs 16h), regimes de volatilite (quantiles 20/80).
