# Synthese EDA - Decisions et ajustements

## 2022 - Interpretation par graphe
- Prix: non stationnaire (ADF p=0.52), dynamique de tendance probable.
- Volatilite rolling: plus elevee (std 0.000844) avec pics.
- Distribution rendements: asymetrie negative forte (skew -2.30) et queues lourdes (kurtosis 122.6).
- Q-Q plot: forte deviation a la normalite, surtout dans les queues.
- ACF: rendements stationnaires, autocorrelations attendues faibles.
- PACF: structure de court terme possible (1-2 lags a verifier).
- Rendement horaire: meilleur rendement moyen vers 18h (mean 0.000043).
- Volatilite horaire: plus forte a 8h (std 0.001403), plus faible a 16h.

## 2023 - Interpretation par graphe
- Prix: non stationnaire (ADF p=0.28), tendance presente.
- Volatilite rolling: plus faible qu'en 2022 (std 0.000591).
- Distribution rendements: asymetrie faible (skew -0.06) mais queues lourdes (kurtosis 19.5).
- Q-Q plot: deviations a la normalite, moins extremes qu'en 2022.
- ACF: rendements stationnaires, autocorrelations faibles.
- PACF: structure de court terme possible (1-2 lags a verifier).
- Rendement horaire: meilleur rendement moyen vers 12h (mean 0.000043).
- Volatilite horaire: plus forte a 8h (std 0.001048), plus faible a 23h.

## 2024 - Interpretation par graphe
- Prix: non stationnaire (ADF p=0.39), tendance probable.
- Volatilite rolling: plus faible des 3 annees (std 0.000430).
- Distribution rendements: asymetrie negative moderee (skew -0.36) et queues lourdes (kurtosis 23.8).
- Q-Q plot: non-normalite persistante, moins extreme qu'en 2022.
- ACF: rendements stationnaires, autocorrelations faibles.
- PACF: structure de court terme possible (1-2 lags a verifier).
- Rendement horaire: meilleur rendement moyen vers 18h (mean 0.000031).
- Volatilite horaire: plus forte a 8h (std 0.000812), plus faible a 0h.

## Actions proposees pour le feature engineering
- Utiliser des rendements/log-rendements (prix non stationnaires).
- Ajouter des lags de rendements: 1, 2, 3, 5, 10.
- Ajouter des features de volatilite: rolling std (20, 60, 120), range (high-low), ATR.
- Ajouter des features temporelles: heure, jour de semaine, sessions (Asie/Europe/US), indicateur d'heure volatile (8h).
- Gerer les queues lourdes: clipping/winsorisation des rendements extremes, mesures robustes (median, MAD).
- Segmenter par regime de volatilite (low/medium/high) pour adapter les modeles.
