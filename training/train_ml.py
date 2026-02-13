from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.backtest import run_backtest
from evaluation.metrics import compute_metrics


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def load_features(year):
    """Charge les features d'une année"""
    p = project_root / 'data' / 'features' / f'DAT_MT_GBPUSD_M15_{year}_features.csv'
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp')
    return df


def prepare_xy(df):
    """
    Prépare X (features) et y (target)
    Target: 1 if future_return > 0 else 0
    """
    y = (df['future_return'] > 0).astype(int)
    
    # Features: toutes colonnes numériques sauf future_return et target_direction
    X = df.select_dtypes(include=[np.number]).copy()
    for drop in ['future_return', 'target_direction']:
        if drop in X.columns:
            X = X.drop(columns=[drop])
    
    return X, y


def to_signal_basic(pred):
    """
    Convertit prédiction binaire en signal SANS seuil
    1 → 1 (long), 0 → -1 (short)
    """
    return pd.Series(np.where(pred == 1, 1, -1))


def to_signal_with_threshold(pred_proba, threshold=0.1):
    """
    Convertit probabilités en signaux avec seuil de confiance
    
    Args:
        pred_proba: array (n_samples, 2) - probabilités [classe 0, classe 1]
        threshold: seuil de confiance (ex: 0.1 = 10%)
    
    Returns:
        signals: 1 (long), -1 (short), 0 (hold)
    """
    signals = []
    
    for proba_down, proba_up in pred_proba:
        if proba_up > (0.5 + threshold):
            signals.append(1)   # BUY - confiant hausse
        elif proba_down > (0.5 + threshold):
            signals.append(-1)  # SELL - confiant baisse
        else:
            signals.append(0)   # HOLD - pas assez confiant
    
    return pd.Series(signals)


def align_dropna(X, y, df):
    """
    Supprime les lignes avec NaN et garde l'alignement
    Retourne X, y, df alignés
    """
    valid_mask = ~X.isnull().any(axis=1)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    df_clean = df.loc[X_clean.index]
    
    return X_clean, y_clean, df_clean


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """Pipeline complet """
    
    years = {'train': 2022, 'val': 2023, 'test': 2024}
    
    # ========================================
    # 1. CHARGEMENT DES DONNÉES
    # ========================================
    print(" Chargement des données...")
    df_train = load_features(years['train'])
    df_val = load_features(years['val'])
    df_test = load_features(years['test'])
    
    print(f" Train (2022): {len(df_train):,} lignes")
    print(f" Val   (2023): {len(df_val):,} lignes")
    print(f" Test  (2024): {len(df_test):,} lignes")
    
    # ========================================
    # 2. PRÉPARATION X, y
    # ========================================
    print("\n Préparation des features et target...")
    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)
    
    # Supprimer NaN (warm-up period)
    X_train, y_train, df_train = align_dropna(X_train, y_train, df_train)
    X_val, y_val, df_val = align_dropna(X_val, y_val, df_val)
    X_test, y_test, df_test = align_dropna(X_test, y_test, df_test)
    
    print(f" → X_train: {X_train.shape}")
    print(f" → X_val:   {X_val.shape}")
    print(f" → X_test:  {X_test.shape}")
    
    feature_list = X_train.columns.tolist()
    print(f" → Nombre de features: {len(feature_list)}")
    
    # ========================================
    # 3. ANALYSE DE LA DISTRIBUTION DU TARGET
    # ========================================
    print(f"\n Distribution du target:")
    
    for name, y in [('Train 2022', y_train), ('Val 2023', y_val), ('Test 2024', y_test)]:
        dist = y.value_counts(normalize=True).sort_index()
        print(f" {name}:")
        print(f" Classe 0 (baisse): {dist.get(0, 0):.2%}")
        print(f" Classe 1 (hausse): {dist.get(1, 0):.2%}")
    
    # ========================================
    # 4. STANDARDISATION
    # ========================================
    print("\n Standardisation des features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        index=X_train.index, 
        columns=feature_list
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        index=X_val.index, 
        columns=feature_list
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        index=X_test.index, 
        columns=feature_list
    )
    
    # ========================================
    # 5. MODÈLES
    # ========================================
    models = {
        'logistic': LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'  # Gère le déséquilibre
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=50,
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        )
    }
    
    results = []
    models_dir = project_root / 'models_registry' / 'v1'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Seuils de confiance à tester
    THRESHOLDS = [0.0, 0.05, 0.1, 0.15]
    
    # ========================================
    # 6. ENTRAÎNEMENT ET ÉVALUATION
    # ========================================
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"MODÈLE : {model_name.upper()}")
        print(f"{'='*80}")
        
        # Entraînement
        print(f"\n Entraînement sur 2022...")
        model.fit(X_train_scaled, y_train)
        print(f" Entraînement terminé")
        
        # Évaluation sur VAL et TEST
        for split, (df_split, X_split, y_split) in [
            ('val', (df_val, X_val_scaled, y_val)),
            ('test', (df_test, X_test_scaled, y_test))
        ]:
            print(f"\n{'─'*80}")
            print(f"Évaluation : {split.upper()} ({2023 if split == 'val' else 2024})")
            print(f"{'─'*80}")
            
            # Prédictions
            pred = model.predict(X_split)
            pred_proba = model.predict_proba(X_split)
            
            # ============================================
            # MÉTRIQUES STATISTIQUES (ML)
            # ============================================
            y_true = y_split.loc[X_split.index]
            
            stats_metrics = {
                'accuracy': accuracy_score(y_true, pred),
                'precision': precision_score(y_true, pred, zero_division=0),
                'recall': recall_score(y_true, pred, zero_division=0),
                'f1_score': f1_score(y_true, pred, zero_division=0)
            }
            
            print(f"\n MÉTRIQUES STATISTIQUES (ML)")
            print(f" Accuracy  : {stats_metrics['accuracy']:.4f}")
            print(f" Precision : {stats_metrics['precision']:.4f}")
            print(f" Recall    : {stats_metrics['recall']:.4f}")
            print(f" F1-Score  : {stats_metrics['f1_score']:.4f}")
            
            # Distribution prédictions
            pred_dist = pd.Series(pred).value_counts()
            print(f"\n   Distribution prédictions:")
            print(f"     Classe 0: {pred_dist.get(0, 0):,} ({pred_dist.get(0, 0)/len(pred)*100:.1f}%)")
            print(f"     Classe 1: {pred_dist.get(1, 0):,} ({pred_dist.get(1, 0)/len(pred)*100:.1f}%)")
            
            # ============================================
            # TEST DE DIFFÉRENTS SEUILS DE CONFIANCE
            # ============================================
            print(f"\n MÉTRIQUES FINANCIÈRES (par seuil de confiance)")
            print(f"{'─'*80}")
            print(f"{'Seuil':>8} | {'Trades':>7} | {'Profit':>8} | {'Sharpe':>7} | {'Max DD':>8} | {'PF':>6}")
            print(f"{'─'*80}")
            
            best_sharpe = -999
            best_threshold = 0.0
            best_metrics = None
            
            for threshold in THRESHOLDS:
                # Générer signaux avec seuil
                sig = to_signal_with_threshold(pred_proba, threshold=threshold)
                sig.index = df_split.index
                
                # Backtest
                bt = run_backtest(df_split, sig)
                fin_metrics = compute_metrics(bt)
                
                # Compter les trades (signaux != 0)
                n_trades = len(sig[sig != 0])
                
                # Afficher
                print(f"{threshold:>8.2f} | "
                      f"{n_trades:>7,} | "
                      f"{fin_metrics['cumulative_profit']:>7.2f}% | "
                      f"{fin_metrics['sharpe']:>7.2f} | "
                      f"{fin_metrics['max_drawdown']:>7.2f}% | "
                      f"{fin_metrics['profit_factor']:>6.2f}")
                
                # Garder le meilleur
                if fin_metrics['sharpe'] > best_sharpe:
                    best_sharpe = fin_metrics['sharpe']
                    best_threshold = threshold
                    best_metrics = fin_metrics.copy()
            
            print(f"{'─'*80}")
            print(f" Meilleur seuil: {best_threshold:.2f} (Sharpe: {best_sharpe:.2f})")
            
            # ============================================
            # STOCKER LES RÉSULTATS (meilleur seuil)
            # ============================================
            results.append({
                'model': model_name,
                'split': split,
                'year': 2023 if split == 'val' else 2024,
                'best_threshold': best_threshold,
                # Métriques statistiques
                'accuracy': stats_metrics['accuracy'],
                'precision': stats_metrics['precision'],
                'recall': stats_metrics['recall'],
                'f1_score': stats_metrics['f1_score'],
                # Métriques financières (meilleur seuil)
                'cumulative_profit': best_metrics['cumulative_profit'],
                'max_drawdown': best_metrics['max_drawdown'],
                'sharpe': best_metrics['sharpe'],
                'profit_factor': best_metrics['profit_factor']
            })
        
        # ========================================
        # 7. SAUVEGARDER LE MODÈLE
        # ========================================
        model_path = models_dir / f'{model_name}_v1.pkl'
        joblib.dump(model, model_path)
        print(f"\n Modèle sauvegardé: {model_path}")
    
    # ========================================
    # 8. SAUVEGARDER SCALER ET MÉTADONNÉES
    # ========================================
    print(f"\n{'='*80}")
    print("SAUVEGARDE")
    print(f"{'='*80}")
    
    # Scaler
    scaler_path = models_dir / 'scaler_v1.pkl'
    joblib.dump(scaler, scaler_path)
    print(f" Scaler: {scaler_path}")
    
    # Métadonnées
    metadata = {
        'feature_list': feature_list,
        'n_features': len(feature_list),
        'scaler': 'StandardScaler',
        'model_version': 'v1',
        'train_year': 2022,
        'val_year': 2023,
        'test_year': 2024,
        'threshold_strategy': 'confidence_based',
        'thresholds_tested': THRESHOLDS
    }
    
    metadata_path = models_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f" Metadata: {metadata_path}")
    
    # ========================================
    # 9. SAUVEGARDER CSV RÉSULTATS
    # ========================================
    results_df = pd.DataFrame(results)
    results_csv = project_root / 'data' / 'ml_baseline_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f" Résultats CSV: {results_csv}")
    
    # ========================================
    # 10. AFFICHER TABLEAU RÉCAPITULATIF
    # ========================================
    print(f"\n{'='*80}")
    print("TABLEAU RÉCAPITULATIF - TEST SET (2024)")
    print(f"{'='*80}\n")
    
    # Filtrer test uniquement
    test_results = results_df[results_df['split'] == 'test'].copy()
    
    print(f"{'Modèle':<15} | {'Thresh':>7} | {'Acc':>6} | {'F1':>6} | {'Return':>8} | {'Sharpe':>7} | {'Max DD':>8}")
    print(f"{'─'*80}")
    
    for _, row in test_results.iterrows():
        print(f"{row['model']:<15} | "
              f"{row['best_threshold']:>7.2f} | "
              f"{row['accuracy']:>6.3f} | "
              f"{row['f1_score']:>6.3f} | "
              f"{row['cumulative_profit']:>7.2f}% | "
              f"{row['sharpe']:>7.2f} | "
              f"{row['max_drawdown']:>7.2f}%")
    
if __name__ == '__main__':
    main()