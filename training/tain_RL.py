import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random


# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models_registry" / "rl"
REPORTS_DIR = PROJECT_ROOT / "data" / "rl_reports"

# Features sélectionnées (ÉTAPE 3 : STATE)
FEATURES = [
    'return_1',          # Momentum immédiat
    'return_5',          # Momentum court terme
    'ema_diff',          # Tendance (ema_20 - ema_50)
    'rsi_14',            # Surachat/survente
    'rolling_std_20',    # Volatilité récente
    'atr_14',            # Volatilité normalisée (ATR)
    'hour',              # Session de trading (0-23)
    'regime_volatility'  # Régime marché (low/high)
]

# Hyperparamètres RL (ÉTAPE 6 : ALGORITHME)
CONFIG = {
    # Q-Learning
    'gamma': 0.95,              # Discount factor (horizon ~4 steps = 1h)
    'learning_rate': 0.01,      # Vitesse apprentissage
    'epsilon_start': 0.2,       # Exploration initiale (20%)
    'epsilon_end': 0.01,        # Exploration finale (1%)
    'epsilon_decay': 0.995,     # Décroissance exponentielle
    'episodes': 200,            # Nombre d'épisodes d'entraînement
    
    # Discrétisation (ÉTAPE 3)
    'n_bins': 5,                # 5 bins par feature (quintiles)
    
    # Reward (ÉTAPE 5)
    'reward_scale': 100,        # Amplification du signal
    'transaction_cost': 0.00015,  # 1 pip spread + 0.5 pip slippage
    'drawdown_penalty': 0.02,   # Pénalité drawdown
    
    # Environnement
    'warm_up': 200,             # Lignes à supprimer (EMA_200, rolling_std_120)
    'max_drawdown_stop': -0.25, # Stop si DD < -10% # ANCIEN -0.10, changé pour -0.25 pour laisser plus de liberté à l'agent
    
    # Reproductibilité
    'seed': 42
}

# Actions possibles (ÉTAPE 4)
ACTIONS = {
    0: 'FLAT',   # Position neutre
    1: 'LONG',   # Position acheteuse
    2: 'SHORT'   # Position vendeuse
}
N_ACTIONS = len(ACTIONS)

print("✓ Configuration chargée")
print(f"  - Features: {len(FEATURES)}")
print(f"  - Actions: {N_ACTIONS}")
print(f"  - Episodes: {CONFIG['episodes']}")
print(f"  - Reward scale: {CONFIG['reward_scale']}")


class DataPreparer:
    """
    Classe pour charger, normaliser et discrétiser les données
    
    Responsabilités:
      - Chargement des 3 années (2022/2023/2024)
      - Suppression warm-up
      - Normalisation (StandardScaler sur train)
      - Discrétisation en bins (quantiles)
      - Retour train/val/test prêts pour RL
    """
    
    def __init__(self, features, warm_up=200, n_bins=5, seed=42):
        """
        Args:
            features: Liste des features à utiliser
            warm_up: Nombre de lignes à supprimer
            n_bins: Nombre de bins pour discrétisation
            seed: Seed pour reproductibilité
        """
        self.features = features
        self.warm_up = warm_up
        self.n_bins = n_bins
        self.seed = seed
        
        # Seront initialisés lors de get_train_val_test()
        self.scaler = None
        self.bins_edges = None
        
        np.random.seed(seed)
        random.seed(seed)
    
    
    def load_data(self, year):

        filepath = DATA_DIR / f"DAT_MT_GBPUSD_M15_{year}_features.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier introuvable: {filepath}")
        
        # Charge
        df = pd.read_csv(filepath)
        print(f"  Chargé {year}: {len(df)} lignes")
        
        # Supprime warm-up (ÉTAPE 2 : 200 lignes pour EMA_200)
        df = df.iloc[self.warm_up:].copy()
        print(f"    Après warm-up: {len(df)} lignes")
        
        # Vérifie features
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Features manquantes: {missing}")
        
        if 'future_return' not in df.columns:
            raise ValueError("Colonne 'future_return' manquante!")
        
        # Sélectionne uniquement ce qu'on utilise
        cols = self.features + ['future_return']
        df = df[cols].copy()
        
        # Supprime NaN (sécurité)
        if df.isnull().any().any():
            print(f"    ⚠️ NaN détectés, suppression...")
            df = df.dropna()
            print(f"    Après dropna: {len(df)} lignes")
        
        if 'regime_volatility' in df.columns:

            mapping = {
                'low': 0,
                'medium': 1,
                'high': 2
            }

            # Nettoyage éventuels espaces / majuscules
            df['regime_volatility'] = (
                df['regime_volatility']
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )

            # Vérification valeurs inconnues
            if df['regime_volatility'].isnull().any():
                unknown_values = df[df['regime_volatility'].isnull()]
                raise ValueError(
                    f"Valeurs inconnues dans regime_volatility: "
                    f"{unknown_values['regime_volatility'].unique()}"
                )

            df['regime_volatility'] = df['regime_volatility'].astype(int)
        
        return df
    
    
    def _normalize(self, df_train, df_val, df_test):
        self.scaler = StandardScaler()
        
        self.categorical_features = ['regime_volatility']
        self.numeric_features = [f for f in self.features if f not in self.categorical_features]

        # Fit sur train uniquement (ÉTAPE 2)
        self.scaler.fit(df_train[self.numeric_features])
        
        # Transform train
        df_train_norm = df_train.copy()
        df_train_norm[self.numeric_features] = self.scaler.transform(df_train[self.numeric_features])
        
        # Transform val (utilise stats de train!)
        df_val_norm = df_val.copy()
        df_val_norm[self.numeric_features] = self.scaler.transform(df_val[self.numeric_features])
        
        # Transform test (utilise stats de train!)
        df_test_norm = df_test.copy()
        df_test_norm[self.numeric_features] = self.scaler.transform(df_test[self.numeric_features])
        
        print(f"  ✓ Normalisation effectuée")
        print(f"    Train mean: {df_train_norm[self.numeric_features].mean().mean():.6f}")
        print(f"    Train std: {df_train_norm[self.numeric_features].std().mean():.6f}")
        
        return df_train_norm, df_val_norm, df_test_norm
    
    
    def _discretize(self, df_train, df_val, df_test):
        # Calcule les bins_edges sur train uniquement (quantiles)
        self.bins_edges = {}
        
        for feature in self.features:
            # Calcule quantiles sur train
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bins = np.quantile(df_train[feature], quantiles)
            
            # Gère cas où bins identiques (feature constante)
            bins = np.unique(bins)
            if len(bins) < 2:
                # Feature constante, on crée bins artificiels
                bins = np.array([-np.inf, 0, np.inf])
            
            # Ajoute -inf et +inf aux extrémités pour gérer outliers
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            self.bins_edges[feature] = bins
        
        print(f"  ✓ Bins calculés ({self.n_bins} bins par feature)")
        
        # Discrétise train
        df_train_disc = self._apply_discretization(df_train)
        
        # Discrétise val (avec bins de train!)
        df_val_disc = self._apply_discretization(df_val)
        
        # Discrétise test (avec bins de train!)
        df_test_disc = self._apply_discretization(df_test)
        
        return df_train_disc, df_val_disc, df_test_disc
    
    
    def _apply_discretization(self, df):
        df_disc = df.copy()
        
        for feature in self.features:
            bins = self.bins_edges[feature]
            
            # np.digitize : trouve l'indice du bin
            # right=False : bins[i-1] <= x < bins[i]
            indices = np.digitize(df[feature], bins[1:-1], right=False)
            
            # Clip pour être sûr que c'est dans [0, n_bins-1]
            indices = np.clip(indices, 0, self.n_bins - 1)
            
            df_disc[feature] = indices.astype(int)
        
        # future_return reste continu (nécessaire pour reward)
        df_disc['future_return'] = df['future_return']
        
        return df_disc
    
    
    def get_train_val_test(self):

        print("\n" + "="*60)
        print("PRÉPARATION DONNÉES POUR RL")
        print("="*60)
        
        # 1. Chargement
        print("\n[1/3] Chargement...")
        df_train = self.load_data(2022)
        df_val = self.load_data(2023)
        df_test = self.load_data(2024)
        print(df_train.dtypes)
        print(f"  - Train: {df_train.dtypes} lignes")
        print(f"  - Val:   {df_val.dtypes} lignes")   
        print(f"  - Test:  {df_test.dtypes} lignes")    
        
        # 2. Normalisation
        print("\n[2/3] Normalisation...")
        df_train_norm, df_val_norm, df_test_norm = self._normalize(
            df_train, df_val, df_test
        )
        
        # 3. Discrétisation
        print("\n[3/3] Discrétisation...")
        df_train_disc, df_val_disc, df_test_disc = self._discretize(
            df_train_norm, df_val_norm, df_test_norm
        )
        
        # Metadata pour sauvegarde
        metadata = {
            'features': self.features,
            'n_bins': self.n_bins,
            'warm_up': self.warm_up,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_std': self.scaler.scale_.tolist(),
            'bins_edges': {k: v.tolist() for k, v in self.bins_edges.items()},
            'train_samples': len(df_train_disc),
            'val_samples': len(df_val_disc),
            'test_samples': len(df_test_disc)
        }
        
        print("\n✓ Préparation terminée")
        print(f"  Train: {len(df_train_disc)} samples")
        print(f"  Val:   {len(df_val_disc)} samples")
        print(f"  Test:  {len(df_test_disc)} samples")
        print(f"  Espace états: {self.n_bins}^{len(self.features)} × 3 positions")
        print(f"                = {self.n_bins**len(self.features) * 3:,} états")
        
        return df_train_disc, df_val_disc, df_test_disc, metadata


# Test de la classe
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST CLASSE DataPreparer")
    print("="*60)
    
    # Initialise
    preparer = DataPreparer(
        features=FEATURES,
        warm_up=CONFIG['warm_up'],
        n_bins=CONFIG['n_bins'],
        seed=CONFIG['seed']
    )
    
    # Prépare tout
    train, val, test, meta = preparer.get_train_val_test()
    
    print("\n✓ Données discrétisées sample:")
    print(train[FEATURES].head(5))
    print("\nValeurs uniques par feature:")
    for feat in FEATURES:
        print(f"  {feat}: {sorted(train[feat].unique())}")





############## CLASS ENVIRONNEMEN ########################
############################################################

# ════════════════════════════════════════════════════════════════
# BLOC 5 : CLASSE TradingEnvironment (ENVIRONNEMENT DE TRADING)
# ════════════════════════════════════════════════════════════════

class TradingEnvironment:
    """
    Environnement de trading pour RL
    
    Responsabilités:
      - Simuler le marché (état, actions, rewards)
      - Gérer position, equity, drawdown
      - Calculer reward selon formule ÉTAPE 5
      - Détecter fin d'épisode (done)
    
    Actions possibles:
      0 = FLAT  (position = 0)
      1 = LONG  (position = +1)
      2 = SHORT (position = -1)
    """
    
    def __init__(self, 
                 data, 
                 features,
                 reward_scale=100,
                 transaction_cost=0.00015,
                 drawdown_penalty=0.01): # ancien 0.02, réduit pour laisser plus de liberté à l'agent
                 #max_drawdown_stop=-0.10):
        """
        Initialise l'environnement
        
        Args:
            data: DataFrame avec features discrétisées + future_return
            features: Liste des noms de features (8 features)
            reward_scale: Facteur amplification reward (ÉTAPE 5)
            transaction_cost: Coût par trade (ÉTAPE 2)
            drawdown_penalty: Pénalité drawdown (ÉTAPE 5)
            max_drawdown_stop: Stop si DD atteint ce seuil
        """
        # Données
        self.data = data.reset_index(drop=True)  # Reset index pour accès par iloc
        self.features = features
        self.n_steps = len(data)
        
        # Paramètres reward (ÉTAPE 5)
        self.reward_scale = reward_scale
        self.transaction_cost = transaction_cost
        self.drawdown_penalty = drawdown_penalty
        #self.max_drawdown_stop = max_drawdown_stop
        
        # Variables d'état (seront initialisées dans reset())
        self.current_step = 0
        self.position = 0          # Position actuelle (-1, 0, 1)
        self.equity = 0.0          # Equity cumulée
        self.equity_curve = []     # Historique equity
        self.peak_equity = 0.0     # Pic d'equity (pour DD)
        self.current_drawdown = 0.0  # Drawdown actuel
        
        print(f"  ✓ Environnement initialisé")
        print(f"    Steps: {self.n_steps}")
        print(f"    Reward scale: {self.reward_scale}")
        print(f"    Transaction cost: {self.transaction_cost}")
        print(f"    Drawdown penalty: {self.drawdown_penalty}")
        #print(f"    Max drawdown stop: {self.max_drawdown_stop}")


    def reset(self):
        # Remet à zéro
        self.current_step = 0
        self.position = 0  # Commence FLAT (neutre)
        self.equity = 0.0
        self.equity_curve = [0.0]  # Initialise avec equity=0
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
        
        # Retourne l'état initial
        state = self._get_state()
        
        return state
    
    def _get_state(self):
        # Récupère les features à l'instant current_step
        row = self.data.iloc[self.current_step]
        
        # Extrait les valeurs discrétisées (déjà des int 0-4)
        feature_values = tuple(int(row[f]) for f in self.features)
        
        # Ajoute la position actuelle (0, 1 ou -1)
        # Important: position fait partie de l'état!
        # Sinon l'agent ne sait pas s'il est déjà positionné
        state = feature_values + (self.position,)
        
        return state


    def step(self, action):
        """
        Exécute une action dans l'environnement (CŒUR DU RL!)
        
        Workflow:
          1. Applique action → nouvelle position
          2. Calcule profit = position × future_return
          3. Calcule coûts transaction
          4. Calcule reward (FORMULE ÉTAPE 5)
          5. Met à jour equity, drawdown
          6. Passe à t+1
          7. Vérifie si done
        
        Args:
            action: int (0=FLAT, 1=LONG, 2=SHORT)
        
        Returns:
            next_state: tuple (état suivant)
            reward: float (récompense)
            done: bool (épisode terminé?)
        """
        # 1. APPLIQUE L'ACTION (change position)
        prev_position = self.position
        
        if action == 0:
            self.position = 0   # FLAT
        elif action == 1:
            self.position = 1   # LONG
        elif action == 2:
            self.position = -1  # SHORT
        else:
            raise ValueError(f"Action invalide: {action}. Doit être 0, 1 ou 2")
        
        # 2. RÉCUPÈRE FUTURE_RETURN (notre target)
        # C'est le mouvement du marché au prochain timestep
        future_return = self.data.iloc[self.current_step]['future_return']
        
        # 3. CALCULE PROFIT BRUT
        # profit = position × mouvement_marché
        # Si LONG (+1) et marché monte (+0.0003) → profit = +0.0003
        # Si SHORT (-1) et marché monte (+0.0003) → profit = -0.0003
        profit = self.position * future_return
        
        # 4. CALCULE COÛTS TRANSACTION
        # Coût appliqué uniquement si changement de position
        position_change = abs(self.position - prev_position)
        transaction_cost = position_change * self.transaction_cost
        
        # 5. CALCULE REWARD (FORMULE ÉTAPE 5)
        reward = self._calculate_reward(profit, transaction_cost)
        
        # 6. MET À JOUR EQUITY
        # Equity = profit - coûts
        net_profit = profit - transaction_cost
        self.equity += net_profit
        self.equity_curve.append(self.equity)
        
        # 7. MET À JOUR DRAWDOWN
        self._update_drawdown()
        
        # 8. PASSE AU TIMESTEP SUIVANT
        self.current_step += 1
        
        # 9. VÉRIFIE SI ÉPISODE TERMINÉ
        done = False
        
        # Done si on a parcouru toutes les données
        if self.current_step >= self.n_steps:
            done = True
        
        # # Done si drawdown trop important (stop loss)
        # if self.current_drawdown <= self.max_drawdown_stop:
        #     done = True
        #     # Pénalité supplémentaire pour avoir ruiné le compte
        #     reward -= abs(self.max_drawdown_stop) * self.reward_scale
        
        # 10. RÉCUPÈRE ÉTAT SUIVANT
        if not done:
            next_state = self._get_state()
        else:
            # Si done, état suivant n'a pas d'importance
            next_state = None
        
        return next_state, reward, done
    # next_state: tuple(9 valeurs) ou None si done
    # reward: float (typiquement entre -0.1 et +0.1 avec scale=100)
    # done: bool (True si fin épisode)

    def _calculate_reward(self, profit, transaction_cost):
        # Composante 1 : Profit brut
        profit_component = profit
        
        # Composante 2 : Coût transaction
        cost_component = transaction_cost
        
        # Composante 3 : Pénalité drawdown
        # Pénalise uniquement si drawdown négatif
        # Plus le DD est fort, plus la pénalité est grande
        drawdown_component = self.drawdown_penalty * max(0, -self.current_drawdown)
        
        # FORMULE FINALE (ÉTAPE 5)
        reward_raw = profit_component - cost_component - drawdown_component
        
        # Amplification par reward_scale (100)
        # Sans ça, reward typique = 0.0003 (trop faible!)
        # Avec ça, reward typique = 0.03 (signal clair)
        reward = self.reward_scale * reward_raw
        
        return reward
    
    def _update_drawdown(self):
        # Si equity dépasse peak → nouveau record
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0.0
        else:
            # Sinon, calcule perte relative depuis peak
            if self.peak_equity > 0:
                self.current_drawdown = (self.equity - self.peak_equity) / self.peak_equity
            else:
                # Cas initial où peak_equity=0
                self.current_drawdown = 0.0






############ class AgentRL (ALGORITHME RL) ########################
############################################################

# ════════════════════════════════════════════════════════════════
# BLOC 5-6-7-8 : CLASSE QLearningAgent (AGENT RL)
# ════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Agent Q-Learning pour trading
    
    Responsabilités:
      - Maintenir Q-table : Q[state, action] = valeur espérée
      - Choisir actions (ε-greedy)
      - Apprendre de l'expérience (équation de Bellman)
      - Sauvegarder/charger modèle
    
    Q-table format:
      Q[(state_tuple, action_int)] = float
      
      Exemple:
        Q[((2,3,1,4,2,3,8,1,0), 1)] = 2.5
           └─ state (9 valeurs) ─┘  └action  └value
    """
    
    def __init__(self, 
                 n_actions=3,
                 gamma=0.95,
                 learning_rate=0.01,
                 seed=42):
        """
        Initialise l'agent Q-Learning

        """
        # Nombre d'actions
        self.n_actions = n_actions
        
        # Hyperparamètres Q-Learning (ÉTAPE 6)
        self.gamma = gamma              # 0.95 = horizon ~4 steps (1h)
        self.learning_rate = learning_rate  # 0.01 = stable
        
        # Q-table : dict {(state, action): Q-value}
        # defaultdict(float) : retourne 0.0 si clé inexistante
        # → Optimistic initialization (encourage exploration états nouveaux)
        self.q_table = defaultdict(float)
        
        # Compteur des états visités (pour debug/analyse)
        self.state_visit_counts = defaultdict(int)
        
        # Seed
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"  ✓ Agent Q-Learning initialisé")
        print(f"    Actions: {self.n_actions}")
        print(f"    Gamma: {self.gamma}")
        print(f"    Learning rate: {self.learning_rate}")

    
    def get_action(self, state, epsilon):
        """
        Choisit une action selon stratégie ε-greedy
        
        Args:
            state: tuple (état actuel)
            epsilon: float (probabilité d'explorer)
        
        Returns:
            action: int (0, 1 ou 2)
        """
        # Exploration vs Exploitation
        if np.random.rand() < epsilon:
            # Exploration : choix aléatoire
            action = random.randint(0,self.n_actions - 1)
            return action
        # Exploitation : meilleure action connue
        # Récupère Q-values pour toutes les actions dans ce state
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        
        # Choisit action avec max Q-value
        # Si égalité, random.choice parmi les max
        max_q = max(q_values)
        best_actions = [a for a in range(self.n_actions) if q_values[a] == max_q]
        action = random.choice(best_actions)
        
        # Compte visite de cet état (pour analyse)
        self.state_visit_counts[state] += 1
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-table selon équation de Bellman
        
        Q(state, action) ← Q(state, action) + α [reward + γ max_a' Q(next_state, a') - Q(state, action)]
        
        Args:
            state: tuple (état actuel)
            action: int (action prise)
            reward: float (récompense reçue)
            next_state: tuple (état suivant)
            done: bool (épisode terminé?)
        """
        current_q = self.q_table[(state, action)]
        
        if done:
            td_target = reward  # Pas de next_state si épisode terminé
        else:
            # Valeur maximale du next_state
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
            td_target = reward + self.gamma * max_next_q
        
        td_error = td_target - current_q
        
        # Mise à jour de la Q-table
        new_q = current_q + self.learning_rate * td_error
        self.q_table[(state, action)] = new_q

    def save(self, filepath):
        """
        Sauvegarde la Q-table et métadata
        
        Args:
            filepath: Path où sauvegarder (pickle)
        """
        data = {
            'q_table': dict(self.q_table),  # Convertit defaultdict → dict
            'state_visit_counts': dict(self.state_visit_counts),
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'n_states_visited': len(self.state_visit_counts),
            'n_q_entries': len(self.q_table)
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"  ✓ Modèle sauvegardé: {filepath}")
        print(f"    Q-entries: {len(self.q_table):,}")
        print(f"    États visités: {len(self.state_visit_counts):,}")
    

    def load(self, filepath):
        """
        Charge une Q-table sauvegardée
        
        Args:
            filepath: Path du fichier à charger
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restaure Q-table en defaultdict
        self.q_table = defaultdict(float, data['q_table'])
        self.state_visit_counts = defaultdict(int, data.get('state_visit_counts', {}))
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        self.learning_rate = data['learning_rate']
        
        print(f"  ✓ Modèle chargé: {filepath}")
        print(f"    Q-entries: {len(self.q_table):,}")
        print(f"    États visités: {len(self.state_visit_counts):,}")



# ════════════════════════════════════════════════════════════════
# BLOC 9 : FONCTION D'ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════

def train_agent(agent, env, episodes=200, epsilon_start=0.2, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Entraîne l'agent sur plusieurs épisodes
    
    Args:
        agent: QLearningAgent
        env: TradingEnvironment
        episodes: Nombre d'épisodes
        epsilon_start: Exploration initiale
        epsilon_end: Exploration finale
        epsilon_decay: Taux de décroissance
    
    Returns:
        training_history: dict avec métriques par épisode
    """
    print("\n" + "="*60)
    print(f"ENTRAÎNEMENT Q-LEARNING ({episodes} épisodes)")
    print("="*60)
    
    history = {
        'episode': [],
        'total_reward': [],
        'final_equity': [],
        'epsilon': [],
        'steps': []
    }

    epsilon = epsilon_start
    
    for episode in range(episodes):
        # Reset environnement
        state = env.reset()
        total_reward = 0
        steps = 0
    
            # Parcourt l'épisode
        done = False
        while not done:
            # Agent choisit action (ε-greedy)
            action = agent.get_action(state, epsilon)
            
            # Environnement exécute action
            next_state, reward, done = env.step(action)
            
            # Agent apprend (Bellman)
            agent.update(state, action, reward, next_state, done)
            
            # Mise à jour
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
            # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Sauvegarde historique
        history['episode'].append(episode)
        history['total_reward'].append(total_reward)
        history['final_equity'].append(env.equity)
        history['epsilon'].append(epsilon)
        history['steps'].append(steps)
        
        # Affichage tous les 20 épisodes
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1:3d}/{episodes} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Equity: {env.equity:8.4f} | "
                  f"ε: {epsilon:.3f} | "
                  f"Steps: {steps:5d}")
    print(f"\n✓ Entraînement terminé")
    print(f"  États visités: {len(agent.state_visit_counts):,}")
    print(f"  Q-entries: {len(agent.q_table):,}")
    
    return history



# ════════════════════════════════════════════════════════════════
# BLOC 10 : FONCTION D'ÉVALUATION
# ════════════════════════════════════════════════════════════════

def evaluate_agent(agent, env, epsilon=0.0, name="Evaluation"):
    """
    Évalue l'agent sur données (epsilon=0 = pure exploitation)
    
    Args:
        agent: QLearningAgent
        env: TradingEnvironment
        epsilon: 0.0 pour test (pur greedy)
        name: Nom pour affichage
    
    Returns:
        results: dict avec métriques
    """
    state = env.reset()
    done = False
    
    positions = []
    returns = []
    
    while not done:
        action = agent.get_action(state, epsilon)
        next_state, reward, done = env.step(action)
        
        positions.append(env.position)
        returns.append(env.equity)
        
        state = next_state

    # Calcule métriques
    equity_curve = np.array(env.equity_curve)
    equity_returns = np.diff(equity_curve)
    
    # Cumulative profit
    cumulative_profit = env.equity
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.where(peak > 0, peak, 1)
    max_drawdown = drawdown.min()

    # Sharpe ratio
    if len(equity_returns) > 0 and equity_returns.std() > 0:
        sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(252 * 96)
    else:
        sharpe = 0
    
    # Profit factor
    gains = equity_returns[equity_returns > 0].sum()
    losses = abs(equity_returns[equity_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf
    
    results = {
        'name': name,
        'cumulative_profit': cumulative_profit,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'total_steps': len(positions)
    }
    
    return results


# ════════════════════════════════════════════════════════════════
# BLOC 11 : FONCTION MAIN
# ════════════════════════════════════════════════════════════════

def main():
    """
    Pipeline complet : Préparation → Entraînement → Évaluation
    """
    print("\n" + "="*70)
    print(" REINFORCEMENT LEARNING - TRADING GBP/USD M15")
    print("="*70)
    
    # 1. PRÉPARATION DONNÉES
    print("\n[ÉTAPE 1/4] Préparation des données")
    print("-" * 60)
    
    preparer = DataPreparer(
        features=FEATURES,
        warm_up=CONFIG['warm_up'],
        n_bins=CONFIG['n_bins'],
        seed=CONFIG['seed']
    )

    train_data, val_data, test_data, metadata = preparer.get_train_val_test()
    
    # 2. CRÉATION ENVIRONNEMENT ET AGENT
    print("\n[ÉTAPE 2/4] Création environnement et agent")
    print("-" * 60)
    
    env_train = TradingEnvironment(
        data=train_data,
        features=FEATURES,
        reward_scale=CONFIG['reward_scale'],
        transaction_cost=CONFIG['transaction_cost'],
        drawdown_penalty=CONFIG['drawdown_penalty']
        #max_drawdown_stop=CONFIG['max_drawdown_stop']
    )
    
    agent = QLearningAgent(
        n_actions=N_ACTIONS,
        gamma=CONFIG['gamma'],
        learning_rate=CONFIG['learning_rate'],
        seed=CONFIG['seed']
    )

    # 3. ENTRAÎNEMENT
    print("\n[ÉTAPE 3/4] Entraînement sur 2022")
    print("-" * 60)
    
    history = train_agent(
        agent=agent,
        env=env_train,
        episodes=CONFIG['episodes'],
        epsilon_start=CONFIG['epsilon_start'],
        epsilon_end=CONFIG['epsilon_end'],
        epsilon_decay=CONFIG['epsilon_decay']
    )
    
    # 4. ÉVALUATION
    print("\n[ÉTAPE 4/4] Évaluation sur 2023 et 2024")
    print("-" * 60)

    # Val 2023
    env_val = TradingEnvironment(val_data, FEATURES, CONFIG['reward_scale'], 
                                  CONFIG['transaction_cost'], CONFIG['drawdown_penalty'])
    results_val = evaluate_agent(agent, env_val, epsilon=0.0, name="2023 (Val)")
    
    # Test 2024
    env_test = TradingEnvironment(test_data, FEATURES, CONFIG['reward_scale'],
                                   CONFIG['transaction_cost'], CONFIG['drawdown_penalty'])
    results_test = evaluate_agent(agent, env_test, epsilon=0.0, name="2024 (Test)")
    
    # Affichage résultats
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)

    for res in [results_val, results_test]:
        print(f"\n{res['name']}:")
        print(f"  Cumulative Profit: {res['cumulative_profit']:.4f}")
        print(f"  Max Drawdown:      {res['max_drawdown']:.4f}")
        print(f"  Sharpe Ratio:      {res['sharpe']:.4f}")
        print(f"  Profit Factor:     {res['profit_factor']:.4f}")
    
    # 5. SAUVEGARDE
    print("\n[SAUVEGARDE] Modèle et métadata")
    print("-" * 60)
    
    model_path = MODELS_DIR / "q_learning_clean.pkl"
    agent.save(model_path)

    # Sauvegarde metadata
    metadata['results_val'] = results_val
    metadata['results_test'] = results_test
    metadata_path = MODELS_DIR / "metadata_clean.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("✓ PIPELINE TERMINÉ")
    print("="*70)
    
    return agent, history, results_val, results_test

# ════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agent, history, val_results, test_results = main()