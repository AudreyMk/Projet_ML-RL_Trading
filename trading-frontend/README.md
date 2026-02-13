# 🎯 Trading Signal Predictor - Frontend

Interface web React + TypeScript pour tester votre API de trading.

## 🚀 Installation

### 1. Installez les dépendances

```bash
npm install
```

### 2. Lancez le serveur de développement

```bash
npm run dev
```

L'application sera accessible sur **http://localhost:3000**

## 🔧 Configuration de l'API Backend

### Activez CORS sur votre FastAPI

Dans votre fichier Python principal (main.py ou équivalent), ajoutez :

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Best Model API")

# ⭐ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Autoriser le frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... reste de votre code
```

### Lancez votre API

```bash
cd api
uvicorn main:app --reload
```

Votre API devrait être sur **http://localhost:8000**

## 📦 Structure du Projet

```
trading-frontend/
├── src/
│   ├── App.tsx          # Composant principal
│   ├── main.tsx         # Point d'entrée
│   └── index.css        # Styles globaux
├── index.html           # Template HTML
├── package.json         # Dépendances
├── tsconfig.json        # Config TypeScript
├── vite.config.ts       # Config Vite
├── tailwind.config.js   # Config Tailwind
└── postcss.config.js    # Config PostCSS
```

## ✨ Fonctionnalités

- ✅ **39 inputs** avec catégories colorées (Prix, Rendements, Indicateurs, Volume)
- ✅ **TypeScript** strict pour la sécurité des types
- ✅ **Interface moderne** avec Tailwind CSS
- ✅ **Système d'onglets** (Formulaire / Résultats)
- ✅ **Bouton "Charger exemple"** pour tester rapidement
- ✅ **Prédictions visuelles** (LONG 📈 / FLAT ➖ / SHORT 📉)
- ✅ **Gestion d'erreurs** complète
- ✅ **Responsive design**

## 🛠️ Scripts Disponibles

- `npm run dev` - Lance le serveur de développement
- `npm run build` - Compile pour la production
- `npm run preview` - Prévisualise la version de production

## 🎨 Technologies Utilisées

- **React 18** - Framework UI
- **TypeScript** - Typage statique
- **Vite** - Build tool ultra-rapide
- **Tailwind CSS** - Framework CSS utility-first
- **Axios** - Client HTTP

## 📝 Utilisation

1. Lancez votre API backend (`uvicorn main:app --reload`)
2. Lancez le frontend (`npm run dev`)
3. Ouvrez http://localhost:3000
4. Cliquez sur "Charger exemple" pour tester
5. Cliquez sur "PRÉDIRE LE SIGNAL"
6. Consultez le résultat dans l'onglet "Résultats"

## ⚠️ Résolution de Problèmes

### Erreur "API non accessible"
- Vérifiez que votre backend est lancé sur http://localhost:8000
- Vérifiez que CORS est activé (voir section Configuration)

### Erreur TypeScript
- Assurez-vous que toutes les dépendances sont installées : `npm install`
- Supprimez node_modules et réinstallez : `rm -rf node_modules && npm install`

### Port 3000 déjà utilisé
- Modifiez le port dans `vite.config.ts`

## 📄 License

MIT
