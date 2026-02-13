import React, { useState } from 'react';
import axios from 'axios';

interface PredictResponse {
  prediction: number;
  model: string;
}

interface FeatureInfo {
  name: string;
  category: 'price' | 'return' | 'indicator' | 'volume' | 'other';
  description: string;
}

const FEATURE_INFO: FeatureInfo[] = [
  { name: 'valeur 1', category: 'price', description: 'Prix d\'ouverture' },
  { name: 'valeur 2', category: 'price', description: 'Prix le plus haut' },
  { name: 'valeur 3', category: 'price', description: 'Prix le plus bas' },
  { name: 'valeur 4', category: 'price', description: 'Prix de clôture' },
  { name: 'valeur 5', category: 'price', description: 'Écart d\'ouverture' },
  { name: 'valeur 6', category: 'volume', description: 'Volume' },

  
];

interface TrainedModel {
  id: string;
  type: 'ML' | 'RL';
  name: string;
  trainDate: string;
  metrics: {
    sharpe: number;
    profit: number;
    drawdown: number;
    trades: number;
  };
  status: 'active' | 'inactive';
  config: any;
}

function App() {
  const [currentPage, setCurrentPage] = useState<'predict' | 'config'>('predict');
  const [modelInfo, setModelInfo] = useState<string>('');
  const [modelType, setModelType] = useState<string>('');
  const [features, setFeatures] = useState<string[]>(Array(39).fill(''));
  const [prediction, setPrediction] = useState<number | null>(null);
  const [modelUsed, setModelUsed] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModel, setLoadingModel] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  
  // Config page states
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([
    {
      id: 'ml_v1',
      type: 'ML',
      name: 'Logistic Regression v1',
      trainDate: '2024-01-15',
      metrics: { sharpe: 0.026, profit: 1.66, drawdown: -8.5, trades: 145 },
      status: 'inactive',
      config: { features: 38, model: 'Logistic' }
    },
    {
      id: 'rl_v1',
      type: 'RL',
      name: 'Q-Learning v1',
      trainDate: '2024-02-10',
      metrics: { sharpe: 0.45, profit: 5.2, drawdown: -6.2, trades: 89 },
      status: 'active',
      config: { features: 8, episodes: 200, gamma: 0.95 }
    }
  ]);
  const [trainingML, setTrainingML] = useState(false);
  const [trainingRL, setTrainingRL] = useState(false);

  const API_BASE = 'http://localhost:8000';

  const fetchModelInfo = async () => {
    setLoadingModel(true);
    setError('');
    try {
      const response = await axios.get(`${API_BASE}/`);
      setModelInfo(response.data.message);
      // Extraire le nom du modèle du message
      const match = response.data.message.match(/: (.+)$/);
      if (match) {
        setModelType(match[1]);
      }
    } catch (err: any) {
      const errorMsg = err.message || 'Erreur de connexion';
      setError(errorMsg);
      setModelInfo('');
      setModelType('');
    } finally {
      setLoadingModel(false);
    }
  };

  const handleInputChange = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handlePredict = async () => {
    setError('');
    setPrediction(null);
    setLoading(true);

    try {
      const featureArray = features.map(f => parseFloat(f) || 0);

      const response = await axios.post<PredictResponse>(
        `${API_BASE}/predict`,
        { features: featureArray }
      );

      setPrediction(response.data.prediction);
      setModelUsed(response.data.model);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur lors de la prédiction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = () => {
    const example = [
      '1.2765', '1.2768', '1.2763', '1.2766', '0.00015',
      '0.0002', '0.0001', '-0.0001', '0.0003', '0.0002',
      '0.0005', '-0.0002', '0.0004', '-0.0001', '0.0003',
      '1.2760', '1.2755', '1.2740', '45.5',
      '0.0005', '0.0006', '-0.0001',
      '1.2770', '1.2766', '1.2762',
      '0.0008', '150000', '145000',
      '65.5', '45.2', '25.3',
      '100', '-45.8', '8.5',
      '1500000', '52.3', '0.0003',
      '1.2765', '1.2764'
    ];
    setFeatures(example);
  };

  const clearAll = () => {
    setFeatures(Array(39).fill(''));
    setPrediction(null);
    setError('');
  };

  const handleActivateModel = (modelId: string) => {
    setTrainedModels(prev => prev.map(m => ({
      ...m,
      status: m.id === modelId ? 'active' : 'inactive'
    })));
  };

  const handleTrainML = () => {
    setTrainingML(true);
    // Mock training
    setTimeout(() => {
      const newModel: TrainedModel = {
        id: `ml_${Date.now()}`,
        type: 'ML',
        name: 'Random Forest v2',
        trainDate: new Date().toISOString().split('T')[0],
        metrics: { 
          sharpe: Math.random() * 0.5, 
          profit: Math.random() * 3, 
          drawdown: -(Math.random() * 10),
          trades: Math.floor(Math.random() * 200) + 50
        },
        status: 'inactive',
        config: { features: 38, model: 'RandomForest' }
      };
      setTrainedModels(prev => [...prev, newModel]);
      setTrainingML(false);
    }, 3000);
  };

  const handleTrainRL = () => {
    setTrainingRL(true);
    // Mock training
    setTimeout(() => {
      const newModel: TrainedModel = {
        id: `rl_${Date.now()}`,
        type: 'RL',
        name: 'Q-Learning v2',
        trainDate: new Date().toISOString().split('T')[0],
        metrics: { 
          sharpe: Math.random() * 0.8, 
          profit: Math.random() * 8, 
          drawdown: -(Math.random() * 8),
          trades: Math.floor(Math.random() * 150) + 40
        },
        status: 'inactive',
        config: { features: 8, episodes: 200, gamma: 0.95 }
      };
      setTrainedModels(prev => [...prev, newModel]);
      setTrainingRL(false);
    }, 5000);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* HEADER */}
      <header className="bg-white border-b border-gray-200 py-4 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-green-500 rounded-lg p-2">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <h1 className="text-xl font-semibold text-gray-900">Trading Predictor</h1>
          </div>
          <div className="bg-gray-100 text-gray-600 px-3 py-1 rounded-md text-sm font-medium">
            ML / RL
          </div>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        
        {currentPage === 'predict' && (
        <>
        {/* SECTION MODELE */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <h2 className="text-lg font-semibold text-gray-900">Modèle</h2>
          </div>
          
          <button
            onClick={fetchModelInfo}
            disabled={loadingModel}
            className="w-full bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white font-medium py-3 px-4 rounded-lg transition-colors disabled:cursor-not-allowed"
          >
            {loadingModel ? 'Chargement...' : 'Charger le modèle'}
          </button>
          
          {modelInfo && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800">{modelInfo}</p>
              {modelType && (
                <p className="text-xs text-green-600 mt-1 font-medium">Type: {modelType}</p>
              )}
            </div>
          )}
        </div>

        {/* ERROR DISPLAY */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-start gap-2">
              <svg className="w-5 h-5 text-red-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="text-sm font-medium text-red-800">Echec de connexion</p>
                <p className="text-sm text-red-600">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* SECTION FEATURES */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <h2 className="text-lg font-semibold text-gray-900">Features</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
            {FEATURE_INFO.map((feature, index) => (
              <div key={index}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {feature.description}
                </label>
                <input
                  type="number"
                  step="any"
                  value={features[index]}
                  onChange={(e) => handleInputChange(index, e.target.value)}
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none text-sm"
                  placeholder={`ex: ${index === 0 ? '1.1050' : index === 1 ? '1.1085' : index === 2 ? '1.1020' : index === 3 ? '1.1070' : index === 4 ? '52340' : '55.3'}`}
                />
              </div>
            ))}
          </div>
          
          <div className="flex gap-3">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="flex-1 bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white font-medium py-3 px-4 rounded-lg transition-colors disabled:cursor-not-allowed"
            >
              {loading ? 'Prédiction en cours...' : 'Prédire'}
            </button>
            <button
              onClick={clearAll}
              className="bg-gray-200 hover:bg-gray-300 text-gray-700 font-medium py-3 px-6 rounded-lg transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* SECTION RESULTAT */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="text-lg font-semibold text-gray-900">Resultat</h2>
          </div>
          
          {prediction === null ? (
            <div className="text-center py-12">
              <svg className="w-16 h-16 text-gray-300 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <p className="text-gray-400">En attente d'une prédiction...</p>
            </div>
          ) : (
            <div className="text-center py-8">
              {prediction === 1 && (
                <div className="inline-block">
                  <div className="text-6xl mb-2">📈</div>
                  <div className="text-2xl font-bold text-green-600">BUY</div>
                  <p className="text-sm text-gray-500 mt-1">Signal haussier - Acheter</p>
                </div>
              )}
              {prediction === 0 && (
                <div className="inline-block">
                  <div className="text-6xl mb-2">➖</div>
                  <div className="text-2xl font-bold text-gray-600">HOLD</div>
                  <p className="text-sm text-gray-500 mt-1">Pas de signal - Maintenir</p>
                </div>
              )}
              {prediction === -1 && (
                <div className="inline-block">
                  <div className="text-6xl mb-2">📉</div>
                  <div className="text-2xl font-bold text-red-600">SELL</div>
                  <p className="text-sm text-gray-500 mt-1">Signal baissier - Vendre</p>
                </div>
              )}
              <p className="text-xs text-gray-400 mt-4">Modèle: {modelUsed}</p>
            </div>
          )}
        </div>

        {/* GUIDE DES SIGNAUX */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <h2 className="text-lg font-semibold text-gray-900">Guide des signaux</h2>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center gap-3">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
                <div>
                  <p className="font-semibold text-green-700">BUY</p>
                  <p className="text-sm text-green-600">Signal haussier - Acheter</p>
                </div>
              </div>
              <span className="text-green-700 font-mono text-sm">1</span>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-orange-50 border border-orange-200 rounded-lg">
              <div className="flex items-center gap-3">
                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                </svg>
                <div>
                  <p className="font-semibold text-orange-700">HOLD</p>
                  <p className="text-sm text-orange-600">Pas de signal - Maintenir</p>
                </div>
              </div>
              <span className="text-orange-700 font-mono text-sm">0</span>
            </div>
            
            <div className="flex items-center justify-between p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center gap-3">
                <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                </svg>
                <div>
                  <p className="font-semibold text-red-700">SELL</p>
                  <p className="text-sm text-red-600">Signal baissier - Vendre</p>
                </div>
              </div>
              <span className="text-red-700 font-mono text-sm">-1</span>
            </div>
          </div>
        </div>
        
        {/* BOUTON CONFIG */}
        <div className="mt-8 text-center">
          <button
            onClick={() => setCurrentPage('config')}
            className="inline-flex items-center gap-2 bg-gray-700 hover:bg-gray-800 text-white px-6 py-3 rounded-lg font-medium transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Config (Dev)
          </button>
        </div>
        </>
        )}

        {currentPage === 'config' && (
          <div>
            {/* HEADER CONFIG */}
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Configuration & Entraînement</h2>
              <button
                onClick={() => setCurrentPage('predict')}
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 font-medium"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Retour
              </button>
            </div>

            {/* SECTION ML */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
                <h3 className="text-lg font-semibold text-gray-900">Machine Learning (ML)</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Modèle</label>
                  <select className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm">
                    <option>Logistic Regression</option>
                    <option>Random Forest</option>
                    <option>XGBoost</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Features</label>
                  <input type="number" defaultValue="38" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Train Year</label>
                  <select className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm">
                    <option>2022</option>
                    <option>2023</option>
                    <option>2022-2023</option>
                  </select>
                </div>
              </div>
              
              <button
                onClick={handleTrainML}
                disabled={trainingML}
                className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                {trainingML ? '⏳ Entraînement ML en cours...' : '🚀 Entraîner Modèle ML'}
              </button>
            </div>

            {/* SECTION RL */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <h3 className="text-lg font-semibold text-gray-900">Reinforcement Learning (RL)</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Features</label>
                  <input type="number" defaultValue="8" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Episodes</label>
                  <input type="number" defaultValue="200" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Gamma (γ)</label>
                  <input type="number" step="0.01" defaultValue="0.95" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Learning Rate</label>
                  <input type="number" step="0.001" defaultValue="0.01" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Epsilon Start</label>
                  <input type="number" step="0.01" defaultValue="0.2" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Reward Scale</label>
                  <input type="number" defaultValue="100" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">N Bins</label>
                  <input type="number" defaultValue="5" className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-sm" />
                </div>
              </div>
              
              <button
                onClick={handleTrainRL}
                disabled={trainingRL}
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white font-medium py-3 px-4 rounded-lg transition-colors"
              >
                {trainingRL ? '⏳ Entraînement RL en cours (~5min)...' : '🚀 Entraîner Modèle RL'}
              </button>
            </div>

            {/* LISTE MODELES */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Modèles Entraînés ({trainedModels.length})</h3>
              
              <div className="space-y-3">
                {trainedModels.map((model) => (
                  <div
                    key={model.id}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      model.status === 'active'
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`px-2 py-1 rounded text-xs font-bold ${
                            model.type === 'ML' ? 'bg-blue-100 text-blue-700' : 'bg-green-100 text-green-700'
                          }`}>
                            {model.type}
                          </span>
                          <h4 className="font-semibold text-gray-900">{model.name}</h4>
                          {model.status === 'active' && (
                            <span className="text-xs bg-green-500 text-white px-2 py-1 rounded">✓ Actif</span>
                          )}
                        </div>
                        <p className="text-xs text-gray-500">Entraîné le {model.trainDate}</p>
                      </div>
                      
                      {model.status === 'inactive' && (
                        <button
                          onClick={() => handleActivateModel(model.id)}
                          className="bg-gray-700 hover:bg-gray-800 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                          Activer
                        </button>
                      )}
                    </div>
                    
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500 text-xs">Sharpe Ratio</p>
                        <p className={`font-bold ${
                          model.metrics.sharpe > 0.5 ? 'text-green-600' : 
                          model.metrics.sharpe > 0.3 ? 'text-orange-600' : 'text-red-600'
                        }`}>
                          {model.metrics.sharpe.toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-500 text-xs">Profit (%)</p>
                        <p className={`font-bold ${
                          model.metrics.profit > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {model.metrics.profit > 0 ? '+' : ''}{model.metrics.profit.toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-500 text-xs">Max DD (%)</p>
                        <p className="font-bold text-red-600">{model.metrics.drawdown.toFixed(2)}%</p>
                      </div>
                      <div>
                        <p className="text-gray-500 text-xs">Trades</p>
                        <p className="font-bold text-gray-700">{model.metrics.trades}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
