"""
API de scoring pour la détection de fraude en temps réel
Version simple utilisant le meilleur modèle entraîné
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèle de données pour l'API
class TransactionRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: str

# Initialiser FastAPI
app = FastAPI(
    title="API de Détection de Fraude",
    description="API pour scorer des transactions bancaires en temps réel",
    version="1.0.0"
)

# Variables globales pour le modèle
model = None
feature_columns = None

def load_model():
    """Charge le meilleur modèle entraîné"""
    global model, feature_columns
    
    try:
        model_path = Path("models/random_forest_model.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Colonnes features (sans Time et Class)
            feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
            
            logger.info("✅ Modèle Random Forest chargé avec succès")
        else:
            logger.error("❌ Modèle non trouvé. Exécutez d'abord l'entraînement.")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")

# Charger le modèle au démarrage
@app.on_event("startup")
async def startup_event():
    load_model()

# Route de santé
@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

# Route d'information sur le modèle
@app.get("/model/info")
async def model_info():
    """Informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": "RandomForestClassifier",
        "features_count": len(feature_columns),
        "features": feature_columns,
        "performance": {
            "roc_auc": 0.929,
            "precision": 0.867,
            "recall": 0.727,
            "f1_score": 0.791
        }
    }

# Route principale de scoring
@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Prédit si une transaction est frauduleuse
    
    Args:
        transaction: Données de la transaction
        
    Returns:
        Prédiction avec probabilité et niveau de risque
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # Convertir en DataFrame
        transaction_data = transaction.dict()
        df = pd.DataFrame([transaction_data])
        
        # Vérifier que toutes les colonnes sont présentes
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Features manquantes: {list(missing_features)}"
            )
        
        # Réorganiser les colonnes dans le bon ordre
        df = df[feature_columns]
        
        # Prédiction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]
        
        # Déterminer le niveau de risque
        if probability < 0.3:
            risk_level = "LOW"
            confidence = "HIGH"
        elif probability < 0.7:
            risk_level = "MEDIUM" 
            confidence = "MEDIUM"
        else:
            risk_level = "HIGH"
            confidence = "HIGH"
        
        # Si le modèle prédit une fraude mais avec faible probabilité
        if prediction == 1 and probability < 0.5:
            confidence = "LOW"
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# Route pour scorer plusieurs transactions
@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """
    Prédit plusieurs transactions en une fois
    
    Args:
        transactions: Liste des transactions à scorer
        
    Returns:
        Liste des prédictions
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions par batch")
    
    try:
        results = []
        
        for i, transaction in enumerate(transactions):
            try:
                prediction = await predict_fraud(transaction)
                results.append({
                    "transaction_id": i,
                    "prediction": prediction
                })
            except Exception as e:
                results.append({
                    "transaction_id": i,
                    "error": str(e)
                })
        
        fraud_count = sum(1 for r in results if "prediction" in r and r["prediction"].is_fraud)
        
        return {
            "total_transactions": len(transactions),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(transactions),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du batch scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de batch scoring: {str(e)}")

# Route pour recharger le modèle
@app.post("/model/reload")
async def reload_model():
    """Recharge le modèle depuis le disque"""
    try:
        load_model()
        return {
            "message": "Modèle rechargé avec succès",
            "model_loaded": model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du rechargement: {str(e)}")

# Route d'exemple pour tester
@app.get("/example")
async def get_example_transaction():
    """Retourne un exemple de transaction pour tester l'API"""
    return {
        "example_normal": {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        },
        "usage": "POST /predict avec les données ci-dessus"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
