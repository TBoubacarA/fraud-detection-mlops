"""
Script d'ingestion simplifié pour débuter - Version sans psycopg2
"""

import pandas as pd
import numpy as np
import mlflow
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataIngestion:
    """Version simplifiée de l'ingestion"""
    
    def __init__(self):
        # Configuration MLflow (port 5001 selon votre docker-compose)
        self.mlflow_uri = "http://localhost:5001"
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Créer l'expérience
        self.experiment_name = "fraud_detection_pipeline"
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"✅ Expérience créée: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"⚠️ MLflow warning: {e}")
            logger.info("⚠️ Continuons sans MLflow pour l'instant")
    
    def check_data_file(self, file_path="data/raw/creditcard.csv"):
        """Vérifie que le fichier de données existe"""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ Fichier non trouvé: {file_path}")
            logger.info("📥 Pour télécharger le dataset:")
            logger.info("   1. Allez sur: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            logger.info("   2. Téléchargez creditcard.csv") 
            logger.info(f"   3. Placez-le dans: {file_path}")
            return False
        
        logger.info(f"✅ Fichier trouvé: {file_path}")
        return True
    
    def load_and_validate_data(self, file_path="data/raw/creditcard.csv"):
        """Charge et valide les données"""
        
        if not self.check_data_file(file_path):
            return None
            
        logger.info("📊 Chargement des données...")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            return None
        
        # Validation basique
        logger.info(f"   • Dimensions: {df.shape}")
        logger.info(f"   • Colonnes: {list(df.columns)}")
        
        if 'Class' in df.columns:
            fraud_count = df['Class'].sum()
            fraud_rate = fraud_count / len(df) * 100
            logger.info(f"   • Fraudes: {fraud_count} ({fraud_rate:.3f}%)")
        else:
            logger.error("❌ Colonne 'Class' manquante")
            return None
        
        # Vérifications
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        logger.info(f"   • Valeurs manquantes: {missing_values}")
        logger.info(f"   • Doublons: {duplicates}")
        
        return df
    
    def split_data(self, df):
        """Divise les données en train/val/test"""
        logger.info("�� Division des données...")
        
        # Séparer features et target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Premier split: train+val / test (80/20)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Deuxième split: train / val (75/25 du temp, soit 60/20 du total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Reconstituer les DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"   • Train: {len(train_df)} lignes ({y_train.mean():.3%} fraude)")
        logger.info(f"   • Validation: {len(val_df)} lignes ({y_val.mean():.3%} fraude)")
        logger.info(f"   • Test: {len(test_df)} lignes ({y_test.mean():.3%} fraude)")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df):
        """Sauvegarde les données prétraitées"""
        logger.info("💾 Sauvegarde des données...")
        
        # Créer le dossier de sortie
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "validation.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        logger.info(f"   ✅ Données sauvegardées dans {output_dir}/")
        
    def run_ingestion(self):
        """Exécute l'ingestion complète"""
        logger.info("🚀 DÉMARRAGE DE L'INGESTION")
        
        try:
            with mlflow.start_run(run_name="data_ingestion_simple"):
                
                # 1. Charger les données
                df = self.load_and_validate_data()
                if df is None:
                    logger.error("❌ Impossible de charger les données")
                    return False
                
                # 2. Logger les métriques initiales
                try:
                    mlflow.log_metrics({
                        "total_rows": len(df),
                        "total_columns": len(df.columns),
                        "fraud_cases": df['Class'].sum(),
                        "fraud_rate": df['Class'].mean()
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Pas de logging MLflow: {e}")
                
                # 3. Diviser les données
                train_df, val_df, test_df = self.split_data(df)
                
                # 4. Sauvegarder
                self.save_processed_data(train_df, val_df, test_df)
                
                # 5. Logger les métriques finales
                try:
                    mlflow.log_metrics({
                        "train_size": len(train_df),
                        "val_size": len(val_df), 
                        "test_size": len(test_df)
                    })
                    
                    mlflow.log_params({
                        "test_ratio": 0.2,
                        "val_ratio": 0.2,
                        "random_state": 42
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Pas de logging MLflow: {e}")
                
                logger.info("✅ INGESTION TERMINÉE AVEC SUCCÈS")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur MLflow, continuons sans: {e}")
            
            # Version sans MLflow
            df = self.load_and_validate_data()
            if df is None:
                return False
            
            train_df, val_df, test_df = self.split_data(df)
            self.save_processed_data(train_df, val_df, test_df)
            
            logger.info("✅ INGESTION TERMINÉE (sans MLflow)")
            return True

def main():
    """Fonction principale"""
    
    print("🏦 INGESTION DE DONNÉES - DÉTECTION DE FRAUDE")
    print("=" * 50)
    
    # Vérifier les packages
    try:
        import pandas
        import numpy  
        import sklearn
        print("✅ Packages Python OK")
    except ImportError as e:
        print(f"❌ Package manquant: {e}")
        return
    
    # Vérifier la structure des dossiers
    required_dirs = ["data/raw", "data/processed", "logs"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Exécuter l'ingestion
    ingestion = SimpleDataIngestion()
    success = ingestion.run_ingestion()
    
    if success:
        print("\n🎉 SUCCÈS!")
        print("📊 Vous pouvez maintenant:")
        print("   1. Vérifier les données dans data/processed/")
        print("   2. Consulter MLflow: http://localhost:5001")
        print("   3. Passer à l'entraînement des modèles")
        
        # Afficher un aperçu
        try:
            train_df = pd.read_csv("data/processed/train.csv")
            print(f"\n📈 Aperçu des données d'entraînement:")
            print(f"   • {len(train_df)} transactions")
            print(f"   • {train_df['Class'].sum()} fraudes")
            print(f"   • {train_df['Class'].mean():.3%} taux de fraude")
        except Exception as e:
            print(f"⚠️ Impossible d'afficher l'aperçu: {e}")
            
    else:
        print("\n❌ ÉCHEC - Vérifiez les logs ci-dessus")
        print("💡 Assurez-vous d'avoir téléchargé creditcard.csv dans data/raw/")

if __name__ == "__main__":
    main()
