
"""
Train Dwell Time Prediction Model

A production-grade machine learning pipeline for predicting train dwell times
using ensemble methods (Random Forest + XGBoost).

Usage:
    python train_dwell_time_model.py --config config.json
    python train_dwell_time_model.py --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TrainDwellTimeConfig:
    """Configuration class for the train dwell time model."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.features = [
            'TrainNo', 'TrainType', 'HaltStation', 'PFNo', 'BlockNo', 
            'BlockLen', 'ApproachingBlockNo', 'CurrentSpeed', 'CurrentDelay', 
            'DFNS', 'RunningStatus'
        ]
        self.target = 'DwellTime'
        self.numeric_cols = ['BlockLen', 'CurrentSpeed', 'CurrentDelay', 'DFNS']
        self.categorical_cols = [
            'TrainNo', 'TrainType', 'HaltStation', 'PFNo', 'BlockNo', 
            'ApproachingBlockNo', 'RunningStatus'
        ]
        
        # Model hyperparameters
        self.rf_params = {
            "model__n_estimators": [200, 300, 500, 700],
            "model__max_depth": [None, 10, 20, 30],
            "model__max_features": ["sqrt", "log2", 0.8],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__bootstrap": [True, False]
        }
        
        self.xgb_params = {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, 10],
            "model__subsample": [0.6, 0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "model__reg_alpha": [0, 0.1, 0.5],
            "model__reg_lambda": [1, 1.5, 2]
        }
        
        # Training parameters
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        self.n_iter = 50
        self.n_jobs = -1
        
        # Ensemble weights
        self.rf_weight = 0.8
        self.xgb_weight = 0.2
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update attributes from config
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")


class EnsembleModel:
    """Custom ensemble model combining Random Forest and XGBoost."""
    
    def __init__(self, rf_model, xgb_model, rf_weight: float = 0.8, xgb_weight: float = 0.2):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        
        # Validate weights
        if not np.isclose(rf_weight + xgb_weight, 1.0):
            raise ValueError("RF and XGB weights must sum to 1.0")
    
    def predict(self, X):
        """Make predictions using weighted ensemble."""
        try:
            y_rf = self.rf_model.predict(X)
            y_xgb = self.xgb_model.predict(X)
            return self.rf_weight * y_rf + self.xgb_weight * y_xgb
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from both models."""
        try:
            rf_importance = self.rf_model.named_steps['model'].feature_importances_
            xgb_importance = self.xgb_model.named_steps['model'].feature_importances_
            
            return {
                'random_forest': rf_importance,
                'xgboost': xgb_importance,
                'ensemble': self.rf_weight * rf_importance + self.xgb_weight * xgb_importance
            }
        except Exception as e:
            logging.warning(f"Could not extract feature importance: {e}")
            return {}


class TrainDwellTimePredictor:
    """Main class for train dwell time prediction model."""
    
    def __init__(self, config: TrainDwellTimeConfig):
        self.config = config
        self.preprocessor = None
        self.ensemble_model = None
        self.is_trained = False
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('train_model.log')
            ]
        )
    
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load and validate the dataset."""
        data_path = Path(data_dir)
        
        try:
            # Load main dataset
            live_trains_path = data_path / "Live_Trains.csv"
            if not live_trains_path.exists():
                raise FileNotFoundError(f"Live_Trains.csv not found in {data_dir}")
            
            df = pd.read_csv(live_trains_path)
            logging.info(f"Loaded dataset with shape: {df.shape}")
            
            # Validate required columns
            missing_cols = set(self.config.features + [self.config.target]) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for model training."""
        try:
            # Extract features and target
            X = df[self.config.features].copy()
            y = df[self.config.target].copy()
            
            # Handle missing values in numeric columns
            X[self.config.numeric_cols] = X[self.config.numeric_cols].fillna(
                X[self.config.numeric_cols].median()
            )
            
            # Handle missing values in categorical columns
            for col in self.config.categorical_cols:
                X[col] = X[col].astype(str).fillna('Unknown')
            
            # Remove any rows with missing target values
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            logging.info(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
            logging.info(f"Target statistics: mean={y.mean():.2f}, std={y.std():.2f}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}")
            raise
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create the data preprocessing pipeline."""
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.config.numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.config.categorical_cols)
            ],
            remainder='drop'
        )
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the ensemble model."""
        try:
            logging.info("Starting model training...")
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Create preprocessor
            self.preprocessor = self.create_preprocessor()
            
            # Train Random Forest
            logging.info("Training Random Forest...")
            rf_pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(random_state=self.config.random_state))
            ])
            
            rf_random = RandomizedSearchCV(
                estimator=rf_pipeline,
                param_distributions=self.config.rf_params,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
                scoring="neg_mean_squared_error",
                random_state=self.config.random_state,
                verbose=1
            )
            
            rf_random.fit(X_train, y_train)
            logging.info(f"Best RF params: {rf_random.best_params_}")
            
            # Train XGBoost
            logging.info("Training XGBoost...")
            xgb_pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", XGBRegressor(
                    objective='reg:squarederror',
                    random_state=self.config.random_state
                ))
            ])
            
            xgb_random = RandomizedSearchCV(
                estimator=xgb_pipeline,
                param_distributions=self.config.xgb_params,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
                scoring="neg_mean_squared_error",
                random_state=self.config.random_state,
                verbose=1
            )
            
            xgb_random.fit(X_train, y_train)
            logging.info(f"Best XGB params: {xgb_random.best_params_}")
            
            # Create ensemble model
            self.ensemble_model = EnsembleModel(
                rf_random.best_estimator_,
                xgb_random.best_estimator_,
                self.config.rf_weight,
                self.config.xgb_weight
            )
            
            # Evaluate models
            self._evaluate_models(X_test, y_test, rf_random.best_estimator_, 
                                xgb_random.best_estimator_)
            
            self.is_trained = True
            logging.info("Model training completed successfully!")
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise
    
    def _evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, 
                        rf_model, xgb_model) -> None:
        """Evaluate and log model performance."""
        try:
            # Get predictions
            y_pred_rf = rf_model.predict(X_test)
            y_pred_xgb = xgb_model.predict(X_test)
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            
            # Calculate metrics
            models = {
                'Random Forest': y_pred_rf,
                'XGBoost': y_pred_xgb,
                'Ensemble': y_pred_ensemble
            }
            
            logging.info("Model Performance:")
            logging.info("-" * 50)
            
            for name, predictions in models.items():
                rmse = np.sqrt(mse(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                logging.info(f"{name}:")
                logging.info(f"  RMSE: {rmse:.4f}")
                logging.info(f"  MAE:  {mae:.4f}")
                logging.info(f"  R²:   {r2:.4f}")
                logging.info("")
                
        except Exception as e:
            logging.error(f"Model evaluation failed: {e}")
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            joblib.dump(self.ensemble_model, output_path)
            
            # Save configuration
            config_path = output_dir / "model_config.json"
            with open(config_path, 'w') as f:
                json.dump(vars(self.config), f, indent=2, default=str)
            
            logging.info(f"Model saved to: {output_path}")
            logging.info(f"Config saved to: {config_path}")
            
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            self.ensemble_model = joblib.load(model_path)
            self.is_trained = True
            logging.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        try:
            return self.ensemble_model.predict(X)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model for predicting train dwell times"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing the CSV data files (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save the trained model (default: ./models)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='ensemble_dwell_time_model.pkl',
        help='Name of the output model file (default: ensemble_dwell_time_model.pkl)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config = TrainDwellTimeConfig(args.config)
        
        # Initialize predictor
        predictor = TrainDwellTimePredictor(config)
        
        # Load and preprocess data
        df = predictor.load_data(args.data_dir)
        X, y = predictor.preprocess_data(df)
        
        # Train model
        predictor.train_model(X, y)
        
        # Save model
        output_path = Path(args.output_dir) / args.model_name
        predictor.save_model(str(output_path))
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()