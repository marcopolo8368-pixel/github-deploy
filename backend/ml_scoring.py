"""
ML-Based Scoring with XGBoost
Learns optimal feature weights from historical backtesting data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pickle
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import XGBoost, fall back gracefully if not available
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, classification_report
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost scikit-learn")


# 30 features: 21 original + 9 advanced context features
ML_FEATURE_NAMES = [
    # Original 21
    "rsi", "bb_breached", "bb_depth", "macd_histogram", "macd_bearish_cross",
    "volume_ratio", "volume_spike", "pct_drop", "at_support", "pct_above_support",
    "rr_ratio", "atr_pct", "sma50_dist", "sma200_dist", "golden_cross",
    "above_sma_200", "roc", "candle_green", "entropy", "hurst", "dip_quality",
    # Advanced context features
    "momentum_5d", "momentum_10d", "momentum_20d",
    "consecutive_red_days", "bounce_strength",
    "rsi_slope", "macd_slope", "vol_regime", "vwap_dist",
]


class MLScoringModel:
    """
    Machine Learning-based scoring system using XGBoost
    
    Learns optimal feature weights from historical backtest data.
    Trained on 30 features extracted from IndicatorResult + raw price data.
    Target: Win/Loss at 20 days from historical trades.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to rule-based scoring")
            self.model = None
            self.scaler = None
            self.feature_names = []
        else:
            self.model = None
            self.scaler = StandardScaler()
            self.feature_names = ML_FEATURE_NAMES
            
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load pre-trained XGBoost model AND scaler"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model')
                self.scaler = data.get('scaler', StandardScaler())
            logger.info(f"Loaded ML scoring model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            self.model = None
    
    def save_model(self, model_path: str):
        """Save trained XGBoost model AND scaler together"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            logger.info(f"Saved ML scoring model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def train_from_backtest_data(
        self,
        backtest_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "won_20d",
        tune_hyperparams: bool = True,
    ):
        """
        Train XGBoost model with optional hyperparameter tuning.
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping training")
            return
        
        try:
            # Prepare data
            X = backtest_df[feature_cols].fillna(0)
            y = backtest_df[target_col]
            
            logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features")
            logger.info(f"Target win rate: {y.mean()*100:.1f}%")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if tune_hyperparams:
                logger.info("\n🔧 Running hyperparameter tuning...")
                base_model = xgb.XGBClassifier(
                    eval_metric='logloss',
                    random_state=42,
                )
                
                param_dist = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 4, 5, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.3, 0.5],
                    'reg_alpha': [0, 0.01, 0.1, 1.0],
                    'reg_lambda': [1, 1.5, 2.0, 3.0],
                }
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = RandomizedSearchCV(
                    base_model, param_dist,
                    n_iter=60,
                    cv=cv,
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0,
                )
                search.fit(X_scaled, y)
                
                best_model = search.best_estimator_
                logger.info(f"Best params: {search.best_params_}")
                logger.info(f"Best CV Accuracy: {search.best_score_:.3f}")
            else:
                best_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
                best_model.fit(X_scaled, y)
            
            # 5-fold cross-validation with best model
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='accuracy')
            logger.info(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(best_model, cv=3, method='isotonic')
            calibrated_model.fit(X_scaled, y)
            self.model = calibrated_model
            
            # Evaluate
            y_pred = self.model.predict(X_scaled)
            y_prob = self.model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_prob)
            
            logger.info(f"Train Acc={self.model.score(X_scaled, y):.3f}, AUC={auc:.3f}")
            logger.info(f"\n{classification_report(y, y_pred, target_names=['Loss', 'Win'])}")
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": importances
                }).sort_values("importance", ascending=False)
                
                logger.info("\nFeature Importance (Top 15):")
                for idx, row in feature_importance_df.head(15).iterrows():
                    logger.info(f"  {row['feature']:22s}: {row['importance']:.4f}")
        
        except Exception as e:
            logger.error(f"Error training ML model: {e}", exc_info=True)
            self.model = None
    
    def predict_score(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict dip score using trained ML model
        
        Returns:
            Tuple of (predicted_score: 0-100, win_probability: 0-1)
        """
        if self.model is None:
            return 50.0, 0.5
        
        try:
            feature_vector = np.array([
                features.get(fname, 0) for fname in self.feature_names
            ]).reshape(1, -1)
            
            feature_scaled = self.scaler.transform(feature_vector)
            probabilities = self.model.predict_proba(feature_scaled)[0]
            win_prob = probabilities[1]
            score = win_prob * 100
            
            return round(score, 1), round(win_prob, 3)
        
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 50.0, 0.5
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model"""
        if self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'calibrated_classifiers_'):
                base_estimator = self.model.calibrated_classifiers_[0].estimator
                if hasattr(base_estimator, 'feature_importances_'):
                    importances = base_estimator.feature_importances_
                    return {
                        name: float(importance)
                        for name, importance in zip(self.feature_names, importances)
                    }
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
        
        return {}


if __name__ == "__main__":
    print("ML Scoring System initialized")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"Features: {len(ML_FEATURE_NAMES)}")
