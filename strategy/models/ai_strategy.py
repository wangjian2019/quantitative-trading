"""
AI Trading Strategy Model
Author: Alvin
Description: Advanced AI strategy with ensemble learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from utils.feature_engineering import FeatureEngineer
from utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class AITradingStrategy:
    """
    Advanced AI Trading Strategy - Strategy Pattern
    Uses ensemble machine learning for trading signal generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model', {})
        self.trading_config = config.get('trading', {})
        
        # Components - Dependency Injection
        self.feature_engineer = FeatureEngineer(config)
        self.technical_indicators = TechnicalIndicators()
        
        # Model management
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        self.is_trained = False
        
        # Ensemble configuration
        self.base_models = self._create_base_models()
        self.model_weights = self.model_config.get('ensemble_weights', {
            'rf': 0.4, 'gb': 0.4, 'lr': 0.2
        })
        
        # Create model save directory
        os.makedirs(self.model_config.get('save_path', 'models'), exist_ok=True)
    
    def _create_base_models(self) -> Dict[str, Any]:
        """Factory method for creating base models"""
        return {
            'rf': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            ),
            'lr': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='liblinear'
            )
        }
    
    def generate_signal(self, current_data: Dict, indicators: Dict, history: List[Dict]) -> Dict[str, Any]:
        """
        Generate trading signal - Template Method Pattern
        """
        try:
            # Use simple strategy if models not trained
            if not self.is_trained:
                return self._simple_strategy(current_data, indicators)
            
            # Prepare features using feature engineer
            features = self.feature_engineer.prepare_features(current_data, indicators, history)
            
            # Convert to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features
            feature_scaled = self.scalers['main'].transform(feature_array)
            
            # Get ensemble prediction
            prediction_result = self._get_ensemble_prediction(feature_scaled)
            
            # Generate explanation
            reason = self._generate_explanation(features, prediction_result['action'], prediction_result['confidence'])
            
            return {
                'action': prediction_result['action'],
                'confidence': float(prediction_result['confidence']),
                'reason': reason,
                'metadata': {
                    'model_predictions': prediction_result['individual_predictions'],
                    'ensemble_probabilities': prediction_result['probabilities'],
                    'key_features': self._get_key_features(features),
                    'market_regime': self._detect_market_regime(features)
                }
            }
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._simple_strategy(current_data, indicators)
    
    def _get_ensemble_prediction(self, feature_scaled: np.ndarray) -> Dict[str, Any]:
        """Get ensemble prediction from all models"""
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(feature_scaled)[0]
            prob = model.predict_proba(feature_scaled)[0]
            
            predictions[model_name] = pred
            probabilities[model_name] = prob.tolist()
        
        # Ensemble prediction using weighted voting
        ensemble_prob = np.zeros(3)  # 3 classes: SELL, HOLD, BUY
        
        for model_name, prob in probabilities.items():
            weight = self.model_weights.get(model_name, 1.0)
            ensemble_prob += weight * np.array(prob)
        
        # Normalize probabilities
        ensemble_prob = ensemble_prob / np.sum(ensemble_prob)
        
        # Get final prediction
        final_prediction = np.argmax(ensemble_prob)
        confidence = np.max(ensemble_prob)
        
        # Convert to action
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        action = action_map[final_prediction]
        
        return {
            'action': action,
            'confidence': confidence,
            'individual_predictions': predictions,
            'probabilities': ensemble_prob.tolist()
        }
    
    def train_models(self, historical_data: List[Dict]) -> bool:
        """
        Train ensemble models - Template Method Pattern
        """
        try:
            min_data = self.model_config.get('min_training_data', 100)
            if len(historical_data) < min_data:
                logger.warning(f"Insufficient historical data: {len(historical_data)} < {min_data}")
                return False
            
            # Prepare training data
            X, y = self._prepare_training_data(historical_data)
            
            if len(X) == 0:
                logger.error("No training data prepared")
                return False
            
            # Train models
            success = self._train_ensemble_models(X, y)
            
            if success and self.model_config.get('auto_save', True):
                self.save_models()
            
            return success
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def _prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        features_list = []
        labels = []
        
        for i in range(50, len(historical_data) - 1):
            current = historical_data[i]
            history = historical_data[:i+1]
            
            # Calculate indicators for this point
            ohlcv_subset = history[-50:]
            indicators = self.technical_indicators.calculate_all_indicators(ohlcv_subset)
            indicators['current_price'] = current['close']
            
            # Prepare features
            features = self.feature_engineer.prepare_features(current, indicators, history[-20:])
            
            # Create label based on future price movement
            future_price = historical_data[i + 1]['close']
            current_price = current['close']
            price_change = (future_price - current_price) / current_price
            threshold = self.trading_config.get('price_change_threshold', 0.015)
            
            # Multi-class classification
            if price_change > threshold:
                label = 2  # BUY
            elif price_change < -threshold:
                label = 0  # SELL
            else:
                label = 1  # HOLD
            
            features_list.append(list(features.values()))
            labels.append(label)
        
        # Store feature columns
        if features_list:
            sample_features = self.feature_engineer.prepare_features(
                historical_data[50], 
                self.technical_indicators.calculate_all_indicators(historical_data[50:100]),
                historical_data[30:50]
            )
            self.feature_columns = list(sample_features.keys())
        
        return np.array(features_list), np.array(labels)
    
    def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train all models in the ensemble"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            for model_name, model in self.base_models.items():
                logger.info(f"Training model: {model_name}")
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[model_name] = model
                self.model_performance[model_name] = accuracy
                
                logger.info(f"{model_name} accuracy: {accuracy:.3f}")
            
            # Store scaler
            self.scalers['main'] = scaler
            self.is_trained = True
            
            logger.info("Ensemble model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
            return False
    
    def _simple_strategy(self, current_data: Dict, indicators: Dict) -> Dict[str, Any]:
        """Simple fallback strategy when ML models are not available"""
        try:
            rsi = indicators.get('RSI', 50)
            ma5 = indicators.get('MA5', 0)
            ma20 = indicators.get('MA20', 0)
            current_price = current_data.get('close', 0)
            
            if rsi < 30 and current_price > ma5:
                return {
                    'action': 'BUY',
                    'confidence': 0.6,
                    'reason': 'RSI oversold with price above MA5 (fallback strategy)'
                }
            elif rsi > 70 and current_price < ma20:
                return {
                    'action': 'SELL',
                    'confidence': 0.6,
                    'reason': 'RSI overbought with price below MA20 (fallback strategy)'
                }
            else:
                return {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'No clear signal from fallback strategy'
                }
                
        except Exception as e:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': f'Strategy error: {str(e)}'
            }
    
    def _generate_explanation(self, features: Dict, action: str, confidence: float) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Trend analysis
        if features.get('ma_slope', 0) > 0.01:
            explanations.append("Strong upward trend")
        elif features.get('ma_slope', 0) < -0.01:
            explanations.append("Strong downward trend")
        
        # RSI condition
        if features.get('rsi_oversold', 0):
            explanations.append("RSI oversold condition")
        elif features.get('rsi_overbought', 0):
            explanations.append("RSI overbought condition")
        
        # Volume analysis
        if features.get('high_volume', 0):
            explanations.append("High volume activity")
        
        # Momentum
        if features.get('momentum_5', 0) > 0.02:
            explanations.append("Strong positive momentum")
        elif features.get('momentum_5', 0) < -0.02:
            explanations.append("Strong negative momentum")
        
        # Signal strength
        if features.get('bullish_strength', 0) > 0.7:
            explanations.append("Multiple bullish signals")
        elif features.get('bearish_strength', 0) > 0.7:
            explanations.append("Multiple bearish signals")
        
        if not explanations:
            explanations.append("Mixed signals from technical indicators")
        
        return f"{action} signal with {confidence:.1%} confidence: " + ", ".join(explanations)
    
    def _get_key_features(self, features: Dict) -> Dict[str, float]:
        """Get most important features"""
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_features[:5])
    
    def _detect_market_regime(self, features: Dict) -> str:
        """Detect current market regime"""
        if features.get('volatility', 0) > 0.02:
            return "High Volatility"
        elif features.get('trend_strength', 0) > 0.015:
            return "Trending Market"
        elif features.get('volume_ratio', 1) < 0.7:
            return "Low Volume"
        else:
            return "Normal Market"
    
    def save_models(self) -> bool:
        """Save trained models to disk"""
        try:
            save_path = self.model_config.get('save_path', 'models')
            
            for model_name, model in self.models.items():
                joblib.dump(model, f'{save_path}/{model_name}_model.pkl')
            
            joblib.dump(self.scalers['main'], f'{save_path}/scaler.pkl')
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'is_trained': self.is_trained,
                'config_snapshot': {
                    'ensemble_weights': self.model_weights,
                    'price_change_threshold': self.trading_config.get('price_change_threshold', 0.015),
                    'min_confidence': self.trading_config.get('min_confidence', 0.6)
                }
            }
            
            with open(f'{save_path}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved successfully to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            save_path = self.model_config.get('save_path', 'models')
            metadata_path = f'{save_path}/metadata.json'
            
            if not os.path.exists(metadata_path):
                logger.info(f"No existing models found at {save_path}")
                return False
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.model_performance = metadata['model_performance']
            self.is_trained = metadata['is_trained']
            
            # Load models
            for model_name in self.base_models.keys():
                model_path = f'{save_path}/{model_name}_model.pkl'
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
            
            # Load scaler
            scaler_path = f'{save_path}/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scalers['main'] = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            
            logger.info(f"Models loaded successfully from {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'performance': self.model_performance,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns[:10] if self.feature_columns else [],
            'model_weights': self.model_weights,
            'config': {
                'ensemble_weights': self.model_weights,
                'min_training_data': self.model_config.get('min_training_data', 100)
            }
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        feature_importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance[model_name] = dict(zip(self.feature_columns, importance))
        
        # Calculate average importance
        if feature_importance:
            avg_importance = {}
            for feature in self.feature_columns:
                scores = [feature_importance[model][feature] for model in feature_importance.keys()
                         if feature in feature_importance[model]]
                avg_importance[feature] = np.mean(scores) if scores else 0
            
            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'average_importance': dict(sorted_importance),
                'model_specific': feature_importance,
                'top_features': sorted_importance[:10]
            }
        
        return {'error': 'No feature importance available'}
    
    def retrain_models(self, new_data: List[Dict]) -> bool:
        """Retrain models with new data"""
        try:
            if len(new_data) < 200:
                logger.warning(f"Insufficient data for retraining: {len(new_data)} < 200")
                return False
            
            # Clear existing models
            self.models = {}
            self.scalers = {}
            self.is_trained = False
            
            # Retrain with new data
            success = self.train_models(new_data)
            
            if success:
                logger.info("Models retrained successfully")
            else:
                logger.error("Model retraining failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Retraining error: {e}")
            return False
