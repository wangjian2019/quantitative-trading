"""
Model API Controller
Author: Alvin
Description: RESTful API for model management
"""

from flask import Blueprint, request, jsonify
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelAPI:
    """
    Model API Controller - Controller Pattern
    Handles model-related API endpoints
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.blueprint = Blueprint('models', __name__)
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        self.blueprint.add_url_rule('/models/info', 'model_info', 
                                   self.get_model_info, methods=['GET'])
        self.blueprint.add_url_rule('/models/train', 'train_models', 
                                   self.train_models, methods=['POST'])
        self.blueprint.add_url_rule('/models/retrain', 'retrain_models', 
                                   self.retrain_models, methods=['POST'])
        self.blueprint.add_url_rule('/models/feature-importance', 'feature_importance', 
                                   self.get_feature_importance, methods=['GET'])
    
    def get_model_info(self):
        """
        Get model information
        GET /api/models/info
        """
        try:
            model_info = self.strategy.get_model_info()
            model_info['service_version'] = '2.0.0'
            return jsonify(model_info)
            
        except Exception as e:
            logger.error(f"Model info error: {e}")
            return jsonify({
                'error': f'Error getting model info: {str(e)}'
            }), 500
    
    def train_models(self):
        """
        Train models with provided data
        POST /api/models/train
        """
        try:
            data = request.get_json()
            
            if not data or 'historical_data' not in data:
                return jsonify({'error': 'No historical data provided'}), 400
            
            historical_data = data['historical_data']
            
            if len(historical_data) < 100:
                return jsonify({
                    'success': False,
                    'message': 'Insufficient historical data, need at least 100 records',
                    'provided': len(historical_data),
                    'required': 100
                }), 400
            
            # Train models
            success = self.strategy.train_models(historical_data)
            
            return jsonify({
                'success': success,
                'message': 'Model training completed' if success else 'Model training failed',
                'performance': self.strategy.model_performance if success else {},
                'model_info': self.strategy.get_model_info() if success else {},
                'service_version': '2.0.0'
            })
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return jsonify({
                'success': False,
                'message': f'Training error: {str(e)}'
            }), 500
    
    def retrain_models(self):
        """
        Retrain models with new data
        POST /api/models/retrain
        """
        try:
            data = request.get_json()
            
            if not data or 'new_data' not in data:
                return jsonify({'error': 'No new data provided'}), 400
            
            new_data = data['new_data']
            
            if len(new_data) < 200:
                return jsonify({
                    'success': False,
                    'message': 'Need at least 200 data points for retraining',
                    'provided': len(new_data),
                    'required': 200
                }), 400
            
            # Retrain models
            success = self.strategy.retrain_models(new_data)
            
            return jsonify({
                'success': success,
                'message': 'Models retrained successfully' if success else 'Retraining failed',
                'new_performance': self.strategy.model_performance if success else {},
                'model_info': self.strategy.get_model_info() if success else {},
                'service_version': '2.0.0'
            })
            
        except Exception as e:
            logger.error(f"Retraining error: {e}")
            return jsonify({
                'success': False,
                'message': f'Retraining error: {str(e)}'
            }), 500
    
    def get_feature_importance(self):
        """
        Get feature importance from trained models
        GET /api/models/feature-importance
        """
        try:
            importance_data = self.strategy.get_feature_importance()
            
            if 'error' in importance_data:
                return jsonify(importance_data), 400
            
            importance_data['service_version'] = '2.0.0'
            return jsonify(importance_data)
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return jsonify({
                'error': f'Error getting feature importance: {str(e)}'
            }), 500
