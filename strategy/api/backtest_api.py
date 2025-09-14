"""
Backtest API Controller
Author: Alvin
Description: RESTful API for backtesting operations
"""

from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

class BacktestAPI:
    """
    Backtest API Controller - Controller Pattern
    """
    
    def __init__(self, backtest_service, strategy):
        self.backtest_service = backtest_service
        self.strategy = strategy
        self.blueprint = Blueprint('backtest', __name__)
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        self.blueprint.add_url_rule('/backtest/run', 'run_backtest', 
                                   self.run_backtest, methods=['POST'])
        self.blueprint.add_url_rule('/backtest/quick', 'quick_backtest', 
                                   self.quick_backtest, methods=['POST'])
    
    def run_backtest(self):
        """Run full backtest"""
        try:
            data = request.get_json() or {}
            historical_data = data.get('historical_data', [])
            
            if len(historical_data) < 50:
                # 不再生成模拟数据，返回错误
                return jsonify({
                    'error': 'Insufficient historical data provided',
                    'message': 'At least 50 data points required for backtesting',
                    'provided_data_points': len(historical_data),
                    'required_data_points': 50
                }), 400
            
            result = self.backtest_service.run_backtest(historical_data, self.strategy)
            result['data_source'] = 'Real historical data'
            result['data_points_used'] = len(historical_data)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def quick_backtest(self):
        """Quick backtest - requires real historical data"""
        return jsonify({
            'error': 'Quick backtest disabled',
            'message': 'Only real historical data backtesting is supported',
            'use_endpoint': '/api/backtest/run',
            'required_data': 'Provide historical_data array with at least 50 data points'
        }), 400