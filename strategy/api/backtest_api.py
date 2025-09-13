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
                # Generate sample data for demo
                historical_data = self.backtest_service.generate_sample_data(500)
            
            result = self.backtest_service.run_backtest(historical_data, self.strategy)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def quick_backtest(self):
        """Quick backtest with sample data"""
        try:
            # Generate sample data
            sample_data = self.backtest_service.generate_sample_data(300)
            
            # Run backtest
            result = self.backtest_service.run_backtest(sample_data, self.strategy)
            
            return jsonify({
                'backtest_summary': result,
                'message': 'Quick backtest completed'
            })
            
        except Exception as e:
            logger.error(f"Quick backtest error: {e}")
            return jsonify({'error': str(e)}), 500