"""
Signal API Controller
Author: Alvin
Description: RESTful API for trading signals
"""

from flask import Blueprint, request, jsonify
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SignalAPI:
    """
    Signal API Controller - Controller Pattern
    Handles signal-related API endpoints
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.blueprint = Blueprint('signals', __name__)
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        self.blueprint.add_url_rule('/signals/generate', 'generate_signal', 
                                   self.generate_signal, methods=['POST'])
        self.blueprint.add_url_rule('/signals/batch', 'batch_signals', 
                                   self.batch_signals, methods=['POST'])
    
    def generate_signal(self):
        """
        Generate trading signal endpoint
        POST /api/signals/generate
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Validate required fields
            required_fields = ['current_data', 'indicators', 'history']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Extract data
            current_data = data['current_data']
            indicators = data['indicators']
            history = data['history']
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Generate signal
            result = self.strategy.generate_signal(current_data, indicators, history)
            
            # Add metadata
            result['symbol'] = symbol
            result['timestamp'] = current_data.get('timestamp')
            result['service_version'] = '2.0.0'
            
            logger.info(f"Signal generated for {symbol}: {result['action']} "
                       f"(confidence: {result['confidence']:.2f})")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return jsonify({
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': f'Service error: {str(e)}',
                'error': True
            }), 500
    
    def batch_signals(self):
        """
        Generate batch signals for multiple symbols
        POST /api/signals/batch
        """
        try:
            data = request.get_json()
            
            if not data or 'requests' not in data:
                return jsonify({'error': 'No batch requests provided'}), 400
            
            requests_data = data['requests']
            if not isinstance(requests_data, list):
                return jsonify({'error': 'Requests must be a list'}), 400
            
            results = []
            
            for req_data in requests_data:
                try:
                    # Generate signal for each request
                    current_data = req_data['current_data']
                    indicators = req_data['indicators']
                    history = req_data['history']
                    symbol = req_data.get('symbol', 'UNKNOWN')
                    
                    result = self.strategy.generate_signal(current_data, indicators, history)
                    result['symbol'] = symbol
                    result['timestamp'] = current_data.get('timestamp')
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Batch signal error for {req_data.get('symbol', 'UNKNOWN')}: {e}")
                    results.append({
                        'symbol': req_data.get('symbol', 'UNKNOWN'),
                        'action': 'HOLD',
                        'confidence': 0.0,
                        'reason': f'Error: {str(e)}',
                        'error': True
                    })
            
            return jsonify({
                'results': results,
                'total_processed': len(results),
                'successful': len([r for r in results if not r.get('error', False)]),
                'failed': len([r for r in results if r.get('error', False)])
            })
            
        except Exception as e:
            logger.error(f"Batch signals error: {e}")
            return jsonify({
                'error': f'Batch processing failed: {str(e)}'
            }), 500
