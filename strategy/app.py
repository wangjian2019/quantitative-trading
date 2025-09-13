"""
AI Trading Strategy Application
Author: Alvin
Description: Main Flask application with modular architecture
"""

from flask import Flask, request, jsonify
import logging
import os
from datetime import datetime

# Import modular components
from config import config
from models.ai_strategy import AITradingStrategy
from services.backtest_service import BacktestService
from api.signal_api import SignalAPI
from api.backtest_api import BacktestAPI
from api.model_api import ModelAPI

# Configure logging
log_config = config.get_logging_config()
import logging.handlers

os.makedirs('logs', exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    log_config['file_path'], 
    maxBytes=log_config['max_file_size'],
    backupCount=log_config['backup_count']
)

logging.basicConfig(
    level=getattr(logging, log_config['level']),
    format=log_config['format'],
    handlers=[
        logging.StreamHandler(),
        file_handler
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize services - Dependency Injection Pattern
strategy = AITradingStrategy(config.config)
backtest_service = BacktestService(config.config)

# Initialize API controllers - Controller Pattern
signal_api = SignalAPI(strategy)
backtest_api = BacktestAPI(backtest_service, strategy)
model_api = ModelAPI(strategy)

# Register API routes
app.register_blueprint(signal_api.blueprint, url_prefix='/api')
app.register_blueprint(backtest_api.blueprint, url_prefix='/api')
app.register_blueprint(model_api.blueprint, url_prefix='/api')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': strategy.is_trained,
        'models_available': list(strategy.models.keys()),
        'service': 'AI Trading Strategy Service',
        'author': 'Alvin',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/info', methods=['GET'])
def service_info():
    """Service information endpoint"""
    return jsonify({
        'name': 'AI Trading Strategy Service',
        'version': '2.0.0',
        'author': 'Alvin',
        'description': 'Advanced AI-powered trading strategy service with ensemble learning',
        'architecture': 'Modular microservice with design patterns',
        'features': [
            'Multi-model ensemble learning',
            'Advanced feature engineering',
            'Comprehensive backtesting',
            'Real-time signal generation',
            'Performance analytics',
            'Model persistence'
        ],
        'endpoints': {
            'signals': '/api/signals/*',
            'backtest': '/api/backtest/*',
            'models': '/api/models/*',
            'health': '/health',
            'info': '/info'
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation',
        'available_endpoints': ['/health', '/info', '/api/signals', '/api/backtest', '/api/models']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check the logs for more details'
    }), 500

def initialize_application():
    """Initialize application components"""
    logger.info("Initializing AI Trading Strategy Service...")
    
    # Try to load existing models
    if strategy.load_models():
        logger.info("✓ Existing models loaded successfully")
        logger.info(f"✓ Model performance: {strategy.model_performance}")
    else:
        logger.info("ℹ No existing models found - will train on first use")
    
    logger.info("Service initialization completed")

if __name__ == '__main__':
    print("="*60)
    print("🚀 AI Trading Strategy Service v2.0")
    print("Author: Alvin")
    print("Architecture: Modular microservice with design patterns")
    print("="*60)
    
    # Initialize application
    initialize_application()
    
    # Print service information
    flask_config = config.get_flask_config()
    print(f"🌐 Starting Flask server on {flask_config['host']}:{flask_config['port']}")
    print(f"🔧 Debug mode: {flask_config['debug']}")
    print("📊 Service endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /info                - Service information")
    print("  POST /api/signals/generate - Generate trading signal")
    print("  POST /api/backtest/run     - Run backtest analysis")
    print("  POST /api/models/train     - Train ML models")
    print("  GET  /api/models/info      - Model information")
    print("="*60)
    
    # Start the Flask application
    try:
        app.run(
            host=flask_config['host'], 
            port=flask_config['port'], 
            debug=flask_config['debug'], 
            threaded=flask_config['threaded']
        )
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        print(f"❌ Service startup failed: {e}")
    finally:
        logger.info("AI Trading Strategy Service shutdown completed")
